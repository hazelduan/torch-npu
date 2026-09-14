import pytest
import torch
import torch.nn.functional as F
from torch import nn

from torch_npu.utils.hifloat8_train import hifloat8_linear as hif8


def _install_cpu_kernel(monkeypatch):
    def quantize(tensor, _kind):
        return tensor.detach().clone(), torch.ones(1, dtype=torch.float32)

    def mm(lhs, rhs, _lhs_scale, _rhs_scale):
        return lhs @ rhs

    monkeypatch.setattr(hif8, "_quantize", quantize)
    monkeypatch.setattr(hif8, "_quantized_mm", mm)


def _install_cpu_grouped_kernel(monkeypatch):
    calls = []

    def quantize(tensor, _kind):
        return tensor.detach().clone(), torch.ones(1, dtype=torch.float32)

    def grouped_matmul(
        inputs,
        weights,
        *,
        scale,
        per_token_scale,
        group_list,
        group_type,
        **kwargs,
    ):
        lhs, rhs = inputs[0], weights[0]
        calls.append(
            {
                "group_type": group_type,
                "lhs_stride": lhs.stride(),
                "lhs_is_contiguous": lhs.is_contiguous(),
                "scale_shape": scale[0].shape,
                "per_token_scale_shape": per_token_scale[0].shape,
            }
        )

        offsets = group_list.tolist()
        if group_type == 0:
            output = lhs.new_empty((lhs.shape[0], rhs.shape[-1]))
            start = 0
            for expert, end in enumerate(offsets):
                output[start:end] = lhs[start:end] @ rhs[expert]
                start = end
        elif group_type == 2:
            output = []
            start = 0
            for end in offsets:
                output.append(lhs[:, start:end] @ rhs[start:end])
                start = end
            output = torch.stack(output)
        else:
            raise AssertionError(f"Unexpected group_type: {group_type}")
        return [output]

    monkeypatch.setattr(hif8, "_ensure_bf16_or_fp16", lambda tensor: tensor)
    monkeypatch.setattr(hif8, "_grouped_quantize", quantize)
    monkeypatch.setattr(hif8.torch_npu, "hifloat8", torch.uint8, raising=False)
    monkeypatch.setattr(hif8.torch_npu, "npu_grouped_matmul", grouped_matmul, raising=False)
    return calls


def _reference_grouped_mm(input_value, weight_value, group_list):
    outputs = []
    start = 0
    for expert, end in enumerate(group_list.tolist()):
        outputs.append(input_value[start:end] @ weight_value[expert])
        start = end
    return torch.cat(outputs)


def test_forward_dx_dw_match_linear_with_mocked_quantization(monkeypatch):
    _install_cpu_kernel(monkeypatch)
    torch.manual_seed(42)
    input_value = torch.randn(2, 3, 4)
    weight_value = torch.randn(8, 4)
    upstream = torch.randn(2, 3, 8)

    reference_input = input_value.clone().requires_grad_()
    reference_weight = weight_value.clone().requires_grad_()
    (F.linear(reference_input, reference_weight) * upstream).sum().backward()

    test_input = input_value.clone().requires_grad_()
    test_weight = weight_value.clone().requires_grad_()
    output = hif8._MatmulWithHiFloat8.apply(test_input, test_weight)
    (output * upstream).sum().backward()

    torch.testing.assert_close(output, F.linear(input_value, weight_value))
    torch.testing.assert_close(test_input.grad, reference_input.grad)
    torch.testing.assert_close(test_weight.grad, reference_weight.grad)


def test_frozen_weight_skips_dw(monkeypatch):
    _install_cpu_kernel(monkeypatch)
    hif8.reset_hifloat8_op_counts()
    input_value = torch.randn(2, 4, requires_grad=True)
    weight = torch.randn(8, 4, requires_grad=False)

    hif8._MatmulWithHiFloat8.apply(input_value, weight).sum().backward()

    assert input_value.grad is not None
    assert weight.grad is None
    counts = hif8.get_hifloat8_op_counts()
    assert counts["forward"] == 1
    assert counts["backward_dx"] == 1
    assert counts.get("backward_dw", 0) == 0


@pytest.mark.parametrize(
    "offsets",
    (
        (2, 2, 5),
        (0, 2, 5),
        (2, 5, 5),
        (1, 1, 1, 5),
        (0, 0, 5),
    ),
)
def test_grouped_forward_dx_dw_support_empty_expert_and_split_k_view(monkeypatch, offsets):
    calls = _install_cpu_grouped_kernel(monkeypatch)
    hif8.reset_hifloat8_op_counts()
    torch.manual_seed(42)
    group_list = torch.tensor(offsets, dtype=torch.int64)
    input_value = torch.randn(5, 4)
    weight_value = torch.randn(len(offsets), 4, 6)
    upstream = torch.randn(5, 6)

    reference_input = input_value.clone().requires_grad_()
    reference_weight = weight_value.clone().requires_grad_()
    reference_output = _reference_grouped_mm(reference_input, reference_weight, group_list)
    (reference_output * upstream).sum().backward()

    test_input = input_value.clone().requires_grad_()
    test_weight = weight_value.clone().requires_grad_()
    output = hif8.hifloat8_grouped_mm(test_input, test_weight, group_list)
    (output * upstream).sum().backward()

    torch.testing.assert_close(output, reference_output)
    torch.testing.assert_close(test_input.grad, reference_input.grad)
    torch.testing.assert_close(test_weight.grad, reference_weight.grad)
    starts = (0, *offsets[:-1])
    for expert, (start, end) in enumerate(zip(starts, offsets)):
        if start == end:
            torch.testing.assert_close(test_weight.grad[expert], torch.zeros_like(test_weight.grad[expert]))

    counts = hif8.get_hifloat8_op_counts()
    assert counts["grouped_forward"] == 1
    assert counts["grouped_backward_dx"] == 1
    assert counts["grouped_backward_dw"] == 1
    assert counts["grouped_matmul_group_type_0"] == 2
    assert counts["grouped_matmul_group_type_2"] == 1

    dw_call = calls[-1]
    assert dw_call["group_type"] == 2
    assert dw_call["lhs_stride"] == (1, input_value.shape[-1])
    assert not dw_call["lhs_is_contiguous"]
    assert dw_call["scale_shape"] == (weight_value.shape[0],)
    assert dw_call["per_token_scale_shape"] == (weight_value.shape[0],)
    assert [call["per_token_scale_shape"] for call in calls[:2]] == [
        (input_value.shape[0],),
        (input_value.shape[0],),
    ]


def test_grouped_frozen_weight_skips_group_type_2(monkeypatch):
    calls = _install_cpu_grouped_kernel(monkeypatch)
    hif8.reset_hifloat8_op_counts()
    input_value = torch.randn(4, 3, requires_grad=True)
    weight = torch.randn(2, 3, 5, requires_grad=False)
    group_list = torch.tensor((1, 4), dtype=torch.int64)

    hif8.hifloat8_grouped_mm(input_value, weight, group_list).sum().backward()

    assert input_value.grad is not None
    assert weight.grad is None
    assert [call["group_type"] for call in calls] == [0, 0]
    counts = hif8.get_hifloat8_op_counts()
    assert counts["grouped_forward"] == 1
    assert counts["grouped_backward_dx"] == 1
    assert counts.get("grouped_backward_dw", 0) == 0
    assert counts.get("grouped_matmul_group_type_2", 0) == 0


def test_grouped_all_empty_local_input_returns_zero_gradients(monkeypatch):
    calls = _install_cpu_grouped_kernel(monkeypatch)
    hif8.reset_hifloat8_op_counts()
    input_value = torch.empty(0, 3, requires_grad=True)
    weight = torch.randn(3, 3, 5, requires_grad=True)
    group_list = torch.zeros(3, dtype=torch.int64)

    output = hif8.hifloat8_grouped_mm(input_value, weight, group_list)
    output.sum().backward()

    assert output.shape == (0, 5)
    torch.testing.assert_close(input_value.grad, torch.zeros_like(input_value))
    torch.testing.assert_close(weight.grad, torch.zeros_like(weight))
    assert calls == []
    counts = hif8.get_hifloat8_op_counts()
    assert counts["grouped_empty_forward"] == 1
    assert counts["grouped_empty_backward"] == 1
    assert counts.get("grouped_matmul_group_type_0", 0) == 0
    assert counts.get("grouped_matmul_group_type_2", 0) == 0


def test_grouped_input_validation(monkeypatch):
    _install_cpu_grouped_kernel(monkeypatch)
    input_value = torch.randn(4, 3)
    weight = torch.randn(2, 3, 5)

    with pytest.raises(ValueError, match="input must be 2D"):
        hif8.hifloat8_grouped_mm(input_value.unsqueeze(0), weight, torch.tensor((2, 4)))
    with pytest.raises(ValueError, match="expert count mismatch"):
        hif8.hifloat8_grouped_mm(input_value, weight, torch.tensor((4,)))
    with pytest.raises(ValueError, match="shape mismatch"):
        hif8.hifloat8_grouped_mm(input_value, torch.randn(2, 4, 5), torch.tensor((2, 4)))
    with pytest.raises(TypeError, match="must use float16"):
        hif8.hifloat8_grouped_mm(input_value.to(torch.int32), weight, torch.tensor((2, 4)))


def test_grouped_capability_check_requires_split_k_schema(monkeypatch):
    class Schema:
        _schema = (
            "npu_grouped_matmul(scale, per_token_scale, group_list, group_type, "
            "split_item, output_dtype, group_list_type, x_dtype, weight_dtype)"
        )

    class Op:
        default = Schema()

    monkeypatch.setattr(hif8.torch_npu, "hifloat8", torch.uint8, raising=False)
    monkeypatch.setattr(hif8.torch_npu, "npu_dynamic_quant", lambda *_args, **_kwargs: None, raising=False)
    monkeypatch.setattr(hif8.torch_npu, "npu_grouped_matmul", lambda *_args, **_kwargs: None, raising=False)
    monkeypatch.setattr(hif8.torch.ops.npu, "npu_grouped_matmul", Op(), raising=False)

    assert hif8._grouped_capability_errors() == ()

    Schema._schema = Schema._schema.replace("weight_dtype", "")
    assert hif8._grouped_capability_errors() == (
        "npu_grouped_matmul schema argument 'weight_dtype'",
    )


def test_grouped_capability_check_can_skip_kernel_probe(monkeypatch):
    monkeypatch.setattr(hif8, "_grouped_capability_errors", lambda: ())
    hif8.assert_hifloat8_grouped_training_available(probe_kernel=False)

    monkeypatch.setattr(hif8, "_grouped_capability_errors", lambda: ("missing symbol",))
    with pytest.raises(RuntimeError, match="missing symbol"):
        hif8.assert_hifloat8_grouped_training_available(probe_kernel=False)


class _ToyBlock(nn.Module):

    def __init__(self):
        super().__init__()
        self.mlp = nn.ModuleDict(
            {
                "gate_proj": nn.Linear(4, 8, bias=False),
                "up_proj": nn.Linear(4, 8, bias=False),
                "down_proj": nn.Linear(8, 4, bias=False),
            }
        )
        self.self_attn = nn.ModuleDict({"q_proj": nn.Linear(4, 4, bias=False)})


def test_filtered_conversion_preserves_parameter_identity_and_keys():
    model = _ToyBlock()
    original_gate = model.mlp["gate_proj"]
    marker = object()
    original_gate.marker = marker
    hook = original_gate.register_forward_hook(lambda *_args: None)
    before_parameters = dict(model.named_parameters())
    before_state_keys = tuple(model.state_dict())

    hif8.convert_to_hifloat8_training(
        model,
        module_filter_fn=lambda _module, fqn: fqn.startswith("mlp."),
    )

    assert isinstance(model.mlp["gate_proj"], hif8.HiFloat8Linear)
    assert model.mlp["gate_proj"] is original_gate
    assert model.mlp["gate_proj"].marker is marker
    assert hook.id in model.mlp["gate_proj"]._forward_hooks
    assert isinstance(model.mlp["up_proj"], hif8.HiFloat8Linear)
    assert isinstance(model.mlp["down_proj"], hif8.HiFloat8Linear)
    assert type(model.self_attn["q_proj"]) is nn.Linear
    assert tuple(model.state_dict()) == before_state_keys
    after_parameters = dict(model.named_parameters())
    assert before_parameters.keys() == after_parameters.keys()
    assert all(after_parameters[name] is parameter for name, parameter in before_parameters.items())


def test_linear_declares_transformers_local_tp_contract():
    assert hif8.HiFloat8Linear._hf_quantized_needs_local_tp is True


def test_filtered_conversion_leaves_linear_subclasses_unchanged():
    class CustomLinear(nn.Linear):
        pass

    model = nn.Sequential(CustomLinear(4, 8, bias=False))
    hif8.convert_to_hifloat8_training(model)

    assert type(model[0]) is CustomLinear
