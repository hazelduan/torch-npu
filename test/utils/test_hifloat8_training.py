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


def test_filtered_conversion_leaves_linear_subclasses_unchanged():
    class CustomLinear(nn.Linear):
        pass

    model = nn.Sequential(CustomLinear(4, 8, bias=False))
    hif8.convert_to_hifloat8_training(model)

    assert type(model[0]) is CustomLinear
