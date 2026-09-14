"""BF16-parameter linear layers with HiFloat8 matrix multiplies.

The implementation follows the forward/dX/dW algorithm introduced by
PyTorchA commit d8f826b, while keeping quantized data as ordinary tensors.
That avoids a broad Tensor-subclass dispatch surface and makes unused LoRA
base-weight gradients cheap to skip.

The grouped path follows torchtitan-npu commit d423d259, including its
group_type=0 forward/dX and group_type=2 split-K dW scale contracts.
"""

from __future__ import annotations

from collections import Counter
from typing import Callable, Dict, Optional, Tuple

import torch
from torch import nn

import torch_npu


_HIFLOAT8_MAX = {"input": 15.0, "weight": 15.0, "grad": 224.0}
_OP_COUNTS: Counter = Counter()
_PROBED_DEVICES = set()
_PROBED_GROUPED_DEVICES = set()


def _capability_errors() -> Tuple[str, ...]:
    errors = []
    if not hasattr(torch_npu, "hifloat8"):
        errors.append("torch_npu.hifloat8")
    for name in ("npu_quantize", "npu_quant_matmul"):
        if not callable(getattr(torch_npu, name, None)):
            errors.append(f"torch_npu.{name}")

    quant_matmul = getattr(getattr(torch.ops, "npu", None), "npu_quant_matmul", None)
    schema = str(getattr(getattr(quant_matmul, "default", None), "_schema", ""))
    for argument in ("x1_dtype", "x2_dtype", "pertoken_scale"):
        if argument not in schema:
            errors.append(f"npu_quant_matmul schema argument '{argument}'")
    return tuple(errors)


def _grouped_capability_errors() -> Tuple[str, ...]:
    errors = []
    if not hasattr(torch_npu, "hifloat8"):
        errors.append("torch_npu.hifloat8")
    for name in ("npu_dynamic_quant", "npu_grouped_matmul"):
        if not callable(getattr(torch_npu, name, None)):
            errors.append(f"torch_npu.{name}")

    grouped_matmul = getattr(getattr(torch.ops, "npu", None), "npu_grouped_matmul", None)
    schema = str(getattr(getattr(grouped_matmul, "default", None), "_schema", ""))
    for argument in (
        "scale",
        "per_token_scale",
        "group_list",
        "group_type",
        "split_item",
        "output_dtype",
        "group_list_type",
        "x_dtype",
        "weight_dtype",
    ):
        if argument not in schema:
            errors.append(f"npu_grouped_matmul schema argument '{argument}'")
    return tuple(errors)


def _quantize(tensor: torch.Tensor, kind: str) -> Tuple[torch.Tensor, torch.Tensor]:
    if kind not in _HIFLOAT8_MAX:
        raise ValueError(f"Unknown HiFloat8 tensor kind: {kind!r}")
    if tensor.device.type != "npu":
        raise RuntimeError(f"HiFloat8 training requires NPU tensors, got {tensor.device}")
    if tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"HiFloat8 training requires a floating input, got {tensor.dtype}")

    tensor = tensor.contiguous()
    scale = tensor.detach().abs().amax().float().clamp_min(1e-12)
    scale = (scale / _HIFLOAT8_MAX[kind]).reshape(1)
    data = torch_npu.npu_quantize(
        tensor,
        scale,
        zero_points=None,
        dtype=torch_npu.hifloat8,
    )
    _OP_COUNTS[f"quantize_{kind}"] += 1
    return data, scale


def _quantized_mm(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    lhs_scale: torch.Tensor,
    rhs_scale: torch.Tensor,
) -> torch.Tensor:
    output = torch_npu.npu_quant_matmul(
        lhs,
        rhs,
        rhs_scale.reshape(-1),
        pertoken_scale=lhs_scale.reshape(-1),
        output_dtype=torch.bfloat16,
        x1_dtype=torch_npu.hifloat8,
        x2_dtype=torch_npu.hifloat8,
    )
    _OP_COUNTS["quant_matmul"] += 1
    return output


def _ensure_bf16_or_fp16(tensor: torch.Tensor) -> torch.Tensor:
    """Return a dtype accepted by ``npu_dynamic_quant`` pertensor mode."""
    if tensor.dtype not in (torch.float16, torch.bfloat16):
        return tensor.to(torch.bfloat16)
    return tensor


def _grouped_quantize(tensor: torch.Tensor, kind: str) -> Tuple[torch.Tensor, torch.Tensor]:
    if kind not in _HIFLOAT8_MAX:
        raise ValueError(f"Unknown HiFloat8 tensor kind: {kind!r}")
    if tensor.device.type != "npu":
        raise RuntimeError(f"HiFloat8 grouped training requires NPU tensors, got {tensor.device}")
    if tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"HiFloat8 grouped training requires a floating input, got {tensor.dtype}")

    data, scale = torch_npu.npu_dynamic_quant(
        _ensure_bf16_or_fp16(tensor),
        dst_type=torch_npu.hifloat8,
        quant_mode="pertensor",
    )
    _OP_COUNTS[f"grouped_quantize_{kind}"] += 1
    return data, scale


def _quantized_grouped_mm(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    lhs_scale: torch.Tensor,
    rhs_scale: torch.Tensor,
    group_list: torch.Tensor,
    *,
    group_type: int,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Call native grouped matmul without materializing transposed operands."""
    if group_type not in (0, 2):
        raise ValueError(f"HiFloat8 grouped matmul only supports group_type 0 or 2, got {group_type}")

    num_experts = group_list.numel()
    lhs_scale_length = lhs.shape[0] if group_type == 0 else num_experts
    output = torch_npu.npu_grouped_matmul(
        [lhs],
        [rhs],
        bias=None,
        scale=[rhs_scale.reshape(1).expand(num_experts).contiguous()],
        per_token_scale=[lhs_scale.reshape(1).expand(lhs_scale_length).contiguous()],
        group_list=group_list,
        group_type=group_type,
        split_item=3,
        output_dtype=output_dtype,
        group_list_type=0,
        x_dtype=torch_npu.hifloat8,
        weight_dtype=torch_npu.hifloat8,
    )[0]
    _OP_COUNTS[f"grouped_matmul_group_type_{group_type}"] += 1
    return output


def assert_hifloat8_training_available(*, probe_kernel: bool = True, device=None) -> None:
    """Fail before model mutation if the Python ABI or NPU kernel is unavailable."""
    errors = _capability_errors()
    if errors:
        raise RuntimeError("HiFloat8 training is unavailable; missing: " + ", ".join(errors))
    if not probe_kernel:
        return

    if device is None:
        device = torch.device("npu", torch.npu.current_device())
    device = torch.device(device)
    if device.type != "npu":
        raise RuntimeError(f"HiFloat8 kernel probe requires an NPU device, got {device}")
    device_key = str(device)
    if device_key in _PROBED_DEVICES:
        return

    counts_before = _OP_COUNTS.copy()
    try:
        lhs = torch.ones((16, 16), dtype=torch.bfloat16, device=device)
        rhs = torch.ones((16, 16), dtype=torch.bfloat16, device=device)
        lhs_data, lhs_scale = _quantize(lhs, "input")
        rhs_data, rhs_scale = _quantize(rhs, "weight")
        _quantized_mm(lhs_data, rhs_data.t(), lhs_scale, rhs_scale)
        torch.npu.synchronize(device)
    except Exception as error:
        device_name = torch_npu.npu.get_device_name(device)
        raise RuntimeError(
            "HiFloat8 Python symbols are present, but the native quantize/matmul "
            f"probe failed on {device_name}: {error}"
        ) from error
    finally:
        _OP_COUNTS.clear()
        _OP_COUNTS.update(counts_before)
    _PROBED_DEVICES.add(device_key)


class _MatmulWithHiFloat8(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        if input.shape[-1] != weight.shape[-1]:
            raise ValueError(
                f"Linear shape mismatch: input K={input.shape[-1]}, weight K={weight.shape[-1]}"
            )

        input_shape = input.shape
        input_2d = input.reshape(-1, input_shape[-1])
        input_data, input_scale = _quantize(input_2d, "input")
        weight_data, weight_scale = _quantize(weight, "weight")

        need_input_grad = input.requires_grad
        need_weight_grad = weight.requires_grad
        saved = []
        if need_input_grad:
            saved.extend((weight_data, weight_scale))
        if need_weight_grad:
            saved.extend((input_data, input_scale))
        ctx.save_for_backward(*saved)
        ctx.need_input_grad = need_input_grad
        ctx.need_weight_grad = need_weight_grad
        ctx.input_shape = input_shape

        output = _quantized_mm(input_data, weight_data.t(), input_scale, weight_scale)
        _OP_COUNTS["forward"] += 1
        return output.reshape(*input_shape[:-1], weight.shape[0])

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        saved = iter(ctx.saved_tensors)
        if ctx.need_input_grad:
            weight_data, weight_scale = next(saved), next(saved)
        if ctx.need_weight_grad:
            input_data, input_scale = next(saved), next(saved)

        grad_input = grad_weight = None
        if ctx.need_input_grad or ctx.need_weight_grad:
            grad_2d = grad_output.reshape(-1, grad_output.shape[-1])
            grad_data, grad_scale = _quantize(grad_2d, "grad")
        if ctx.need_input_grad:
            grad_input = _quantized_mm(grad_data, weight_data, grad_scale, weight_scale)
            grad_input = grad_input.reshape(ctx.input_shape)
            _OP_COUNTS["backward_dx"] += 1
        if ctx.need_weight_grad:
            grad_weight = _quantized_mm(grad_data.t(), input_data, grad_scale, input_scale)
            _OP_COUNTS["backward_dw"] += 1
        return grad_input, grad_weight


def _validate_grouped_inputs(
    input: torch.Tensor,
    weight: torch.Tensor,
    group_list: torch.Tensor,
) -> None:
    if input.ndim != 2:
        raise ValueError(f"Grouped HiFloat8 input must be 2D, got {input.ndim}D")
    if weight.ndim != 3:
        raise ValueError(f"Grouped HiFloat8 weight must be 3D, got {weight.ndim}D")
    if group_list.ndim != 1:
        raise ValueError(f"Grouped HiFloat8 group_list must be 1D, got {group_list.ndim}D")
    if weight.shape[0] != group_list.numel():
        raise ValueError(
            "Grouped HiFloat8 expert count mismatch: "
            f"weight has {weight.shape[0]} experts but group_list has {group_list.numel()} offsets"
        )
    if input.shape[-1] != weight.shape[-2]:
        raise ValueError(
            f"Grouped HiFloat8 shape mismatch: input K={input.shape[-1]}, weight K={weight.shape[-2]}"
        )
    supported_dtypes = (torch.float16, torch.bfloat16, torch.float32)
    if input.dtype not in supported_dtypes or weight.dtype not in supported_dtypes:
        raise TypeError(
            "Grouped HiFloat8 input and weight must use float16, bfloat16, or float32; "
            f"got {input.dtype} and {weight.dtype}"
        )
    if input.device != weight.device:
        raise ValueError(
            f"Grouped HiFloat8 input and weight must share a device, got {input.device} and {weight.device}"
        )


class _GroupedMatmulWithHiFloat8(torch.autograd.Function):
    """Per-tensor HiFloat8 grouped matmul with native group0/group2 backward."""

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        group_list: torch.Tensor,
    ) -> torch.Tensor:
        _validate_grouped_inputs(input, weight, group_list)

        ctx.need_input_grad = input.requires_grad
        ctx.need_weight_grad = weight.requires_grad
        ctx.empty_input = input.shape[0] == 0
        if ctx.empty_input:
            ctx.save_for_backward(input, weight)
            _OP_COUNTS["grouped_forward"] += 1
            _OP_COUNTS["grouped_empty_forward"] += 1
            return _ensure_bf16_or_fp16(input).new_empty((0, weight.shape[-1]))

        input_work = _ensure_bf16_or_fp16(input)
        weight_work = _ensure_bf16_or_fp16(weight)
        group_list = group_list.to(dtype=torch.int64)
        input_data, input_scale = _grouped_quantize(input_work, "input")
        weight_data, weight_scale = _grouped_quantize(weight_work, "weight")

        output = _quantized_grouped_mm(
            input_data,
            weight_data,
            input_scale,
            weight_scale,
            group_list,
            group_type=0,
            output_dtype=input_work.dtype,
        )

        ctx.save_for_backward(input_data, input_scale, weight_work, group_list)
        ctx.output_dtype = input_work.dtype
        _OP_COUNTS["grouped_forward"] += 1
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if ctx.empty_input:
            input, weight = ctx.saved_tensors
            grad_input = torch.zeros_like(input) if ctx.need_input_grad else None
            grad_weight = torch.zeros_like(weight) if ctx.need_weight_grad else None
            if ctx.need_input_grad:
                _OP_COUNTS["grouped_backward_dx"] += 1
            if ctx.need_weight_grad:
                _OP_COUNTS["grouped_backward_dw"] += 1
            _OP_COUNTS["grouped_empty_backward"] += 1
            return grad_input, grad_weight, None

        input_data, input_scale, weight, group_list = ctx.saved_tensors
        grad_input = grad_weight = None

        if ctx.need_input_grad or ctx.need_weight_grad:
            grad_work = _ensure_bf16_or_fp16(grad_output)
            grad_data, grad_scale = _grouped_quantize(grad_work, "grad")

        if ctx.need_input_grad:
            weight_t_data, weight_t_scale = _grouped_quantize(
                weight.transpose(-1, -2),
                "weight",
            )
            grad_input = _quantized_grouped_mm(
                grad_data,
                weight_t_data,
                grad_scale,
                weight_t_scale,
                group_list,
                group_type=0,
                output_dtype=ctx.output_dtype,
            )
            _OP_COUNTS["grouped_backward_dx"] += 1

        if ctx.need_weight_grad:
            # The native split-K kernel consumes this transpose view directly.
            # Making it contiguous changes the group_type=2 layout contract.
            grad_weight = _quantized_grouped_mm(
                input_data.transpose(0, 1),
                grad_data,
                input_scale,
                grad_scale,
                group_list,
                group_type=2,
                output_dtype=ctx.output_dtype,
            )
            _OP_COUNTS["grouped_backward_dw"] += 1

        return grad_input, grad_weight, None


def hifloat8_grouped_mm(
    input: torch.Tensor,
    weight: torch.Tensor,
    group_list: torch.Tensor,
) -> torch.Tensor:
    """Multiply token rows by expert weights through native HiFloat8 grouped GEMM.

    ``input`` has shape ``[M, K]`` and ``weight`` has shape ``[E, K, N]``.
    ``group_list`` contains ``E`` cumulative row offsets; repeated offsets
    represent empty experts. The returned tensor has shape ``[M, N]``.
    """
    return _GroupedMatmulWithHiFloat8.apply(input, weight, group_list)


def assert_hifloat8_grouped_training_available(
    *,
    probe_kernel: bool = True,
    device=None,
) -> None:
    """Fail before model loading if native HiFloat8 grouped training is unavailable."""
    errors = _grouped_capability_errors()
    if errors:
        raise RuntimeError("HiFloat8 grouped training is unavailable; missing: " + ", ".join(errors))
    if not probe_kernel:
        return

    if device is None:
        device = torch.device("npu", torch.npu.current_device())
    device = torch.device(device)
    if device.type != "npu":
        raise RuntimeError(f"HiFloat8 grouped kernel probe requires an NPU device, got {device}")
    device_key = str(device)
    if device_key in _PROBED_GROUPED_DEVICES:
        return

    counts_before = _OP_COUNTS.copy()
    try:
        input = torch.ones((16, 16), dtype=torch.bfloat16, device=device, requires_grad=True)
        weight = torch.ones((2, 16, 16), dtype=torch.bfloat16, device=device, requires_grad=True)
        group_list = torch.tensor((0, 16), dtype=torch.int64, device=device)
        output = hifloat8_grouped_mm(input, weight, group_list)
        output.sum().backward()

        tensors = (output, input.grad, weight.grad)
        if any(tensor is None or not torch.isfinite(tensor).all().item() for tensor in tensors):
            raise RuntimeError("native grouped forward or backward produced a non-finite tensor")
        if torch.count_nonzero(weight.grad[0]).item() != 0:
            raise RuntimeError("native group_type=2 kernel produced a non-zero gradient for an empty expert")
        torch.npu.synchronize(device)
    except Exception as error:
        device_name = torch_npu.npu.get_device_name(device)
        raise RuntimeError(
            "HiFloat8 grouped Python symbols are present, but the native group_type=0/2 "
            f"kernel probe failed on {device_name}: {error}"
        ) from error
    finally:
        _OP_COUNTS.clear()
        _OP_COUNTS.update(counts_before)
    _PROBED_GROUPED_DEVICES.add(device_key)


class HiFloat8Linear(nn.Linear):
    """Linear module whose three training GEMMs use native HiFloat8 operands."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = _MatmulWithHiFloat8.apply(input, self.weight)
        if self.bias is not None:
            output = output + self.bias.to(output.dtype)
        return output

    @classmethod
    def from_float(cls, module: nn.Linear) -> "HiFloat8Linear":
        if type(module) is not nn.Linear:
            raise TypeError(f"HiFloat8 conversion only supports exact nn.Linear modules, got {type(module).__name__}")
        module.__class__ = cls
        return module


def convert_to_hifloat8_training(
    module: nn.Module,
    *,
    module_filter_fn: Optional[Callable[[nn.Module, str], bool]] = None,
) -> nn.Module:
    """Replace selected ``nn.Linear`` modules while preserving Parameters."""
    torch._C._log_api_usage_once("torch_npu.hifloat8_training")

    if type(module) is nn.Linear:
        if module_filter_fn is None or module_filter_fn(module, ""):
            return HiFloat8Linear.from_float(module)
        return module

    def visit(parent: nn.Module, prefix: str = "") -> None:
        for child_name, child in list(parent.named_children()):
            fqn = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, HiFloat8Linear):
                continue
            if type(child) is nn.Linear and (
                module_filter_fn is None or module_filter_fn(child, fqn)
            ):
                setattr(parent, child_name, HiFloat8Linear.from_float(child))
            else:
                visit(child, fqn)

    visit(module)
    return module


def get_hifloat8_op_counts() -> Dict[str, int]:
    return dict(_OP_COUNTS)


def reset_hifloat8_op_counts() -> None:
    _OP_COUNTS.clear()


__all__ = [
    "HiFloat8Linear",
    "assert_hifloat8_grouped_training_available",
    "assert_hifloat8_training_available",
    "convert_to_hifloat8_training",
    "get_hifloat8_op_counts",
    "hifloat8_grouped_mm",
    "reset_hifloat8_op_counts",
]
