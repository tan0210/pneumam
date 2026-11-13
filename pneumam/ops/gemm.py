import torch
import torch.nn.functional as F

from ..backends import quack, reference, triton, warpy
from ..backends._fp8 import FP8Weight, quantize_weight
from . import _dispatch

__all__ = [
    "FP8Weight",
    "fp8_linear",
    "fp8_supported",
    "linear",
    "linear_gelu",
    "mlp",
    "quantize_weight",
    "rowquant",
]

_DATACENTER = ("sm90", "sm100")


def _needs_grad(*tensors: torch.Tensor) -> bool:
    return torch.is_grad_enabled() and any(t.requires_grad for t in tensors)


def linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Compute ``x @ weight.T``. Differentiable.

    Args:
        x: Activations, shape ``[..., K]``.
        weight: Weight with the dtype of ``x``, shape ``[N, K]``.

    Returns:
        Tensor with the dtype of ``x``, shape ``[..., N]``.

    Raises:
        RuntimeError: If ``x`` is not on a CUDA device.
    """
    if _dispatch.arch(x.device) == "sm120" and warpy.fits(x, weight):
        return warpy.gemm(x, weight)
    return F.linear(x, weight)


def linear_gelu(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Compute ``gelu(x @ weight.T)`` with the exact erf GELU and no bias. Differentiable.

    Args:
        x: Activations, shape ``[..., K]``.
        weight: Weight with the dtype of ``x``, shape ``[N, K]``.

    Returns:
        Tensor with the dtype of ``x``, shape ``[..., N]``.

    Raises:
        RuntimeError: If ``x`` is not on a CUDA device.
    """
    arch = _dispatch.arch(x.device)
    if arch == "sm120" and warpy.fits(x, weight) and not _needs_grad(x, weight):
        return warpy.linear_gelu(x, weight)
    if arch in _DATACENTER and quack.fits(x, weight):
        return quack.linear_gelu(x, weight)
    return reference.linear_gelu(x, weight)


def mlp(x: torch.Tensor, w_up: torch.Tensor, w_down: torch.Tensor) -> torch.Tensor:
    """Compute ``gelu(x @ w_up.T) @ w_down.T`` with the exact erf GELU and no biases.

    Differentiable.

    Args:
        x: Activations, shape ``[..., K]``.
        w_up: Weight with the dtype of ``x``, shape ``[H, K]``.
        w_down: Weight with the dtype of ``x``, shape ``[N, H]``.

    Returns:
        Tensor with the dtype of ``x``, shape ``[..., N]``.

    Raises:
        RuntimeError: If ``x`` is not on a CUDA device.
    """
    if _dispatch.arch(x.device) in _DATACENTER and quack.fits(x, w_up, w_down):
        return quack.mlp(x, w_up, w_down)
    return linear(linear_gelu(x, w_up), w_down)


def fp8_supported(device: torch.device) -> bool:
    """Tell whether FP8 products run as real FP8 kernels on a device.

    Args:
        device: Device to check.

    Returns:
        True if ``fp8_linear`` has an FP8 kernel there, or if ops are pinned to the reference
        path. If False it still works, but slowly.

    Raises:
        RuntimeError: If ``device`` is not a CUDA device.
    """
    arch = _dispatch.arch(device)
    return (
        arch == "reference"
        or (arch == "sm120" and warpy.available)
        or (arch in _DATACENTER and triton.available)
    )


def fp8_linear(
    q: torch.Tensor, s: torch.Tensor, w: FP8Weight, fast_accum: bool = False
) -> torch.Tensor:
    """Multiply FP8 tokens by an FP8 weight. Forward only.

    Args:
        q: FP8 tokens, shape ``[B, S, K]``.
        s: Token scales, shape ``[B * S, 1]``. See ``rowquant``.
        w: Weight from ``quantize_weight``, shape ``[N, K]``.
        fast_accum: Accumulate with reduced precision for extra speed, where the GPU offers it.

    Returns:
        bfloat16 result, shape ``[B, S, N]``.

    Raises:
        RuntimeError: If ``q`` is not on a CUDA device.
    """
    arch = _dispatch.arch(q.device)
    if arch == "sm120" and warpy.fits(q, w.q):
        return warpy.fp8_linear(q, s, w, fast_accum)
    if arch in _DATACENTER and triton.available:
        return triton.fp8_linear(q, s, w, fast_accum)
    return reference.fp8_linear(q, s, w, fast_accum)


def rowquant(x: torch.Tensor, gelu: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize tokens to FP8 with a scale per token, optionally applying GELU first.

    Forward only.

    Args:
        x: Tokens, shape ``[B, S, K]``.
        gelu: Apply GELU before quantizing. It is within 1e-6 of the exact erf GELU.

    Returns:
        ``(q, s)``: the FP8 tokens ``[B, S, K]`` and their float32 scales ``[B * S, 1]``.
        See ``fp8_linear``.

    Raises:
        RuntimeError: If ``x`` is not on a CUDA device.
    """
    if triton.available and _dispatch.arch(x.device) != "reference":
        return triton.rowquant(x, gelu)
    return reference.rowquant(x, gelu)
