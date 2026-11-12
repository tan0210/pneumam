import torch

from ..backends import reference, triton
from . import _dispatch


def rms_silu(x: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    """Channel RMS norm followed by SiLU. Differentiable.

    Computes ``silu(normalize(x, dim=-1) * sqrt(C) * gamma)``.

    Args:
        x: Activations with channels last, shape ``[..., C]``.
        gamma: Per-channel scale, shape ``[C]``.

    Returns:
        Tensor with the shape and dtype of ``x``.

    Raises:
        RuntimeError: If ``x`` is not on a CUDA device.
    """
    if triton.available and _dispatch.arch(x.device) != "reference":
        return triton.rms_silu(x, gamma)
    return reference.rms_silu(x, gamma)


def add_rms_silu(
    x: torch.Tensor, residual: torch.Tensor, gamma: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Residual add followed by ``rms_silu`` of the sum. Differentiable.

    Args:
        x: Branch output with channels last, shape ``[..., C]``.
        residual: Skip connection, same shape as ``x``.
        gamma: Per-channel scale of the following norm, shape ``[C]``.

    Returns:
        ``(x + residual, rms_silu(x + residual, gamma))``.

    Raises:
        RuntimeError: If ``x`` is not on a CUDA device.
    """
    if triton.available and _dispatch.arch(x.device) != "reference":
        return triton.add_rms_silu(x, residual, gamma)
    return reference.add_rms_silu(x, residual, gamma)


def shuffle_rms_silu(
    u: torch.Tensor, gamma: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Depth-to-space by 2 followed by ``rms_silu``. Differentiable.

    Args:
        u: Sub-pixel activations, shape ``[B, H, W, 4 * C]``. Channel ``(2 * a + b) * C + c`` is
            the value of channel ``c`` at output row ``2 * i + a``, column ``2 * j + b``.
        gamma: Per-channel scale, shape ``[C]``.

    Returns:
        ``(x, rms_silu(x, gamma))`` with ``x`` of shape ``[B, 2 * H, 2 * W, C]``.

    Raises:
        RuntimeError: If ``u`` is not on a CUDA device.
    """
    if triton.available and _dispatch.arch(u.device) != "reference":
        return triton.shuffle_rms_silu(u, gamma)
    return reference.shuffle_rms_silu(u, gamma)


def head_norm(
    x: torch.Tensor, weight: torch.Tensor, heads: int, eps: float = 1e-5
) -> torch.Tensor:
    """Split tokens into heads and RMS-normalize each head. Differentiable.

    Args:
        x: Tokens, contiguous, shape ``[B, S, heads * hd]``.
        weight: Per-dimension scale, shape ``[hd]``.
        heads: Number of heads.
        eps: Epsilon added to the mean square.

    Returns:
        Tensor of shape ``[B, S, heads, hd]`` with the dtype of ``x``.

    Raises:
        RuntimeError: If ``x`` is not on a CUDA device.
        ValueError: If ``x`` is not contiguous ``[B, S, heads * hd]``.
    """
    if triton.available and _dispatch.arch(x.device) != "reference":
        return triton.head_norm(x, weight, heads, eps)
    return reference.head_norm(x, weight, heads, eps)


def head_norm_fp8(
    x: torch.Tensor,
    weight: torch.Tensor,
    heads: int,
    scale: float,
    eps: float = 1e-5,
) -> torch.Tensor:
    """``head_norm`` whose output is FP8, multiplied by a fixed scale. Forward only.

    Values beyond 448 in magnitude after scaling are clipped. ``norm_scale`` of ``weight`` is a
    scale that never clips. See ``norm_scale``.

    Args:
        x: Tokens, contiguous, shape ``[B, S, heads * hd]``.
        weight: Per-dimension scale, shape ``[hd]``.
        heads: Number of heads.
        scale: Factor applied before quantizing.
        eps: Epsilon added to the mean square.

    Returns:
        FP8 tensor of shape ``[B, S, heads, hd]``. Divide by ``scale`` to recover the values.

    Raises:
        RuntimeError: If ``x`` is not on a CUDA device.
        ValueError: If ``x`` is not contiguous ``[B, S, heads * hd]``.
    """
    if triton.available and _dispatch.arch(x.device) != "reference":
        return triton.head_norm_fp8(x, weight, heads, scale, eps)
    return reference.head_norm_fp8(x, weight, heads, scale, eps)
