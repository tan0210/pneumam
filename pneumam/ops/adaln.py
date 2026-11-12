import torch

from ..backends import reference, triton
from . import _dispatch


def adaln(
    z: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """LayerNorm without learned parameters, then a per-sample scale and shift. Differentiable.

    Args:
        z: Tokens, shape ``[B, S, D]``.
        shift: Shift, shape ``[B, D]``.
        scale: Scale (already including the +1), shape ``[B, D]``.
        eps: LayerNorm epsilon.

    Returns:
        Tensor with the shape and dtype of ``z``.

    Raises:
        RuntimeError: If ``z`` is not on a CUDA device.
    """
    if triton.available and _dispatch.arch(z.device) != "reference":
        return triton.adaln(z, shift, scale, eps)
    return reference.adaln(z, shift, scale, eps)


def residual_adaln(
    x: torch.Tensor,
    residual: torch.Tensor,
    gate: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gated residual add, then ``adaln`` of the updated residual stream. Differentiable.

    Args:
        x: Sub-layer output, shape ``[B, S, D]``.
        residual: Residual stream, same shape and dtype as ``x``.
        gate: Gate, shape ``[B, D]``.
        shift: Shift, shape ``[B, D]``.
        scale: Scale (already including the +1), shape ``[B, D]``.
        eps: LayerNorm epsilon.

    Returns:
        ``(h, z)``: ``z = residual + gate * x`` and ``h = adaln(z, shift, scale)``.

    Raises:
        RuntimeError: If ``x`` is not on a CUDA device.
        ValueError: If ``x`` and ``residual`` differ in shape or dtype.
    """
    if x.shape != residual.shape or x.dtype != residual.dtype:
        raise ValueError("x and residual must match")
    if triton.available and _dispatch.arch(x.device) != "reference":
        return triton.residual_adaln(x, residual, gate, shift, scale, eps)
    return reference.residual_adaln(x, residual, gate, shift, scale, eps)


def adaln_fp8(
    z: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor, eps: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor]:
    """``adaln`` whose output is returned as FP8 with a scale per token. Forward only.

    Args:
        z: Tokens, shape ``[B, S, D]``.
        shift: Shift, shape ``[B, D]``.
        scale: Scale (already including the +1), shape ``[B, D]``.
        eps: LayerNorm epsilon.

    Returns:
        ``(q, s)``: the FP8 output ``[B, S, D]`` and its float32 token scales ``[B * S, 1]``.

    Raises:
        RuntimeError: If ``z`` is not on a CUDA device.
    """
    if triton.available and _dispatch.arch(z.device) != "reference":
        return triton.adaln_fp8(z, shift, scale, eps)
    return reference.adaln_fp8(z, shift, scale, eps)


def residual_adaln_fp8(
    x: torch.Tensor,
    residual: torch.Tensor,
    gate: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``residual_adaln`` whose modulated output is returned as FP8 with a scale per token.

    Forward only.

    Args:
        x: Sub-layer output, shape ``[B, S, D]``.
        residual: Residual stream, same shape and dtype as ``x``.
        gate: Gate, shape ``[B, D]``.
        shift: Shift, shape ``[B, D]``.
        scale: Scale (already including the +1), shape ``[B, D]``.
        eps: LayerNorm epsilon.

    Returns:
        ``(q, s, z)``: the FP8 output ``[B, S, D]``, its float32 token scales ``[B * S, 1]``, and
        the updated residual stream ``z`` ``[B, S, D]`` in the input dtype.

    Raises:
        RuntimeError: If ``x`` is not on a CUDA device.
        ValueError: If ``x`` and ``residual`` differ in shape or dtype.
    """
    if x.shape != residual.shape or x.dtype != residual.dtype:
        raise ValueError("x and residual must match")
    if triton.available and _dispatch.arch(x.device) != "reference":
        return triton.residual_adaln_fp8(x, residual, gate, shift, scale, eps)
    return reference.residual_adaln_fp8(x, residual, gate, shift, scale, eps)
