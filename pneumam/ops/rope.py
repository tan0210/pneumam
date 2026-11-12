import torch

from ..backends import reference, triton
from ..backends._fp8 import norm_scale
from . import _dispatch

__all__ = ["norm_scale", "qkv_norm_rope", "qkv_norm_rope_fp8"]


def qkv_norm_rope(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    heads: int,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split a fused QKV projection, RMS-normalize Q and K per head and rotate them.

    The rotation pairs dimension ``i`` with dimension ``i + hd // 2``. Differentiable.

    Args:
        qkv: Fused projection, contiguous, shape ``[B, S, 3 * heads * hd]``.
        q_weight: Q norm scale, shape ``[hd]``.
        k_weight: K norm scale, shape ``[hd]``.
        cos: Rotary cosines, shape ``[S, hd // 2]`` with ``hd // 2`` a power of two.
        sin: Rotary sines, shape ``[S, hd // 2]``.
        heads: Number of heads.
        eps: Epsilon added to the mean square.

    Returns:
        ``(q, k, v)``, each ``[B, S, heads, hd]`` with the dtype of ``qkv``. ``v`` may be a strided
        view of ``qkv``.

    Raises:
        RuntimeError: If ``qkv`` is not on a CUDA device.
        ValueError: If a shape requirement above is not met.
    """
    if triton.available and _dispatch.arch(qkv.device) != "reference":
        return triton.qkv_norm_rope(qkv, q_weight, k_weight, cos, sin, heads, eps)
    return reference.qkv_norm_rope(qkv, q_weight, k_weight, cos, sin, heads, eps)


def qkv_norm_rope_fp8(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    heads: int,
    q_scale: float,
    k_scale: float,
    v_scale: float = 1.0,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``qkv_norm_rope`` whose outputs are FP8, each multiplied by a fixed scale. Forward only.

    Values beyond 448 in magnitude after scaling are clipped. ``norm_scale`` of the Q or K norm
    weight is a scale that never clips, and it can be computed once per weight. V is not
    normalized, so no such scale exists for it. See ``attention_quantized``.

    Args:
        qkv: Fused projection, contiguous, shape ``[B, S, 3 * heads * hd]``.
        q_weight: Q norm scale, shape ``[hd]``.
        k_weight: K norm scale, shape ``[hd]``.
        cos: Rotary cosines, shape ``[S, hd // 2]`` with ``hd // 2`` a power of two.
        sin: Rotary sines, shape ``[S, hd // 2]``.
        heads: Number of heads.
        q_scale: Factor applied to Q before quantizing.
        k_scale: Factor applied to K before quantizing.
        v_scale: Factor applied to V before quantizing.
        eps: Epsilon added to the mean square.

    Returns:
        ``(q, k, v)``, each FP8 of shape ``[B, S, heads, hd]``. Divide by the scales to recover
        the values.

    Raises:
        RuntimeError: If ``qkv`` is not on a CUDA device.
        ValueError: If a shape requirement above is not met.
    """
    args = (qkv, q_weight, k_weight, cos, sin, heads, q_scale, k_scale, v_scale, eps)
    if triton.available and _dispatch.arch(qkv.device) != "reference":
        return triton.qkv_norm_rope_fp8(*args)
    return reference.qkv_norm_rope_fp8(*args)
