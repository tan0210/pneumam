import torch
import torch.nn.functional as F

from ..backends import ffpa, flash_attn, reference, triton, warpy
from ..backends._fp8 import to_fp8
from . import _dispatch

__all__ = [
    "attention",
    "attention_fp8",
    "attention_quantized",
    "cross_attention_trimmed",
    "cross_attention_trimmed_quantized",
    "to_fp8",
    "trims_fast",
    "wide_attention",
]

_HALF = (torch.bfloat16, torch.float16)


def _flash(q: torch.Tensor) -> bool:
    return _dispatch.arch(q.device).startswith("sm") and flash_attn.fits(q)


def _warpy(q: torch.Tensor) -> bool:
    return _dispatch.arch(q.device) == "sm120" and warpy.fits_attention(q)


def _ctx_scale(out: torch.Tensor, lse: torch.Tensor, ln_pad: torch.Tensor):
    if triton.available and _dispatch.arch(out.device) != "reference":
        return triton.ctx_scale(out, lse, ln_pad)
    return reference.ctx_scale(out, lse, ln_pad)


def _dequantize(q, k, v, q_scale, k_scale, v_scale):
    pairs = ((q, q_scale), (k, k_scale), (v, v_scale))
    return [(t.float() / s).to(torch.bfloat16) for t, s in pairs]


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Full, non-causal multi-head attention. Differentiable.

    Args:
        q: Queries, shape ``[B, Sq, H, hd]``.
        k: Keys, shape ``[B, Sk, H, hd]``.
        v: Values, shape ``[B, Sk, H, hd]``.

    Returns:
        Attention output with heads merged, shape ``[B, Sq, H * hd]``, dtype of ``q``.

    Raises:
        RuntimeError: If ``q`` is not on a CUDA device.
    """
    if _flash(q):
        return flash_attn.attention(q, k, v).flatten(2)
    return reference.attention(q, k, v).flatten(2)


def attention_fp8(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, rescale: bool = True
) -> torch.Tensor:
    """``attention`` computed in FP8 where the GPU has a kernel for it. Forward only.

    Elsewhere this is ``attention``.

    Args:
        q: Queries, shape ``[B, Sq, H, hd]``.
        k: Keys, shape ``[B, Sk, H, hd]``.
        v: Values, shape ``[B, Sk, H, hd]``.
        rescale: Whether to scale each head of each operand to the full FP8 range before
            quantizing. Turning it off is experimental. It is faster, but values above 448 in
            magnitude are clipped and values far below 1 lose precision.

    Returns:
        Attention output with heads merged, shape ``[B, Sq, H * hd]``, bfloat16 where FP8 ran and
        the dtype of ``q`` otherwise.

    Raises:
        RuntimeError: If ``q`` is not on a CUDA device.
    """
    if _warpy(q):
        return warpy.attention(q, k, v, rescale).flatten(2)
    return attention(q, k, v)


def attention_quantized(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_scale: float = 1.0,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
) -> torch.Tensor:
    """``attention`` on operands that are already FP8. Forward only.

    See ``qkv_norm_rope_fp8``.

    Args:
        q: FP8 queries holding the true queries times ``q_scale``, shape ``[B, Sq, H, hd]``.
        k: FP8 keys times ``k_scale``, shape ``[B, Sk, H, hd]``.
        v: FP8 values times ``v_scale``, shape ``[B, Sk, H, hd]``.
        q_scale: Factor the queries were multiplied by before quantizing.
        k_scale: Factor the keys were multiplied by before quantizing.
        v_scale: Factor the values were multiplied by before quantizing.

    Returns:
        Attention output in true units with heads merged, bfloat16, shape ``[B, Sq, H * hd]``.

    Raises:
        RuntimeError: If ``q`` is not on a CUDA device.
    """
    if _warpy(q):
        return warpy.attention_quantized(q, k, v, q_scale, k_scale, v_scale).flatten(2)
    return attention(*_dequantize(q, k, v, q_scale, k_scale, v_scale))


def trims_fast(q: torch.Tensor) -> bool:
    """Tell whether trimming the context speeds up cross-attention for inputs like ``q``.

    Args:
        q: Representative attention input.

    Returns:
        True if ``cross_attention_trimmed`` skips the trimmed keys. If False it still returns the
        right result, but is slower than ``attention`` over the padded context.

    Raises:
        RuntimeError: If ``q`` is not on a CUDA device.
    """
    return _flash(q) or _warpy(q)


def cross_attention_trimmed(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seqused_k: torch.Tensor,
    ln_pad: torch.Tensor,
    fp8: bool = False,
    rescale: bool = True,
) -> torch.Tensor:
    """Cross-attention over a trimmed context, corrected for the padding that was removed.

    Only the first ``seqused_k[b]`` keys of batch item ``b`` are attended. The output matches
    attention over the full context when the removed keys and values were all zero. Forward only.

    Args:
        q: Queries, shape ``[B, Sq, H, hd]``.
        k: Trimmed keys, shape ``[B, Sk_trim, H, hd]``.
        v: Trimmed values, shape ``[B, Sk_trim, H, hd]``.
        seqused_k: Valid key count per batch item, int32, shape ``[B]``.
        ln_pad: ``log(S_full - seqused_k)`` per batch item, float32, shape ``[B]``.
        fp8: Whether to compute in FP8 where the GPU has a kernel for it.
        rescale: As in ``attention_fp8``. Only used with ``fp8``.

    Returns:
        Attention output with heads merged, shape ``[B, Sq, H * hd]``.

    Raises:
        RuntimeError: If ``q`` is not on a CUDA device.
    """
    if fp8 and _warpy(q):
        out, lse = warpy.attention_trimmed(q, k, v, seqused_k, rescale)
    elif _flash(q):
        out, lse = flash_attn.attention_trimmed(q, k, v, seqused_k)
    else:
        out, lse = reference.attention_trimmed(q, k, v, seqused_k)
    return _ctx_scale(out, lse, ln_pad).flatten(2)


def cross_attention_trimmed_quantized(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seqused_k: torch.Tensor,
    ln_pad: torch.Tensor,
    q_scale: float = 1.0,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
) -> torch.Tensor:
    """``cross_attention_trimmed`` on operands that are already FP8. Forward only.

    See ``head_norm_fp8`` for the queries and ``to_fp8`` for a context computed once per prompt.

    Args:
        q: FP8 queries holding the true queries times ``q_scale``, shape ``[B, Sq, H, hd]``.
        k: Trimmed FP8 keys times ``k_scale``, shape ``[B, Sk_trim, H, hd]``.
        v: Trimmed FP8 values times ``v_scale``, shape ``[B, Sk_trim, H, hd]``.
        seqused_k: Valid key count per batch item, int32, shape ``[B]``.
        ln_pad: ``log(S_full - seqused_k)`` per batch item, float32, shape ``[B]``.
        q_scale: Factor the queries were multiplied by before quantizing.
        k_scale: Factor the keys were multiplied by before quantizing.
        v_scale: Factor the values were multiplied by before quantizing.

    Returns:
        Attention output in true units with heads merged, bfloat16, shape ``[B, Sq, H * hd]``.

    Raises:
        RuntimeError: If ``q`` is not on a CUDA device.
    """
    if _warpy(q):
        out, lse = warpy.attention_trimmed_quantized(
            q, k, v, seqused_k, q_scale, k_scale, v_scale
        )
        return _ctx_scale(out, lse, ln_pad).flatten(2)
    q, k, v = _dequantize(q, k, v, q_scale, k_scale, v_scale)
    return cross_attention_trimmed(q, k, v, seqused_k, ln_pad)


def wide_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Full, non-causal attention for head widths above 256. Differentiable.

    Heads come before the sequence here, unlike in ``attention``.

    Args:
        q: Queries, shape ``[B, H, Sq, hd]``.
        k: Keys, shape ``[B, H, Sk, hd]``.
        v: Values, shape ``[B, H, Sk, hd]``.

    Returns:
        Attention output, shape ``[B, H, Sq, hd]``, dtype of ``q``.

    Raises:
        RuntimeError: If ``q`` is not on a CUDA device.
    """
    if (
        ffpa.available
        and q.dtype in _HALF
        and _dispatch.arch(q.device).startswith("sm")
    ):
        return ffpa.attention(q.contiguous(), k.contiguous(), v.contiguous())
    return F.scaled_dot_product_attention(q, k, v)
