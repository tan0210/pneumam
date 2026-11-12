import torch

try:
    from ffpa_attn import ffpa_attn_func as _ffpa_attn_func

    available = True
except Exception:  # noqa: BLE001
    _ffpa_attn_func = None
    available = False


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Full, non-causal attention. Differentiable.

    Heads come before the sequence here, unlike in the other attention backends.

    Args:
        q: Queries, bf16 or fp16, contiguous, shape ``[B, H, Sq, hd]``.
        k: Keys, shape ``[B, H, Sk, hd]``.
        v: Values, shape ``[B, H, Sk, hd]``.

    Returns:
        Attention output, shape ``[B, H, Sq, hd]``, dtype of ``q``.
    """
    return _ffpa_attn_func(q, k, v)  # type: ignore
