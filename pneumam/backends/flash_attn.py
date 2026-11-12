import torch

try:
    from flash_attn.cute.interface import _flash_attn_bwd, _flash_attn_fwd  # type: ignore

    available = True
except Exception:  # noqa: BLE001  # cutlass can fail to import in ways other than ImportError
    _flash_attn_bwd = None
    _flash_attn_fwd = None
    available = False

_DTYPES = (torch.bfloat16, torch.float16)


def fits(q: torch.Tensor) -> bool:
    """Tell whether this backend can run attention on inputs like ``q``.

    Args:
        q: Representative attention input.

    Returns:
        True if the library is installed and ``q`` is bf16 or fp16.
    """
    return available and q.dtype in _DTYPES


@torch.library.custom_op("pneumam::fa_fwd", mutates_args=())
def _fa_fwd(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    out, lse, *_ = _flash_attn_fwd(q, k, v, causal=False, return_lse=True)  # type: ignore
    expected = (q.shape[0], q.shape[2], q.shape[1])
    if tuple(lse.shape) != expected:
        raise RuntimeError(
            f"Unexpected FA4 lse shape {tuple(lse.shape)}, expected {expected}"
        )
    return out, lse


@_fa_fwd.register_fake
def _(q, k, v):
    return torch.empty_like(q), q.new_empty(
        (q.shape[0], q.shape[2], q.shape[1]), dtype=torch.float32
    )


@torch.library.custom_op("pneumam::fa_fwd_nolse", mutates_args=())
def _fa_fwd_nolse(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    out, *_ = _flash_attn_fwd(q, k, v, causal=False, return_lse=False)  # type: ignore
    return out


@_fa_fwd_nolse.register_fake
def _(q, k, v):
    return q.new_empty(q.shape)


@torch.library.custom_op("pneumam::fa_fwd_k", mutates_args=())
def _fa_fwd_k(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, seqused_k: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    out, lse, *_ = _flash_attn_fwd(  # type: ignore
        q, k, v, seqused_k=seqused_k, causal=False, return_lse=True
    )
    return out, lse


@_fa_fwd_k.register_fake
def _(q, k, v, seqused_k):
    return torch.empty_like(q), q.new_empty(
        (q.shape[0], q.shape[2], q.shape[1]), dtype=torch.float32
    )


@torch.library.custom_op("pneumam::fa_bwd", mutates_args=())
def _fa_bwd(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dq, dk, dv, *_ = _flash_attn_bwd(  # type: ignore
        q, k, v, out, dout.contiguous(), lse, causal=False, deterministic=False
    )
    return dq, dk, dv


@_fa_bwd.register_fake
def _(dout, q, k, v, out, lse):
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


def _setup_context(ctx, inputs, output):
    q, k, v = inputs
    out, lse = output
    ctx.mark_non_differentiable(lse)
    ctx.save_for_backward(q, k, v, out, lse)


def _backward(ctx, dout, dlse):
    q, k, v, out, lse = ctx.saved_tensors
    return _fa_bwd(dout, q, k, v, out, lse)


_fa_fwd.register_autograd(_backward, setup_context=_setup_context)


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Full, non-causal multi-head attention. Differentiable.

    Inputs may be strided views as long as the last dimension has stride 1.

    Args:
        q: Queries accepted by ``fits``, shape ``[B, Sq, H, hd]``.
        k: Keys, shape ``[B, Sk, H, hd]``.
        v: Values, shape ``[B, Sk, H, hd]``.

    Returns:
        Attention output, shape ``[B, Sq, H, hd]``, dtype of ``q``.
    """
    if torch.is_grad_enabled() and (
        q.requires_grad or k.requires_grad or v.requires_grad
    ):
        out, _ = _fa_fwd(q, k, v)
        return out
    return _fa_fwd_nolse(q, k, v)


def attention_trimmed(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, seqused_k: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Attention over the first ``seqused_k[b]`` keys of each batch item. Forward only.

    Args:
        q: Queries accepted by ``fits``, shape ``[B, Sq, H, hd]``.
        k: Keys, shape ``[B, Sk, H, hd]``.
        v: Values, shape ``[B, Sk, H, hd]``.
        seqused_k: Valid key count per batch item, int32, shape ``[B]``.

    Returns:
        ``(out, lse)``: the output ``[B, Sq, H, hd]`` and the float32 natural-log softmax normalizer
        ``[B, H, Sq]``. See ``ctx_scale``.
    """
    return _fa_fwd_k(q.contiguous(), k, v, seqused_k)
