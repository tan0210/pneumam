import torch

from .gemm import FP8Weight, fp8_linear, rowquant


@torch.library.custom_op("pneumam::lora_add", mutates_args=())
def _lora_add(
    base: torch.Tensor,
    x: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    a: torch.Tensor,
    bs: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    parts, o, r = bs.shape
    low = x @ a.T
    if parts == 1:
        return torch.addmm(base, low, bs[0].T), low
    out = base.clone()
    out.view(-1, parts, o).permute(1, 0, 2).baddbmm_(
        low.view(-1, parts, r).permute(1, 0, 2), bs.transpose(1, 2)
    )
    return out, low


@_lora_add.register_fake
def _(base, x, A, B, a, bs, scale):
    return torch.empty_like(base), x.new_empty(x.shape[0], a.shape[0])


def _setup_lora_add(ctx, inputs, output):
    _, x, A, B, a, bs, scale = inputs
    ctx.save_for_backward(x, output[1], a, bs)
    ctx.scale, ctx.dtypes = scale, (A.dtype, B.dtype)


def _bwd_lora_add(ctx, g, _):
    x, low, a, bs = ctx.saved_tensors
    parts, o, r = bs.shape
    n = x.shape[0]
    g = g.contiguous()
    gp = g.view(n, parts, o).permute(1, 0, 2)
    lowp = low.view(n, parts, r).permute(1, 0, 2)
    dlow = torch.bmm(gp, bs).permute(1, 0, 2).reshape(n, parts * r)
    dx = dlow @ a if ctx.needs_input_grad[1] else None
    dA = (dlow.T @ x).to(ctx.dtypes[0]) if ctx.needs_input_grad[2] else None
    dB = None
    if ctx.needs_input_grad[3]:
        dB = torch.baddbmm(
            torch.empty(parts, o, r, dtype=x.dtype, device=x.device),
            gp.transpose(1, 2),
            lowp,
            beta=0,
            alpha=ctx.scale,
        ).to(ctx.dtypes[1])
    return g, dx, dA, dB, None, None, None


_lora_add.register_autograd(_bwd_lora_add, setup_context=_setup_lora_add)


def lora_add(
    base: torch.Tensor,
    x: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    a: torch.Tensor,
    bs: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Add a low-rank update to the output of a linear layer. Differentiable.

    The output is split into ``parts`` equal slices and each slice gets its own rank-``r`` update:
    slice ``p`` gains ``scale * (x @ A_p.T) @ B[p].T``, where ``A_p`` is rows ``p * r`` to
    ``(p + 1) * r`` of ``A``. Gradients reach ``A`` and ``B`` although the product is computed
    from ``a`` and ``bs``.

    Args:
        base: Output of the base layer, shape ``[n, parts * o]``.
        x: Input of the base layer, shape ``[n, in]``.
        A: Trainable down projection, shape ``[parts * r, in]``.
        B: Trainable up projections, shape ``[parts, o, r]``.
        a: ``A`` cast to the dtype of ``x``, without a gradient.
        bs: ``B * scale`` cast to the dtype of ``x``, without a gradient.
        scale: The scale already folded into ``bs``.

    Returns:
        Tensor with the shape and dtype of ``base``.
    """
    return _lora_add(base, x, A, B, a, bs, scale)[0]


@torch.library.custom_op("pneumam::lora_linear", mutates_args=())
def _lora_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    a: torch.Tensor,
    bs: torch.Tensor,
    scale: float,
    wq: torch.Tensor | None,
    ws: torch.Tensor | None,
    tq: torch.Tensor | None,
    ts: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if wq is None or ws is None:
        base = x @ weight.T
    else:
        q, s = rowquant(x[None])
        base = fp8_linear(q, s, FP8Weight(wq, ws))[0].to(x.dtype)
    low = x @ a.T
    # one dense product adds every slice's update in a single pass over the output
    return torch.addmm(base, low, torch.block_diag(*bs.unbind(0)).T), low


@_lora_linear.register_fake
def _(x, weight, A, B, a, bs, scale, wq, ws, tq, ts):
    return x.new_empty(x.shape[0], weight.shape[0]), x.new_empty(x.shape[0], a.shape[0])


def _setup_lora_linear(ctx, inputs, output):
    x, weight, A, B, a, bs, scale, _, _, tq, ts = inputs
    ctx.fp8 = tq is not None and ts is not None
    ctx.save_for_backward(x, output[1], a, bs, *((tq, ts) if ctx.fp8 else (weight,)))
    ctx.scale, ctx.dtypes = scale, (A.dtype, B.dtype)


def _bwd_lora_linear(ctx, g, _):
    x, low, a, bs, *base = ctx.saved_tensors
    parts, o, r = bs.shape
    n = x.shape[0]
    g = g.contiguous()
    dlow = g @ torch.block_diag(*bs.unbind(0))
    dx = None
    if ctx.needs_input_grad[0]:
        dx = dlow @ a
        if ctx.fp8:
            gq, gs = rowquant(g[None])
            dx.add_(fp8_linear(gq, gs, FP8Weight(*base))[0])
        else:
            dx.addmm_(g, base[0])  # the base product accumulates into the same buffer
    dA = (dlow.T @ x).to(ctx.dtypes[0]) if ctx.needs_input_grad[2] else None
    dB = None
    if ctx.needs_input_grad[3]:
        dB = torch.baddbmm(
            torch.empty(parts, o, r, dtype=x.dtype, device=x.device),
            g.view(n, parts, o).permute(1, 2, 0),
            low.view(n, parts, r).permute(1, 0, 2),
            beta=0,
            alpha=ctx.scale,
        ).to(ctx.dtypes[1])
    return dx, None, dA, dB, None, None, None, None, None, None, None


_lora_linear.register_autograd(_bwd_lora_linear, setup_context=_setup_lora_linear)


def lora_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    a: torch.Tensor,
    bs: torch.Tensor,
    scale: float,
    w8: FP8Weight | None = None,
    w8_t: FP8Weight | None = None,
) -> torch.Tensor:
    """Apply a frozen linear layer together with its low-rank update. Differentiable.

    This is ``lora_add`` of ``x @ weight.T``, computed and differentiated as one step, which is
    faster. Gradients reach ``x``, ``A`` and ``B`` but not ``weight``.

    Args:
        x: Input, shape ``[n, in]``.
        weight: Frozen weight, shape ``[parts * o, in]``.
        A: Trainable down projection, shape ``[parts * r, in]``.
        B: Trainable up projections, shape ``[parts, o, r]``.
        a: ``A`` cast to the dtype of ``x``, without a gradient.
        bs: ``B * scale`` cast to the dtype of ``x``, without a gradient.
        scale: The scale already folded into ``bs``.
        w8: ``quantize_weight(weight)``. If given, the base product runs in FP8.
        w8_t: ``quantize_weight(weight.T)``. If given, so does its backward pass, which adds
            quantization noise to the gradient of ``x``.

    Returns:
        Tensor with the dtype of ``x``, shape ``[n, parts * o]``.

    Raises:
        RuntimeError: If FP8 is requested and ``x`` is not on a CUDA device.
    """
    wq, ws = (None, None) if w8 is None else w8
    tq, ts = (None, None) if w8_t is None else w8_t
    return _lora_linear(x, weight, A, B, a, bs, scale, wq, ws, tq, ts)[0]
