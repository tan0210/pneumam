import torch


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
