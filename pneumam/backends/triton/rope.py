import torch
import triton
from triton import language as tl

from .._fp8 import FP8
from .common import tuned_rows

_HEAD_CONFIGS = [
    triton.Config({"HPP": hpp}, num_warps=nw) for hpp in (2, 4, 8, 16) for nw in (4, 8)
]


@triton.jit
def _section(
    src_ptr,
    dst_ptr,
    w_ptr,
    cos_ptr,
    sin_ptr,
    token,
    position,
    group,
    eps,
    scale,
    SRC_ROW: tl.constexpr,
    SRC_COL: tl.constexpr,
    DST_ROW: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HPP: tl.constexpr,
    HALF: tl.constexpr,
    NORM: tl.constexpr,
    ROTATE: tl.constexpr,
    FP8: tl.constexpr,
):
    head = group * HPP + tl.arange(0, HPP)[:, None]
    dim = tl.arange(0, HALF)[None, :]
    mask = head < NUM_HEADS
    column = head * (2 * HALF) + dim
    src = src_ptr + token * SRC_ROW + SRC_COL + column
    dst = dst_ptr + token * DST_ROW + column

    a = tl.load(src, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(src + HALF, mask=mask, other=0.0).to(tl.float32)
    if NORM:
        rstd = tl.rsqrt(
            (tl.sum(a * a, axis=1) + tl.sum(b * b, axis=1)) / (2 * HALF) + eps
        )[:, None]
        wa = tl.load(w_ptr + dim).to(tl.float32)
        wb = tl.load(w_ptr + HALF + dim).to(tl.float32)
        a = (a * rstd * wa).to(src_ptr.dtype.element_ty).to(tl.float32)
        b = (b * rstd * wb).to(src_ptr.dtype.element_ty).to(tl.float32)
    if ROTATE:
        cos = tl.load(cos_ptr + position * HALF + dim).to(tl.float32)
        sin = tl.load(sin_ptr + position * HALF + dim).to(tl.float32)
        out_a = a * cos - b * sin
        out_b = b * cos + a * sin
    else:
        out_a = a
        out_b = b
    if FP8:
        out_a = tl.minimum(tl.maximum(out_a * scale, -448.0), 448.0)
        out_b = tl.minimum(tl.maximum(out_b * scale, -448.0), 448.0)
    tl.store(dst, out_a.to(dst_ptr.dtype.element_ty), mask=mask)
    tl.store(dst + HALF, out_b.to(dst_ptr.dtype.element_ty), mask=mask)


@triton.autotune(
    configs=_HEAD_CONFIGS, key=["HEADS", "HALF", "FP8"], cache_results=True
)
@triton.jit
def _qkv_norm_rope_kernel(
    qkv_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    qw_ptr,
    kw_ptr,
    cos_ptr,
    sin_ptr,
    S,
    eps,
    q_scale,
    k_scale,
    v_scale,
    HEADS: tl.constexpr,
    HALF: tl.constexpr,
    HPP: tl.constexpr,
    FP8: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    group = tl.program_id(1)
    position = token % S
    width: tl.constexpr = HEADS * 2 * HALF
    row: tl.constexpr = 3 * width
    gq: tl.constexpr = (HEADS + HPP - 1) // HPP
    if group < gq:
        _section(
            qkv_ptr,
            q_ptr,
            qw_ptr,
            cos_ptr,
            sin_ptr,
            token,
            position,
            group,
            eps,
            q_scale,
            row,
            tl.constexpr(0),
            width,
            HEADS,
            HPP,
            HALF,
            tl.constexpr(True),
            tl.constexpr(True),
            FP8,
        )
    elif group < 2 * gq:
        _section(
            qkv_ptr,
            k_ptr,
            kw_ptr,
            cos_ptr,
            sin_ptr,
            token,
            position,
            group - gq,
            eps,
            k_scale,
            row,
            width,
            width,
            HEADS,
            HPP,
            HALF,
            tl.constexpr(True),
            tl.constexpr(True),
            FP8,
        )
    else:
        _section(
            qkv_ptr,
            v_ptr,
            kw_ptr,
            cos_ptr,
            sin_ptr,
            token,
            position,
            group - 2 * gq,
            eps,
            v_scale,
            row,
            2 * width,
            width,
            HEADS,
            HPP,
            HALF,
            tl.constexpr(False),
            tl.constexpr(False),
            FP8,
        )


@triton.autotune(
    configs=_HEAD_CONFIGS, key=["HEADS", "HALF", "FP8"], cache_results=True
)
@triton.jit
def _head_norm_kernel(
    x_ptr,
    y_ptr,
    w_ptr,
    eps,
    scale,
    HEADS: tl.constexpr,
    HALF: tl.constexpr,
    HPP: tl.constexpr,
    FP8: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    group = tl.program_id(1)
    width: tl.constexpr = HEADS * 2 * HALF
    _section(
        x_ptr,
        y_ptr,
        w_ptr,
        w_ptr,
        w_ptr,
        token,
        token,
        group,
        eps,
        scale,
        width,
        tl.constexpr(0),
        width,
        HEADS,
        HPP,
        HALF,
        tl.constexpr(True),
        tl.constexpr(False),
        FP8,
    )


@torch.library.custom_op("pneumam::qkv_norm_rope", mutates_args=())
def _qkv_norm_rope(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    heads: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, S, _ = qkv.shape
    half = cos.shape[1]
    q = qkv.new_empty((B, S, heads, 2 * half))
    k = torch.empty_like(q)
    grid = lambda meta: (B * S, 2 * triton.cdiv(heads, meta["HPP"]))
    _qkv_norm_rope_kernel[grid](
        qkv,
        q,
        k,
        q,
        q_weight,
        k_weight,
        cos,
        sin,
        S,
        eps,
        1.0,
        1.0,
        1.0,
        HEADS=heads,
        HALF=half,
        FP8=False,
    )
    return q, k


@_qkv_norm_rope.register_fake
def _(qkv, q_weight, k_weight, cos, sin, heads, eps):
    B, S, _ = qkv.shape
    q = qkv.new_empty((B, S, heads, 2 * cos.shape[1]))
    return q, torch.empty_like(q)


@torch.library.custom_op("pneumam::qkv_norm_rope_fp8", mutates_args=())
def _qkv_norm_rope_fp8(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    heads: int,
    q_scale: float,
    k_scale: float,
    v_scale: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, S, _ = qkv.shape
    half = cos.shape[1]
    q = torch.empty((B, S, heads, 2 * half), dtype=FP8, device=qkv.device)
    k = torch.empty_like(q)
    v = torch.empty_like(q)
    grid = lambda meta: (B * S, 3 * triton.cdiv(heads, meta["HPP"]))
    _qkv_norm_rope_kernel[grid](
        qkv,
        q,
        k,
        v,
        q_weight,
        k_weight,
        cos,
        sin,
        S,
        eps,
        q_scale,
        k_scale,
        v_scale,
        HEADS=heads,
        HALF=half,
        FP8=True,
    )
    return q, k, v


@_qkv_norm_rope_fp8.register_fake
def _(qkv, q_weight, k_weight, cos, sin, heads, q_scale, k_scale, v_scale, eps):
    B, S, _ = qkv.shape
    q = torch.empty((B, S, heads, 2 * cos.shape[1]), dtype=FP8, device=qkv.device)
    return q, torch.empty_like(q), torch.empty_like(q)


@torch.library.custom_op("pneumam::head_norm", mutates_args=())
def _head_norm(
    x: torch.Tensor, weight: torch.Tensor, heads: int, eps: float
) -> torch.Tensor:
    B, S, _ = x.shape
    hd = weight.numel()
    y = x.new_empty((B, S, heads, hd))
    grid = lambda meta: (B * S, triton.cdiv(heads, meta["HPP"]))
    _head_norm_kernel[grid](
        x, y, weight, eps, 1.0, HEADS=heads, HALF=hd // 2, FP8=False
    )
    return y


@_head_norm.register_fake
def _(x, weight, heads, eps):
    return x.new_empty((x.shape[0], x.shape[1], heads, weight.numel()))


@torch.library.custom_op("pneumam::head_norm_fp8", mutates_args=())
def _head_norm_fp8(
    x: torch.Tensor, weight: torch.Tensor, heads: int, scale: float, eps: float
) -> torch.Tensor:
    B, S, _ = x.shape
    hd = weight.numel()
    y = torch.empty((B, S, heads, hd), dtype=FP8, device=x.device)
    grid = lambda meta: (B * S, triton.cdiv(heads, meta["HPP"]))
    _head_norm_kernel[grid](
        x, y, weight, eps, scale, HEADS=heads, HALF=hd // 2, FP8=True
    )
    return y


@_head_norm_fp8.register_fake
def _(x, weight, heads, scale, eps):
    shape = (x.shape[0], x.shape[1], heads, weight.numel())
    return torch.empty(shape, dtype=FP8, device=x.device)


_BWD_CONFIGS = [triton.Config({}, num_warps=nw) for nw in (4, 8)]


@triton.jit
def _section_bwd(
    src_ptr,
    dout_ptr,
    dsrc_ptr,
    w_ptr,
    cos_ptr,
    sin_ptr,
    token,
    position,
    group,
    eps,
    valid,
    SRC_ROW: tl.constexpr,
    SRC_COL: tl.constexpr,
    DOUT_ROW: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HPP: tl.constexpr,
    HALF: tl.constexpr,
    ROTATE: tl.constexpr,
):
    head = group * HPP + tl.arange(0, HPP)[:, None]
    dim = tl.arange(0, HALF)[None, :]
    mask = (head < NUM_HEADS) & (dim < tl.where(valid, HALF, 0))
    column = head * (2 * HALF) + dim
    src = src_ptr + token * SRC_ROW + SRC_COL + column
    dsrc = dsrc_ptr + token * SRC_ROW + SRC_COL + column
    dout = dout_ptr + token * DOUT_ROW + column

    a = tl.load(src, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(src + HALF, mask=mask, other=0.0).to(tl.float32)
    da = tl.load(dout, mask=mask, other=0.0).to(tl.float32)
    db = tl.load(dout + HALF, mask=mask, other=0.0).to(tl.float32)
    if ROTATE:
        cos = tl.load(cos_ptr + position * HALF + dim).to(tl.float32)
        sin = tl.load(sin_ptr + position * HALF + dim).to(tl.float32)
        da, db = da * cos + db * sin, db * cos - da * sin
    rstd = tl.rsqrt((tl.sum(a * a, axis=1) + tl.sum(b * b, axis=1)) / (2 * HALF) + eps)[
        :, None
    ]
    wa = tl.load(w_ptr + dim).to(tl.float32)
    wb = tl.load(w_ptr + HALF + dim).to(tl.float32)
    xa = a * rstd
    xb = b * rstd
    ua = da * wa
    ub = db * wb
    c = ((tl.sum(ua * xa, axis=1) + tl.sum(ub * xb, axis=1)) / (2 * HALF))[:, None]
    tl.store(dsrc, (rstd * (ua - xa * c)).to(dsrc_ptr.dtype.element_ty), mask=mask)
    tl.store(
        dsrc + HALF, (rstd * (ub - xb * c)).to(dsrc_ptr.dtype.element_ty), mask=mask
    )
    return tl.sum(da * xa, axis=0), tl.sum(db * xb, axis=0)


@triton.autotune(
    configs=_BWD_CONFIGS, key=["HEADS", "HALF", "ROWS"], cache_results=True
)
@triton.jit
def _qkv_norm_rope_bwd_kernel(
    qkv_ptr,
    dq_ptr,
    dk_ptr,
    dv_ptr,
    dqkv_ptr,
    qw_ptr,
    kw_ptr,
    cos_ptr,
    sin_ptr,
    pw_ptr,
    S,
    eps,
    HEADS: tl.constexpr,
    HALF: tl.constexpr,
    HPP: tl.constexpr,
    ROWS: tl.constexpr,
    WG: tl.constexpr,
):
    chunk = tl.program_id(0)
    group = tl.program_id(1)
    batch = chunk // ((S + ROWS - 1) // ROWS)
    first = (chunk % ((S + ROWS - 1) // ROWS)) * ROWS
    width: tl.constexpr = HEADS * 2 * HALF
    row: tl.constexpr = 3 * width
    gq: tl.constexpr = (HEADS + HPP - 1) // HPP
    if group >= 2 * gq:
        for r in range(ROWS):
            s = first + r
            if s < S:
                _section(
                    dv_ptr,
                    dqkv_ptr + 2 * width,
                    qw_ptr,
                    cos_ptr,
                    sin_ptr,
                    (batch * S + s).to(tl.int64),
                    s,
                    group - 2 * gq,
                    eps,
                    1.0,
                    width,
                    tl.constexpr(0),
                    row,
                    HEADS,
                    HPP,
                    HALF,
                    tl.constexpr(False),
                    tl.constexpr(False),
                    tl.constexpr(False),
                )
    else:
        is_q = group < gq
        local = tl.where(is_q, group, group - gq)
        acc_a = tl.zeros([HALF], dtype=tl.float32)
        acc_b = tl.zeros([HALF], dtype=tl.float32)
        for r in range(ROWS):
            s = first + r
            token = (batch * S + s).to(tl.int64)
            position = tl.minimum(s, S - 1)
            if is_q:
                wa, wb = _section_bwd(
                    qkv_ptr,
                    dq_ptr,
                    dqkv_ptr,
                    qw_ptr,
                    cos_ptr,
                    sin_ptr,
                    token,
                    position,
                    local,
                    eps,
                    s < S,
                    row,
                    tl.constexpr(0),
                    width,
                    HEADS,
                    HPP,
                    HALF,
                    tl.constexpr(True),
                )
            else:
                wa, wb = _section_bwd(
                    qkv_ptr,
                    dk_ptr,
                    dqkv_ptr,
                    kw_ptr,
                    cos_ptr,
                    sin_ptr,
                    token,
                    position,
                    local,
                    eps,
                    s < S,
                    row,
                    width,
                    width,
                    HEADS,
                    HPP,
                    HALF,
                    tl.constexpr(True),
                )
            if WG:
                acc_a += wa
                acc_b += wb
        if WG:
            dim = tl.arange(0, HALF)
            out = pw_ptr + (chunk * tl.num_programs(1) + group) * (2 * HALF)
            tl.store(out + dim, acc_a)
            tl.store(out + HALF + dim, acc_b)


@triton.autotune(
    configs=_BWD_CONFIGS, key=["HEADS", "HALF", "ROWS"], cache_results=True
)
@triton.jit
def _head_norm_bwd_kernel(
    x_ptr,
    dy_ptr,
    dx_ptr,
    w_ptr,
    pw_ptr,
    S,
    eps,
    HEADS: tl.constexpr,
    HALF: tl.constexpr,
    HPP: tl.constexpr,
    ROWS: tl.constexpr,
    WG: tl.constexpr,
):
    chunk = tl.program_id(0)
    group = tl.program_id(1)
    batch = chunk // ((S + ROWS - 1) // ROWS)
    first = (chunk % ((S + ROWS - 1) // ROWS)) * ROWS
    width: tl.constexpr = HEADS * 2 * HALF
    acc_a = tl.zeros([HALF], dtype=tl.float32)
    acc_b = tl.zeros([HALF], dtype=tl.float32)
    for r in range(ROWS):
        s = first + r
        token = (batch * S + s).to(tl.int64)
        wa, wb = _section_bwd(
            x_ptr,
            dy_ptr,
            dx_ptr,
            w_ptr,
            w_ptr,
            w_ptr,
            token,
            token,
            group,
            eps,
            s < S,
            width,
            tl.constexpr(0),
            width,
            HEADS,
            HPP,
            HALF,
            tl.constexpr(False),
        )
        if WG:
            acc_a += wa
            acc_b += wb
    if WG:
        dim = tl.arange(0, HALF)
        out = pw_ptr + (chunk * tl.num_programs(1) + group) * (2 * HALF)
        tl.store(out + dim, acc_a)
        tl.store(out + HALF + dim, acc_b)


def _bwd_hpp(heads):
    return min(4, triton.next_power_of_2(heads))


def _head_norm_bwd(x, dy, weight, heads, eps, wgrad=True):
    B, S, _ = x.shape
    hd = weight.numel()
    hpp = _bwd_hpp(heads)
    groups = triton.cdiv(heads, hpp)
    dy = dy.contiguous()

    def launch(rows):
        chunks = B * triton.cdiv(S, rows)
        dx = torch.empty_like(x)
        pw = (
            torch.empty(chunks, groups, hd, dtype=torch.float32, device=x.device)
            if wgrad
            else dx
        )
        _head_norm_bwd_kernel[(chunks, groups)](
            x,
            dy,
            dx,
            weight,
            pw,
            S,
            eps,
            HEADS=heads,
            HALF=hd // 2,
            HPP=hpp,
            ROWS=rows,
            WG=wgrad,
        )
        return dx, pw

    rows = tuned_rows(("head_norm_bwd", B, S, heads, hd, wgrad, x.dtype), launch)
    dx, pw = launch(rows)
    return dx, (pw.sum((0, 1)).to(weight.dtype) if wgrad else weight.new_empty(0))


@torch.library.custom_op("pneumam::head_norm_bwd", mutates_args=())
def _head_norm_bwd_op(
    x: torch.Tensor,
    dy: torch.Tensor,
    weight: torch.Tensor,
    heads: int,
    eps: float,
    wgrad: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _head_norm_bwd(x, dy, weight, heads, eps, wgrad)


@_head_norm_bwd_op.register_fake
def _(x, dy, weight, heads, eps, wgrad):
    return torch.empty_like(x), (
        torch.empty_like(weight) if wgrad else weight.new_empty(0)
    )


def _head_norm_setup(ctx, inputs, output):
    x, weight, heads, eps = inputs
    ctx.save_for_backward(x, weight)
    ctx.heads = heads
    ctx.eps = eps


def _head_norm_backward(ctx, dy):
    x, weight = ctx.saved_tensors
    wgrad = ctx.needs_input_grad[1]
    dx, dw = _head_norm_bwd_op(x, dy, weight, ctx.heads, ctx.eps, wgrad)
    return dx, (dw if wgrad else None), None, None


torch.library.register_autograd(
    "pneumam::head_norm", _head_norm_backward, setup_context=_head_norm_setup
)


@torch.library.custom_op("pneumam::qkv_norm_rope_v", mutates_args=())
def _qkv_norm_rope_v(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    heads: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, S, _ = qkv.shape
    half = cos.shape[1]
    q = qkv.new_empty((B, S, heads, 2 * half))
    k = torch.empty_like(q)
    v = torch.empty_like(q)
    grid = lambda meta: (B * S, 3 * triton.cdiv(heads, meta["HPP"]))
    _qkv_norm_rope_kernel[grid](
        qkv,
        q,
        k,
        v,
        q_weight,
        k_weight,
        cos,
        sin,
        S,
        eps,
        1.0,
        1.0,
        1.0,
        HEADS=heads,
        HALF=half,
        FP8=False,
    )
    return q, k, v


@_qkv_norm_rope_v.register_fake
def _(qkv, q_weight, k_weight, cos, sin, heads, eps):
    B, S, _ = qkv.shape
    q = qkv.new_empty((B, S, heads, 2 * cos.shape[1]))
    return q, torch.empty_like(q), torch.empty_like(q)


def _qkv_norm_rope_setup(ctx, inputs, output):
    qkv, q_weight, k_weight, cos, sin, heads, eps = inputs
    ctx.save_for_backward(qkv, q_weight, k_weight, cos, sin)
    ctx.heads = heads
    ctx.eps = eps


def _qkv_norm_rope_bwd(
    qkv, dq, dk, dv, q_weight, k_weight, cos, sin, heads, eps, wgrad=True
):
    B, S, _ = qkv.shape
    half = cos.shape[1]
    hpp = _bwd_hpp(heads)
    gq = triton.cdiv(heads, hpp)
    dq, dk, dv = dq.contiguous(), dk.contiguous(), dv.contiguous()

    def launch(rows):
        chunks = B * triton.cdiv(S, rows)
        dqkv = torch.empty_like(qkv)
        pw = (
            torch.empty(
                chunks, 3 * gq, 2 * half, dtype=torch.float32, device=qkv.device
            )
            if wgrad
            else dqkv
        )
        _qkv_norm_rope_bwd_kernel[(chunks, 3 * gq)](
            qkv,
            dq,
            dk,
            dv,
            dqkv,
            q_weight,
            k_weight,
            cos,
            sin,
            pw,
            S,
            eps,
            HEADS=heads,
            HALF=half,
            HPP=hpp,
            ROWS=rows,
            WG=wgrad,
        )
        return dqkv, pw

    rows = tuned_rows(
        ("qkv_norm_rope_bwd", B, S, heads, half, wgrad, qkv.dtype), launch
    )
    dqkv, pw = launch(rows)
    if not wgrad:
        return dqkv, q_weight.new_empty(0), k_weight.new_empty(0)
    dqw = pw[:, :gq].sum((0, 1)).to(q_weight.dtype)
    dkw = pw[:, gq : 2 * gq].sum((0, 1)).to(k_weight.dtype)
    return dqkv, dqw, dkw


@torch.library.custom_op("pneumam::qkv_norm_rope_bwd", mutates_args=())
def _qkv_norm_rope_bwd_op(
    qkv: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    heads: int,
    eps: float,
    wgrad: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _qkv_norm_rope_bwd(
        qkv, dq, dk, dv, q_weight, k_weight, cos, sin, heads, eps, wgrad
    )


@_qkv_norm_rope_bwd_op.register_fake
def _(qkv, dq, dk, dv, q_weight, k_weight, cos, sin, heads, eps, wgrad):
    p = torch.empty_like if wgrad else (lambda t: t.new_empty(0))
    return torch.empty_like(qkv), p(q_weight), p(k_weight)


def _qkv_norm_rope_backward(ctx, dq, dk, dv):
    qkv, q_weight, k_weight, cos, sin = ctx.saved_tensors
    wgrad = any(ctx.needs_input_grad[1:3])
    dqkv, dqw, dkw = _qkv_norm_rope_bwd_op(
        qkv, dq, dk, dv, q_weight, k_weight, cos, sin, ctx.heads, ctx.eps, wgrad
    )
    if not wgrad:
        dqw = dkw = None
    return dqkv, dqw, dkw, None, None, None, None


torch.library.register_autograd(
    "pneumam::qkv_norm_rope_v",
    _qkv_norm_rope_backward,
    setup_context=_qkv_norm_rope_setup,
)


def _check_qkv(qkv, cos, sin, heads):
    if (
        not qkv.is_contiguous()
        or qkv.ndim != 3
        or qkv.shape[-1] != 3 * heads * 2 * cos.shape[1]
    ):
        raise ValueError("qkv must be contiguous [B, S, 3 * heads * head_dim]")
    if (
        cos.shape != sin.shape
        or cos.shape[0] != qkv.shape[1]
        or (cos.shape[1] & (cos.shape[1] - 1))
    ):
        raise ValueError("cos/sin must be [S, head_dim // 2] with a power-of-two width")


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
        ``(q, k, v)``, each ``[B, S, heads, hd]`` with the dtype of ``qkv``. Without autograd,
        ``v`` is a strided view of ``qkv``.

    Raises:
        ValueError: If a shape requirement above is not met.
    """
    _check_qkv(qkv, cos, sin, heads)
    args = (
        qkv,
        q_weight.contiguous(),
        k_weight.contiguous(),
        cos.contiguous(),
        sin.contiguous(),
        heads,
        eps,
    )
    if torch.is_grad_enabled() and (qkv.requires_grad or q_weight.requires_grad):
        return _qkv_norm_rope_v(*args)
    q, k = _qkv_norm_rope(*args)
    # V needs no norm or rotation: return a strided view of qkv instead of copying it.
    B, S, _ = qkv.shape
    v = qkv.view(B, S, 3, heads, 2 * cos.shape[1])[:, :, 2]
    return q, k, v


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

    Values beyond 448 in magnitude after scaling are clipped. See ``norm_scale`` for a scale
    that never clips Q or K.

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
        ValueError: If a shape requirement above is not met.
    """
    _check_qkv(qkv, cos, sin, heads)
    return _qkv_norm_rope_fp8(
        qkv,
        q_weight.contiguous(),
        k_weight.contiguous(),
        cos.contiguous(),
        sin.contiguous(),
        heads,
        q_scale,
        k_scale,
        v_scale,
        eps,
    )


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
        ValueError: If ``x`` is not contiguous ``[B, S, heads * hd]``.
    """
    if not x.is_contiguous() or x.shape[-1] != heads * weight.numel():
        raise ValueError("x must be contiguous [B, S, heads * head_dim]")
    return _head_norm(x, weight.contiguous(), heads, eps)


def head_norm_fp8(
    x: torch.Tensor,
    weight: torch.Tensor,
    heads: int,
    scale: float,
    eps: float = 1e-5,
) -> torch.Tensor:
    """``head_norm`` whose output is FP8, multiplied by a fixed scale. Forward only.

    Values beyond 448 in magnitude after scaling are clipped. See ``norm_scale``.

    Args:
        x: Tokens, contiguous, shape ``[B, S, heads * hd]``.
        weight: Per-dimension scale, shape ``[hd]``.
        heads: Number of heads.
        scale: Factor applied before quantizing.
        eps: Epsilon added to the mean square.

    Returns:
        FP8 tensor of shape ``[B, S, heads, hd]``.

    Raises:
        ValueError: If ``x`` is not contiguous ``[B, S, heads * hd]``.
    """
    if not x.is_contiguous() or x.shape[-1] != heads * weight.numel():
        raise ValueError("x must be contiguous [B, S, heads * head_dim]")
    return _head_norm_fp8(x, weight.contiguous(), heads, scale, eps)
