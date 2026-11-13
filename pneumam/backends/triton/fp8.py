import torch
import triton
from triton import language as tl

from .._fp8 import FP8, FP8Weight, quantize_weight  # noqa: F401

_ROW_CONFIGS = [triton.Config({}, num_warps=nw) for nw in (4, 8, 16)]


@triton.autotune(configs=_ROW_CONFIGS, key=["D"], cache_results=True)
@triton.jit
def _residual_adaln_fp8_kernel(
    x_ptr,
    res_ptr,
    gate_ptr,
    shift_ptr,
    scale_ptr,
    q_ptr,
    s_ptr,
    z_ptr,
    xr_ptr,
    xc_ptr,
    S,
    eps,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_X: tl.constexpr,
    X_SCALED: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    batch = row // S
    cols = tl.arange(0, BLOCK)
    mask = cols < D
    off = row * D + cols
    res = tl.load(res_ptr + off, mask=mask, other=0.0)
    if HAS_X:
        x = tl.load(x_ptr + off, mask=mask, other=0.0).to(tl.float32)
        if X_SCALED:
            xc = tl.load(xc_ptr + cols, mask=mask, other=0.0)
            x = x * tl.load(xr_ptr + row) * xc
        g = tl.load(gate_ptr + batch * D + cols, mask=mask, other=0.0).to(tl.float32)
        prod = (g * x).to(res_ptr.dtype.element_ty).to(tl.float32)
        z = (res.to(tl.float32) + prod).to(res_ptr.dtype.element_ty)
        tl.store(z_ptr + off, z, mask=mask)
    else:
        z = res
    zf = z.to(tl.float32)
    mean = tl.sum(zf, axis=0) / D
    d = tl.where(mask, zf - mean, 0.0)
    rstd = tl.rsqrt(tl.sum(d * d, axis=0) / D + eps)
    sc = tl.load(scale_ptr + batch * D + cols, mask=mask, other=0.0).to(tl.float32)
    sh = tl.load(shift_ptr + batch * D + cols, mask=mask, other=0.0).to(tl.float32)
    y = d * rstd * sc + sh
    row_scale = tl.maximum(tl.max(tl.abs(y), axis=0), 1e-12) / 448.0
    q = tl.minimum(tl.maximum(y / row_scale, -448.0), 448.0)
    tl.store(q_ptr + off, q.to(q_ptr.dtype.element_ty), mask=mask)
    tl.store(s_ptr + row, row_scale)


@torch.library.custom_op("pneumam::residual_adaln_fp8", mutates_args=())
def _residual_adaln_fp8(
    x: torch.Tensor,
    residual: torch.Tensor,
    gate: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    eps: float,
    x_rows: torch.Tensor | None,
    x_cols: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, S, D = residual.shape
    q = torch.empty(B, S, D, dtype=FP8, device=residual.device)
    s = torch.empty(B * S, 1, dtype=torch.float32, device=residual.device)
    z = torch.empty_like(residual)
    scaled = x_rows is not None and x_cols is not None
    _residual_adaln_fp8_kernel[(B * S,)](
        x,
        residual,
        gate,
        shift,
        scale,
        q,
        s,
        z,
        x_rows if scaled else s,
        x_cols if scaled else s,
        S,
        eps,
        D=D,
        BLOCK=triton.next_power_of_2(D),
        HAS_X=True,
        X_SCALED=scaled,
    )
    return q, s, z


@_residual_adaln_fp8.register_fake
def _(x, residual, gate, shift, scale, eps, x_rows, x_cols):
    B, S, D = residual.shape
    return (
        torch.empty(B, S, D, dtype=FP8, device=residual.device),
        torch.empty(B * S, 1, dtype=torch.float32, device=residual.device),
        torch.empty_like(residual),
    )


@torch.library.custom_op("pneumam::adaln_fp8", mutates_args=())
def _adaln_fp8(
    z: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    B, S, D = z.shape
    q = torch.empty(B, S, D, dtype=FP8, device=z.device)
    s = torch.empty(B * S, 1, dtype=torch.float32, device=z.device)
    _residual_adaln_fp8_kernel[(B * S,)](
        z,
        z,
        shift,
        shift,
        scale,
        q,
        s,
        z,
        s,
        s,
        S,
        eps,
        D=D,
        BLOCK=triton.next_power_of_2(D),
        HAS_X=False,
        X_SCALED=False,
    )
    return q, s


@_adaln_fp8.register_fake
def _(z, shift, scale, eps):
    B, S, D = z.shape
    return (
        torch.empty(B, S, D, dtype=FP8, device=z.device),
        torch.empty(B * S, 1, dtype=torch.float32, device=z.device),
    )


def residual_adaln_fp8(
    x: torch.Tensor,
    residual: torch.Tensor,
    gate: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    eps: float = 1e-6,
    x_scales: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``residual_adaln`` whose modulated output is returned as FP8 with a scale per token.

    Args:
        x: Sub-layer output, shape ``[B, S, D]``.
        residual: Residual stream, same shape and dtype as ``x``.
        gate: Gate, shape ``[B, D]``.
        shift: Shift, shape ``[B, D]``.
        scale: Scale (already including the +1), shape ``[B, D]``.
        eps: LayerNorm epsilon.
        x_scales: Float32 ``(rows, cols)`` of shapes ``[B * S, 1]`` and ``[D]``. If given, the
            sub-layer output is ``x * rows * cols``.

    Returns:
        ``(q, s, z)``: the FP8 output ``[B, S, D]``, its float32 token scales ``[B * S, 1]``, and the
        updated residual stream ``z`` ``[B, S, D]`` in the input dtype.

    Raises:
        ValueError: If ``x`` and ``residual`` differ in shape or dtype.
    """
    if x.shape != residual.shape or x.dtype != residual.dtype:
        raise ValueError("x and residual must match")
    return _residual_adaln_fp8(
        x.contiguous(),
        residual.contiguous(),
        gate.contiguous(),
        shift.contiguous(),
        scale.contiguous(),
        eps,
        *(x_scales or (None, None)),
    )


def adaln_fp8(
    z: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor, eps: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor]:
    """``adaln`` whose output is returned as FP8 with a scale per token.

    Args:
        z: Tokens, shape ``[B, S, D]``.
        shift: Shift, shape ``[B, D]``.
        scale: Scale (already including the +1), shape ``[B, D]``.
        eps: LayerNorm epsilon.

    Returns:
        ``(q, s)``: the FP8 output ``[B, S, D]`` and its float32 token scales ``[B * S, 1]``.
    """
    return _adaln_fp8(z.contiguous(), shift.contiguous(), scale.contiguous(), eps)


def fp8_linear(
    q: torch.Tensor, s: torch.Tensor, w: FP8Weight, fast_accum: bool = False
) -> torch.Tensor:
    """Multiply FP8 tokens by an FP8 weight. Runs on SM90 and SM100 GPUs only.

    Args:
        q: FP8 tokens, shape ``[B, S, K]``.
        s: Token scales from ``adaln_fp8`` or ``residual_adaln_fp8``, shape ``[B * S, 1]``.
        w: Weight from ``quantize_weight``, shape ``[N, K]``.
        fast_accum: Accumulate with reduced precision for extra speed.

    Returns:
        bfloat16 result, shape ``[B, S, N]``.
    """
    B, S, K = q.shape
    out = torch._scaled_mm(  # type: ignore
        q.view(B * S, K),
        w.q.t(),
        scale_a=s,
        scale_b=w.s.view(1, -1),
        out_dtype=torch.bfloat16,
        use_fast_accum=fast_accum,
    )
    return out.view(B, S, -1)


@triton.jit
def _gelu(x):
    z = x * 0.7071067811865476
    az = tl.abs(z)
    t = 1.0 / (1.0 + 0.3275911 * az)
    poly = (
        (((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t
        + 0.254829592
    ) * t
    y = 1.0 - poly * tl.exp(-az * az)
    return 0.5 * x * (1.0 + tl.where(z < 0, -y, y))


@triton.autotune(configs=_ROW_CONFIGS, key=["K"], cache_results=True)
@triton.jit
def _rowquant_kernel(
    x_ptr,
    q_ptr,
    s_ptr,
    r_ptr,
    c_ptr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
    GELU: tl.constexpr,
    SCALED: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < K
    x = tl.load(x_ptr + row * K + cols, mask=mask, other=0.0).to(tl.float32)
    if SCALED:
        x = x * tl.load(r_ptr + row) * tl.load(c_ptr + cols, mask=mask, other=0.0)
    if GELU:
        x = _gelu(x)
    row_scale = tl.maximum(tl.max(tl.abs(x), axis=0), 1e-12) / 448.0
    q = tl.minimum(tl.maximum(x / row_scale, -448.0), 448.0)
    tl.store(q_ptr + row * K + cols, q.to(q_ptr.dtype.element_ty), mask=mask)
    tl.store(s_ptr + row, row_scale)


@torch.library.custom_op("pneumam::rowquant", mutates_args=())
def _rowquant(
    x: torch.Tensor,
    gelu: bool,
    rows: torch.Tensor | None,
    cols: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, S, K = x.shape
    q = torch.empty(B, S, K, dtype=FP8, device=x.device)
    s = torch.empty(B * S, 1, dtype=torch.float32, device=x.device)
    scaled = rows is not None and cols is not None
    _rowquant_kernel[(B * S,)](
        x,
        q,
        s,
        rows if scaled else s,
        cols if scaled else s,
        K=K,
        BLOCK=triton.next_power_of_2(K),
        GELU=gelu,
        SCALED=scaled,
    )
    return q, s


@_rowquant.register_fake
def _(x, gelu, rows, cols):
    B, S, K = x.shape
    return (
        torch.empty(B, S, K, dtype=FP8, device=x.device),
        torch.empty(B * S, 1, dtype=torch.float32, device=x.device),
    )


def rowquant(
    x: torch.Tensor,
    gelu: bool = False,
    scales: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize tokens to FP8 with a scale per token, optionally applying GELU first.

    Args:
        x: Tokens, shape ``[B, S, K]``.
        gelu: Apply GELU before quantizing. It is within 1e-6 of the exact erf GELU.
        scales: Float32 ``(rows, cols)`` of shapes ``[B * S, 1]`` and ``[K]``. If given, the
            tokens are ``x * rows * cols``.

    Returns:
        ``(q, s)``: the FP8 tokens ``[B, S, K]`` and their float32 scales ``[B * S, 1]``.
        See ``fp8_linear``.
    """
    return _rowquant(x.contiguous(), gelu, *(scales or (None, None)))


@triton.jit  # not autotuned: tuning reruns a kernel, and this one works in place
def _scale_kernel(x_ptr, r_ptr, c_ptr, K: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < K
    x = tl.load(x_ptr + row * K + cols, mask=mask, other=0.0).to(tl.float32)
    y = x * tl.load(r_ptr + row) * tl.load(c_ptr + cols, mask=mask, other=0.0)
    tl.store(x_ptr + row * K + cols, y.to(x_ptr.dtype.element_ty), mask=mask)


@torch.library.custom_op("pneumam::scale_rows_cols_", mutates_args=("x",))
def _scale(x: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor) -> None:
    M, K = x.shape
    _scale_kernel[(M,)](x, rows, cols, K=K, BLOCK=triton.next_power_of_2(K))  # type: ignore


def scale_rows_cols_(
    x: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor
) -> torch.Tensor:
    """Multiply a matrix by a scale per row and a scale per column, in place.

    Args:
        x: Contiguous matrix, shape ``[M, K]``.
        rows: Float32 row scales, shape ``[M, 1]``.
        cols: Float32 column scales, shape ``[K]``.

    Returns:
        ``x``, now holding ``x * rows * cols``.
    """
    _scale(x, rows.contiguous(), cols.contiguous())
    return x
