import json
import math
import os

import torch

from ._fp8 import FP8, FP8_MAX, FP8Weight

try:
    import warpy_kernels as _wk

    available = True
except Exception:  # noqa: BLE001  # cutlass can fail to import in ways other than ImportError
    _wk = None
    available = False

_BM = 128
_DTYPES = (torch.bfloat16, torch.float16)
TUNE_CACHE_PATH = os.environ.get(
    "PNEUMAM_TUNE_CACHE", os.path.expanduser("~/.cache/pneumam/warpy_gemm.json")
)
_tuned: dict[str, dict] | None = None


def supports(feature: str, device: torch.device) -> bool:
    """Tell whether the library has a kernel for a feature on a GPU.

    Args:
        feature: A ``warpy_kernels.supports`` feature, such as ``"gemm"`` or ``"gemm_fp8"``.
        device: CUDA device to check.

    Returns:
        True if the library is installed and runs ``feature`` on ``device``.
    """
    return available and _wk.supports(feature, device)  # type: ignore


def _weight_fits(w: torch.Tensor) -> bool:
    n, k = w.shape
    return (
        w.is_contiguous() and n % 128 == 0 and k % (128 if w.dtype == FP8 else 64) == 0
    )


def fits(x: torch.Tensor, *weights: torch.Tensor) -> bool:
    """Tell whether this backend can run these operands.

    Args:
        x: Activations, shape ``[..., K]``.
        *weights: The ``[N, K]`` weights ``x`` is multiplied by. For an FP8 product, pass the FP8
            tokens and ``FP8Weight.q``.

    Returns:
        True if the library is installed, everything shares one of bf16, fp16 or FP8, and each weight is
        contiguous with N a multiple of 128 and K a multiple of 64, or of 128 for FP8.
    """
    return (
        available
        and (x.dtype in _DTYPES or x.dtype == FP8)
        and all(w.dtype == x.dtype and _weight_fits(w) for w in weights)
    )


def _key(a: torch.Tensor, w: torch.Tensor, act: str, trans: bool = False) -> str:
    gpu = torch.cuda.get_device_name(a.device)
    m = max(a.shape[0] // _BM, 1) * _BM  # the rows of the main launch
    key = f"{gpu}|{a.dtype}|m={m}|n={w.shape[0]}|k={w.shape[1]}|act={act}"
    return f"{key}|trans" if trans else key


def _tuned_config(key: str) -> dict:
    global _tuned
    if _tuned is None:
        try:
            with open(TUNE_CACHE_PATH) as f:
                _tuned = json.load(f)
        except (OSError, ValueError):
            _tuned = {}
    return _tuned.get(key, {})  # type: ignore


def _store_config(key: str, cfg: dict) -> None:
    _tuned_config(key)
    _tuned[key] = cfg  # type: ignore
    try:
        os.makedirs(os.path.dirname(TUNE_CACHE_PATH), exist_ok=True)
        with open(TUNE_CACHE_PATH, "w") as f:
            json.dump(_tuned, f, indent=1, sort_keys=True)
    except OSError:
        pass


def _pad_rows(a: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    a = a.contiguous()
    mp = -(-m // _BM) * _BM
    if mp == m:
        return a
    # Copy as raw bytes: FP8 tensors support very few ops, and zero bytes are 0.0 in every dtype.
    buf = torch.zeros(mp, k * a.element_size(), dtype=torch.uint8, device=a.device)
    buf[:m] = a.view(torch.uint8)
    return buf.view(a.dtype)


def _launch(a, w, act, out=None, sa=None, sb=None) -> torch.Tensor:
    tuned = _tuned_config(_key(a, w, act))
    y = _wk.gemm(a, w, out=out, act=act, scale_a=sa, scale_b=sb, **tuned)  # type: ignore
    return y[0] if act == "gelu" else y  # type: ignore


def _pad_scale(s: torch.Tensor) -> torch.Tensor:
    mp = -(-s.shape[0] // _BM) * _BM
    out = torch.ones(mp, dtype=s.dtype, device=s.device)
    out[: s.shape[0]] = s
    return out


def _run(
    a: torch.Tensor,
    w: torch.Tensor,
    act: str = "",
    sa: torch.Tensor | None = None,
    sb: torch.Tensor | None = None,
) -> torch.Tensor:
    m = a.shape[0]
    a, w = a.detach().contiguous(), w.detach()  # the kernel cannot take graph tensors
    if sa is not None:
        sa = sa.detach().reshape(-1).contiguous()
    whole = m // _BM * _BM
    if whole == m:
        return _launch(a, w, act, None, sa, sb)
    dtype = torch.bfloat16 if a.dtype == FP8 else a.dtype
    y = torch.empty(m, w.shape[0], dtype=dtype, device=a.device)
    if whole:
        bulk = None if sa is None else sa[:whole]
        _launch(
            a[:whole], w, act, y[:whole], bulk, sb
        )  # the bulk runs in place, uncopied
    tail = None if sa is None else _pad_scale(sa[whole:])
    y[whole:] = _launch(_pad_rows(a[whole:]), w, act, None, tail, sb)[: m - whole]
    return y


@torch.library.custom_op("pneumam::gemm_warpy", mutates_args=())
def _gemm(a: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return _run(a, w)


@_gemm.register_fake
def _(a, w):
    return a.new_empty(a.shape[0], w.shape[0])


def _gemm_setup(ctx, inputs, output):
    ctx.save_for_backward(*inputs)


def _gemm_backward(ctx, g):
    a, w = ctx.saved_tensors
    da = g @ w if ctx.needs_input_grad[0] else None
    dw = g.T @ a if ctx.needs_input_grad[1] else None
    return da, dw


_gemm.register_autograd(_gemm_backward, setup_context=_gemm_setup)


@torch.library.custom_op("pneumam::linear_gelu_warpy", mutates_args=())
def _linear_gelu(a: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return _run(a, w, "gelu")


@_linear_gelu.register_fake
def _(a, w):
    return a.new_empty(a.shape[0], w.shape[0])


@torch.library.custom_op("pneumam::fp8_gemm_warpy", mutates_args=())
def _fp8_gemm(q: torch.Tensor, wq: torch.Tensor) -> torch.Tensor:
    return _run(q, wq)


@_fp8_gemm.register_fake
def _(q, wq):
    return torch.empty(q.shape[0], wq.shape[0], dtype=torch.bfloat16, device=q.device)


@torch.library.custom_op("pneumam::fp8_linear_warpy", mutates_args=())
def _fp8_linear(
    q: torch.Tensor, s: torch.Tensor, wq: torch.Tensor, ws: torch.Tensor
) -> torch.Tensor:
    return _run(q, wq, "", s, ws)


@_fp8_linear.register_fake
def _(q, s, wq, ws):
    return torch.empty(q.shape[0], wq.shape[0], dtype=torch.bfloat16, device=q.device)


def gemm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Compute ``x @ weight.T``. Differentiable.

    Args:
        x: Activations accepted by ``fits``, bf16 or fp16, shape ``[..., K]``.
        weight: Weight with the dtype of ``x``, shape ``[N, K]``.

    Returns:
        Tensor with the dtype of ``x``, shape ``[..., N]``.
    """
    y = _gemm(x.reshape(-1, x.shape[-1]), weight)
    return y.view(*x.shape[:-1], weight.shape[0])


def linear_gelu(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Compute ``gelu(x @ weight.T)`` with the exact erf GELU and no bias. Forward only.

    Args:
        x: Activations accepted by ``fits``, bf16 or fp16, shape ``[..., K]``.
        weight: Weight with the dtype of ``x``, shape ``[N, K]``.

    Returns:
        Tensor with the dtype of ``x``, shape ``[..., N]``.
    """
    y = _linear_gelu(x.reshape(-1, x.shape[-1]), weight)
    return y.view(*x.shape[:-1], weight.shape[0])


@torch.library.custom_op("pneumam::mlp_warpy_train", mutates_args=())
def _mlp_train(
    x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    x, w1, w2 = x.detach().contiguous(), w1.detach(), w2.detach()
    tuned = _tuned_config(_key(x, w1, "gelu"))
    post, pre = _wk.gemm(x, w1, act="gelu", store_pre=True, **tuned)  # type: ignore
    return _launch(post, w2, ""), pre


@_mlp_train.register_fake
def _(x, w1, w2):
    return x.new_empty(x.shape[0], w2.shape[0]), x.new_empty(x.shape[0], w1.shape[0])


@torch.library.custom_op("pneumam::mlp_warpy_bwd", mutates_args=())
def _mlp_bwd(
    dout: torch.Tensor,
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    pre: torch.Tensor,
    need_dx: bool,
    need_dw1: bool,
    need_dw2: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dout, x, pre = (t.detach().contiguous() for t in (dout.to(w2.dtype), x, pre))
    w1, w2 = w1.detach(), w2.detach()
    # one kernel multiplies by the weight and by the GELU derivative at the stored input
    tuned = _tuned_config(_key(dout, w2, "dgelu", True))
    dpre = _wk.gemm(dout, w2, b_trans=True, act="dgelu", pre=pre, **tuned)  # type: ignore
    dx = x.new_empty(0)
    if need_dx:
        tuned = _tuned_config(_key(dpre, w1, "", True))
        dx = _wk.gemm(dpre, w1, b_trans=True, **tuned)  # type: ignore
    dw1 = dpre.T @ x if need_dw1 else w1.new_empty(0)  # type: ignore
    dw2 = dout.T @ torch.nn.functional.gelu(pre) if need_dw2 else w2.new_empty(0)
    return dx, dw1, dw2  # type: ignore


@_mlp_bwd.register_fake
def _(dout, x, w1, w2, pre, need_dx, need_dw1, need_dw2):
    return (
        x.new_empty(x.shape if need_dx else (0,)),
        w1.new_empty(w1.shape if need_dw1 else (0,)),
        w2.new_empty(w2.shape if need_dw2 else (0,)),
    )


def _mlp_setup(ctx, inputs, output):
    ctx.mark_non_differentiable(output[1])
    ctx.save_for_backward(*inputs, output[1])


def _mlp_backward(ctx, dout, _):
    x, w1, w2, pre = ctx.saved_tensors
    need = ctx.needs_input_grad
    grads = _mlp_bwd(dout, x, w1, w2, pre, *need)
    return tuple(g if n else None for g, n in zip(grads, need, strict=True))


_mlp_train.register_autograd(_mlp_backward, setup_context=_mlp_setup)


def fits_mlp_grad(x: torch.Tensor, w_up: torch.Tensor, w_down: torch.Tensor) -> bool:
    """Tell whether ``mlp_grad`` can run these operands.

    Args:
        x: Activations, shape ``[..., K]``.
        w_up: First weight, shape ``[H, K]``.
        w_down: Second weight, shape ``[N, H]``.

    Returns:
        True if ``fits`` accepts them in bf16 or fp16 and the activations have a multiple of
        128 rows.
    """
    rows = x.numel() // x.shape[-1]
    return x.dtype in _DTYPES and fits(x, w_up, w_down) and rows % _BM == 0


def mlp_grad(x: torch.Tensor, w_up: torch.Tensor, w_down: torch.Tensor) -> torch.Tensor:
    """Compute ``gelu(x @ w_up.T) @ w_down.T`` with the exact erf GELU. Differentiable.

    The GELU and its derivative run inside the matrix products, in both directions.

    Args:
        x: Activations accepted by ``fits_mlp_grad``, shape ``[..., K]``.
        w_up: Weight with the dtype of ``x``, shape ``[H, K]``.
        w_down: Weight with the dtype of ``x``, shape ``[N, H]``.

    Returns:
        Tensor with the dtype of ``x``, shape ``[..., N]``.
    """
    y, _ = _mlp_train(x.reshape(-1, x.shape[-1]), w_up, w_down)
    return y.view(*x.shape[:-1], w_down.shape[0])


def fits_accum(
    x: torch.Tensor,
    weight: torch.Tensor,
    transposed: bool = False,
    out: torch.Tensor | None = None,
    pre: torch.Tensor | None = None,
) -> bool:
    """Tell whether the accumulating products can run these operands.

    Args:
        x: Activations, shape ``[M, K]``, or ``[M, N]`` if ``transposed``.
        weight: Weight, shape ``[N, K]``.
        transposed: Whether the product is ``x @ weight`` instead of ``x @ weight.T``.
        out: The buffer the product is added to, if known.
        pre: The GELU input of ``dgelu_addmm_``, if known.

    Returns:
        True if the library is installed, the operands are contiguous bf16 or fp16
        of one dtype, ``M`` is a multiple of 128, and the product's output and inner widths are
        multiples of 128 and 64. ``out`` and ``pre``, when given, must also be contiguous with
        the dtype of ``x``.
    """
    n, k = weight.shape[::-1] if transposed else weight.shape
    return (
        available
        and x.dtype in _DTYPES
        and weight.dtype == x.dtype
        and weight.is_contiguous()
        and x.shape[0] % _BM == 0
        and n % 128 == 0
        and k % 64 == 0
        and all(
            t is None or (t.dtype == x.dtype and t.is_contiguous()) for t in (out, pre)
        )
    )


def addmm_(
    out: torch.Tensor, x: torch.Tensor, weight: torch.Tensor, transposed: bool = False
) -> torch.Tensor:
    """Add a matrix product to a buffer in place. Forward only.

    Args:
        out: Contiguous buffer with the dtype of ``x``, shape ``[M, N]``, or ``[M, K]`` if
            ``transposed``.
        x: Activations accepted by ``fits_accum``.
        weight: Weight, shape ``[N, K]``.
        transposed: Whether to add ``x @ weight`` instead of ``x @ weight.T``.

    Returns:
        ``out``.
    """
    x, weight = x.detach().contiguous(), weight.detach()
    tuned = _tuned_config(_key(x, weight, "", transposed))
    _wk.gemm(x, weight, out=out, b_trans=transposed, accum=True, **tuned)  # type: ignore
    return out


def gelu_addmm_(
    out: torch.Tensor, x: torch.Tensor, weight: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute ``gelu(out + x @ weight.T)`` with the exact erf GELU. Forward only.

    Args:
        out: Contiguous buffer holding the addend, shape ``[M, N]``. It is overwritten.
        x: Activations accepted by ``fits_accum``.
        weight: Weight, shape ``[N, K]``.

    Returns:
        ``(post, pre)``: the result, and the sum the GELU was applied to.
    """
    x, weight = x.detach().contiguous(), weight.detach()
    tuned = _tuned_config(_key(x, weight, "gelu"))
    return _wk.gemm(x, weight, out=out, act="gelu", store_pre=True, accum=True, **tuned)  # type: ignore


def dgelu_addmm_(
    out: torch.Tensor, g: torch.Tensor, weight: torch.Tensor, pre: torch.Tensor
) -> torch.Tensor:
    """Compute ``(out + g @ weight) * gelu'(pre)`` in place. Forward only.

    Args:
        out: Contiguous buffer holding the addend, shape ``[M, K]``.
        g: Gradient accepted by ``fits_accum`` with ``transposed``, shape ``[M, N]``.
        weight: Weight, shape ``[N, K]``.
        pre: Input the GELU was applied to, shape ``[M, K]``.

    Returns:
        ``out``.
    """
    g, weight, pre = g.detach().contiguous(), weight.detach(), pre.detach().contiguous()
    tuned = _tuned_config(_key(g, weight, "dgelu", True))
    _wk.gemm(  # type: ignore
        g, weight, out=out, b_trans=True, act="dgelu", pre=pre, accum=True, **tuned
    )
    return out


def fits_accum_fp8(
    q: torch.Tensor,
    w: FP8Weight,
    out: torch.Tensor,
    pre: torch.Tensor | None = None,
) -> bool:
    """Tell whether the accumulating FP8 products can run these operands.

    Args:
        q: FP8 tokens, shape ``[M, K]``.
        w: Quantized weight, shape ``[N, K]``.
        out: The bf16 or fp16 buffer the product is added to, shape ``[M, N]``.
        pre: The GELU input of ``fp8_dgelu_addmm_``, if any.

    Returns:
        True if ``fits(q, w.q)`` holds, ``M`` is a multiple of 128, and ``out`` and ``pre`` are
        contiguous with one dtype.
    """
    return (
        fits(q, w.q)
        and q.shape[0] % _BM == 0
        and out.dtype in _DTYPES
        and all(
            t is None or (t.dtype == out.dtype and t.is_contiguous())
            for t in (out, pre)
        )
    )


def _fp8_call(
    out: torch.Tensor, q: torch.Tensor, s: torch.Tensor, w: FP8Weight, act: str, **kw
):
    q, s = q.detach().contiguous(), s.detach().contiguous()
    scales = {"a_scale_mx": s} if s.dtype == torch.uint8 else {"scale_a": s.reshape(-1)}
    tuned = _tuned_config(_key(q, w.q, act))
    return _wk.gemm(  # type: ignore
        q,
        w.q.detach(),
        out=out,
        scale_b=w.s.detach(),
        act=act,
        accum=True,
        **scales,
        **kw,
        **tuned,
    )


def fp8_addmm_(
    out: torch.Tensor,
    q: torch.Tensor,
    s: torch.Tensor,
    w: FP8Weight,
    mx_out: bool = False,
):
    """Add the product of FP8 tokens and an FP8 weight to a buffer in place. Forward only.

    Args:
        out: Buffer accepted by ``fits_accum_fp8``, shape ``[M, N]``.
        q: FP8 tokens, shape ``[M, K]``.
        s: Float32 token scales ``[M, 1]``, or uint8 MXFP8 scales ``[M, K / 32]``.
        w: Quantized weight, shape ``[N, K]``.
        mx_out: Also return the sum as MXFP8.

    Returns:
        ``out``, or ``(out, q8, s8)`` with ``mx_out``: e4m3 values ``[M, N]`` and uint8 scales
        ``[M, N / 32]``.
    """
    return _fp8_call(out, q, s, w, "", out_mx=mx_out)


def fp8_gelu_addmm_(
    out: torch.Tensor,
    q: torch.Tensor,
    s: torch.Tensor,
    w: FP8Weight,
    mx_out: bool = False,
) -> tuple:
    """Compute ``gelu(out + product)`` for FP8 tokens and an FP8 weight. Forward only.

    Args:
        out: Buffer accepted by ``fits_accum_fp8`` holding the addend. It is overwritten.
        q: FP8 tokens, shape ``[M, K]``.
        s: Float32 token scales ``[M, 1]``, or uint8 MXFP8 scales ``[M, K / 32]``.
        w: Quantized weight, shape ``[N, K]``.
        mx_out: Also return the result as MXFP8.

    Returns:
        ``(post, pre)``: the result, and the sum the GELU was applied to, followed by
        ``(q8, s8)`` with ``mx_out``.
    """
    return _fp8_call(out, q, s, w, "gelu", store_pre=True, out_mx=mx_out)


def fp8_dgelu_addmm_(
    out: torch.Tensor,
    q: torch.Tensor,
    s: torch.Tensor,
    w: FP8Weight,
    pre: torch.Tensor,
    mx_out: bool = False,
):
    """Compute ``(out + product) * gelu'(pre)`` in place for FP8 tokens and weight. Forward only.

    Args:
        out: Buffer accepted by ``fits_accum_fp8`` holding the addend, shape ``[M, N]``.
        q: FP8 tokens, shape ``[M, K]``.
        s: Float32 token scales ``[M, 1]``, or uint8 MXFP8 scales ``[M, K / 32]``.
        w: Quantized weight, shape ``[N, K]``.
        pre: Input the GELU was applied to, shape ``[M, N]``.
        mx_out: Also return the result as MXFP8.

    Returns:
        ``out``, or ``(out, q8, s8)`` with ``mx_out``.
    """
    return _fp8_call(
        out, q, s, w, "dgelu", pre=pre.detach().contiguous(), out_mx=mx_out
    )


def fp8_gemm(q: torch.Tensor, w: FP8Weight) -> torch.Tensor:
    """Multiply FP8 tokens by an FP8 weight without applying either scale. Forward only.

    Args:
        q: FP8 tokens, shape ``[B, S, K]``.
        w: Quantized weight accepted by ``fits(q, w.q)``, shape ``[N, K]``.

    Returns:
        bfloat16 product of the stored values, shape ``[B, S, N]``. Multiply it by the token
        scales and by ``w.s`` to get the true product.
    """
    B, S, K = q.shape
    return _fp8_gemm(q.reshape(B * S, K), w.q).view(B, S, -1)


def fp8_linear(
    q: torch.Tensor, s: torch.Tensor, w: FP8Weight, fast_accum: bool = False
) -> torch.Tensor:
    """Multiply FP8 tokens by an FP8 weight. Forward only.

    Args:
        q: FP8 tokens, shape ``[B, S, K]``.
        s: Token scales, shape ``[B * S, 1]``.
        w: Quantized weight accepted by ``fits(q, w.q)``, shape ``[N, K]``.
        fast_accum: Ignored.

    Returns:
        bfloat16 result, shape ``[B, S, N]``.
    """
    B, S, K = q.shape
    return _fp8_linear(q.reshape(B * S, K), s, w.q, w.s).view(B, S, -1)


def warmup(
    rows: int,
    weight: torch.Tensor,
    act: str = "",
    tune: bool = True,
    verbose: bool = False,
) -> None:
    """Prepare one GEMM shape so that later calls with it start fast.

    The first tuning of a shape on a GPU takes a while. Its result is saved to ``TUNE_CACHE_PATH``
    and reused by later processes. Set the ``PNEUMAM_TUNE_CACHE`` environment variable to move
    that file.

    Args:
        rows: Rows of the activations the weight will be multiplied by: batch times tokens.
        weight: The ``[N, K]`` weight, on its final device and in its final dtype. Pass
            ``FP8Weight.q`` for an FP8 product.
        act: ``"gelu"`` to prepare ``linear_gelu``, empty for ``gemm`` and ``fp8_linear``.
        tune: Whether to tune the shape if no saved result exists for it.
        verbose: Whether to print progress while tuning.

    Raises:
        ValueError: If ``fits`` rejects the weight.
    """
    if not fits(weight, weight):
        raise ValueError(
            f"warpy has no kernel for a {tuple(weight.shape)} {weight.dtype} weight"
        )
    a = torch.zeros(
        rows,
        weight.shape[1] * weight.element_size(),
        dtype=torch.uint8,
        device=weight.device,
    ).view(weight.dtype)
    key = _key(a, weight, act)
    if tune and not _tuned_config(key):
        bulk = a[: rows // _BM * _BM] if rows >= _BM else _pad_rows(a)
        best, _ = _wk.autotune_gemm(bulk, weight, verbose=verbose, act=act)  # type: ignore
        _store_config(key, best)
    _run(a, weight, act)
    torch.cuda.synchronize(weight.device)


def warmup_backward(
    rows: int,
    weight: torch.Tensor | FP8Weight,
    dgelu: bool = False,
    tune: bool = True,
    verbose: bool = False,
) -> None:
    """Tune the product that carries a gradient back through a linear layer.

    Results are saved with those of ``warmup``.

    Args:
        rows: Rows of the gradient: batch times tokens. Rounded down to a multiple of 128.
        weight: The layer's ``[N, K]`` bf16 or fp16 weight, used transposed as in
            ``addmm_(..., transposed=True)``. For FP8, ``quantize_weight(weight.T)`` of shape
            ``[K, N]``, used as in ``fp8_addmm_``.
        dgelu: Whether a GELU precedes the layer, so the product also applies its derivative
            as in ``dgelu_addmm_`` and ``fp8_dgelu_addmm_``.
        tune: Whether to tune if no saved result exists.
        verbose: Whether to print progress while tuning.
    """
    fp8 = isinstance(weight, FP8Weight)
    b = weight.q if fp8 else weight
    m = max(rows // _BM, 1) * _BM
    n, k = (b.shape[1], b.shape[0]) if fp8 else b.shape
    dtype = torch.bfloat16 if fp8 else b.dtype
    a = torch.zeros(m, n, dtype=b.dtype, device=b.device)
    act = "dgelu" if dgelu else ""
    key = _key(a, b, act, not fp8)
    if not tune or _tuned_config(key):
        return
    out = torch.zeros(m, k, dtype=dtype, device=b.device)
    kwargs: dict = {"out": out, "accum": True}
    if dgelu:
        kwargs |= {"act": "dgelu", "pre": torch.zeros_like(out)}
    if fp8:
        ones = torch.ones(m, dtype=torch.float32, device=b.device)
        kwargs |= {"scale_a": ones, "scale_b": weight.s.detach()}  # type: ignore
    else:
        kwargs["b_trans"] = True
    best, _ = _wk.autotune_gemm(a, b.detach(), verbose=verbose, **kwargs)  # type: ignore
    _store_config(key, best)
    torch.cuda.synchronize(b.device)


def fits_attention(q: torch.Tensor) -> bool:
    """Tell whether this backend can run attention on inputs like ``q``.

    Args:
        q: Representative attention input, shape ``[B, S, H, hd]``.

    Returns:
        True if the library is installed and ``hd`` is 64 or 128.
    """
    return available and q.shape[-1] in (64, 128)


def _to_fp8(x: torch.Tensor, rescale: bool) -> tuple[torch.Tensor, torch.Tensor | None]:
    if not rescale:
        return x.to(FP8), None  # the cast saturates at +-448
    s = (x.abs().amax(dim=(1, 3)).float() / FP8_MAX).clamp_min(1e-12)
    return (x / s[:, None, :, None]).to(FP8), s


def _launch_attention(q8, k8, v8, qs, ks, vs, seqused_k=None, softmax_scale=None):
    out = torch.empty(q8.shape, dtype=torch.bfloat16, device=q8.device)
    # warpy wants heads before the sequence and accepts strided views
    args = (q8.transpose(1, 2), k8.transpose(1, 2), v8.transpose(1, 2), qs, ks, vs)
    _, lse, _ = _wk.attn_fwd(  # type: ignore
        *args,
        softmax_scale=softmax_scale,
        out=out.transpose(1, 2),
        return_lse=seqused_k is not None,
        seqused_k=seqused_k,
        lse_natural=True,
    )
    return out, lse


def _attend(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    rescale: bool,
    seqused_k: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    (q8, qs), (k8, ks), (v8, vs) = (_to_fp8(t, rescale) for t in (q, k, v))
    return _launch_attention(q8, k8, v8, qs, ks, vs, seqused_k)


def _attend_quantized(q, k, v, qk_scale, v_scale, seqused_k=None):
    vs = None
    if v_scale != 1.0:
        vs = torch.full((q.shape[0], q.shape[2]), 1.0 / v_scale, device=q.device)
    softmax_scale = 1.0 / (
        math.sqrt(q.shape[-1]) * qk_scale
    )  # undoes the Q and K scales
    return _launch_attention(q, k, v, None, None, vs, seqused_k, softmax_scale)


@torch.library.custom_op("pneumam::attention_quantized_warpy", mutates_args=())
def _attention_quantized(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, qk_scale: float, v_scale: float
) -> torch.Tensor:
    return _attend_quantized(q, k, v, qk_scale, v_scale)[0]


@_attention_quantized.register_fake
def _(q, k, v, qk_scale, v_scale):
    return torch.empty(q.shape, dtype=torch.bfloat16, device=q.device)


@torch.library.custom_op("pneumam::attention_warpy", mutates_args=())
def _attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, rescale: bool
) -> torch.Tensor:
    return _attend(q, k, v, rescale)[0]


@_attention.register_fake
def _(q, k, v, rescale):
    return q.new_empty(q.shape, dtype=torch.bfloat16)


@torch.library.custom_op("pneumam::attention_trimmed_warpy", mutates_args=())
def _attention_trimmed(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seqused_k: torch.Tensor,
    rescale: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    out, lse = _attend(q, k, v, rescale, seqused_k)
    return out, lse.contiguous()  # type: ignore


@_attention_trimmed.register_fake
def _(q, k, v, seqused_k, rescale):
    return q.new_empty(q.shape, dtype=torch.bfloat16), q.new_empty(
        (q.shape[0], q.shape[2], q.shape[1]), dtype=torch.float32
    )


@torch.library.custom_op("pneumam::attention_grad_warpy", mutates_args=())
def _attention_grad(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, q_scale: float, k_scale: float
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    q, k, v = q.detach(), k.detach(), v.detach()
    q8, k8, v8 = (t.to(FP8) for t in (q * q_scale, k * k_scale, v))
    scale = 1.0 / (math.sqrt(q.shape[-1]) * q_scale * k_scale)
    out = torch.empty(q.shape, dtype=torch.bfloat16, device=q.device)
    heads_first = (q8.transpose(1, 2), k8.transpose(1, 2), v8.transpose(1, 2))
    _, lse, pmax = _wk.attn_fwd(  # type: ignore
        *heads_first, softmax_scale=scale, out=out.transpose(1, 2), return_lse=True
    )
    return out, q8, k8, v8, lse.float().contiguous(), pmax.float().contiguous()


@_attention_grad.register_fake
def _(q, k, v, q_scale, k_scale):
    stats = (q.shape[0], q.shape[2], q.shape[1])
    return (
        q.new_empty(q.shape, dtype=torch.bfloat16),
        q.new_empty(q.shape, dtype=FP8),
        k.new_empty(k.shape, dtype=FP8),
        v.new_empty(v.shape, dtype=FP8),
        q.new_empty(stats, dtype=torch.float32),
        q.new_empty(stats, dtype=torch.float32),
    )


@torch.library.custom_op("pneumam::attention_grad_bwd_warpy", mutates_args=())
def _attention_grad_bwd(
    q8: torch.Tensor,
    k8: torch.Tensor,
    v8: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    pmax: torch.Tensor,
    g: torch.Tensor,
    q_scale: float,
    k_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scale = 1.0 / (math.sqrt(q8.shape[-1]) * q_scale * k_scale)
    # tensors saved by autograd still carry a graph, which the kernel cannot take
    operands = [t.detach().transpose(1, 2).contiguous() for t in (q8, k8, v8, out)]
    dout = g.detach().transpose(1, 2).to(torch.bfloat16).contiguous()
    stats = (lse.detach(), pmax.detach())
    dq, dk, dv = _wk.attn_bwd(*operands, *stats, dout, softmax_scale=scale)  # type: ignore
    # the kernel differentiates in the scaled units the FP8 operands hold
    dq, dk = dq * q_scale, dk * k_scale
    return tuple(t.transpose(1, 2).contiguous() for t in (dq, dk, dv))  # type: ignore


@_attention_grad_bwd.register_fake
def _(q8, k8, v8, out, lse, pmax, g, q_scale, k_scale):
    return tuple(t.new_empty(t.shape, dtype=torch.bfloat16) for t in (q8, k8, v8))


def _attention_grad_setup(ctx, inputs, output):
    ctx.save_for_backward(*output)
    ctx.scales, ctx.dtype = (inputs[3], inputs[4]), inputs[0].dtype


def _attention_grad_backward(ctx, g, *unused):
    out, q8, k8, v8, lse, pmax = ctx.saved_tensors
    grads = _attention_grad_bwd(q8, k8, v8, out, lse, pmax, g, *ctx.scales)
    return (*(t.to(ctx.dtype) for t in grads), None, None)


_attention_grad.register_autograd(
    _attention_grad_backward, setup_context=_attention_grad_setup
)


def attention_grad(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, q_scale: float, k_scale: float
) -> torch.Tensor:
    """Full, non-causal multi-head attention with FP8 forward and backward passes.

    Differentiable. Gradients treat quantization as the identity.

    Args:
        q: Queries accepted by ``fits_attention``, shape ``[B, Sq, H, hd]``. After scaling they
            must stay within 448 in magnitude.
        k: Keys, shape ``[B, Sk, H, hd]``, with the same bound after scaling.
        v: Values, shape ``[B, Sk, H, hd]``. They are quantized unscaled.
        q_scale: Factor the queries are multiplied by before quantizing.
        k_scale: Factor the keys are multiplied by before quantizing.

    Returns:
        Attention output in true units, dtype of ``q``, shape ``[B, Sq, H, hd]``.
    """
    return _attention_grad(q, k, v, q_scale, k_scale)[0].to(q.dtype)


def attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, rescale: bool = True
) -> torch.Tensor:
    """Full, non-causal multi-head attention computed in FP8. Forward only.

    Args:
        q: Queries accepted by ``fits_attention``, any floating dtype, shape ``[B, Sq, H, hd]``.
        k: Keys, shape ``[B, Sk, H, hd]``.
        v: Values, shape ``[B, Sk, H, hd]``.
        rescale: Whether to scale each head of each operand to the full FP8 range before
            quantizing. Turning it off is experimental. It is faster, but values above 448 in
            magnitude are clipped and values far below 1 lose precision.

    Returns:
        Attention output, bfloat16, shape ``[B, Sq, H, hd]``.
    """
    return _attention(q, k, v, rescale)


def attention_quantized(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_scale: float = 1.0,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
) -> torch.Tensor:
    """Full, non-causal multi-head attention on operands that are already FP8. Forward only.

    Args:
        q: FP8 queries accepted by ``fits_attention``, holding the true queries times
            ``q_scale``, shape ``[B, Sq, H, hd]``.
        k: FP8 keys times ``k_scale``, shape ``[B, Sk, H, hd]``.
        v: FP8 values times ``v_scale``, shape ``[B, Sk, H, hd]``.
        q_scale: Factor the queries were multiplied by before quantizing.
        k_scale: Factor the keys were multiplied by before quantizing.
        v_scale: Factor the values were multiplied by before quantizing.

    Returns:
        Attention output in true units, bfloat16, shape ``[B, Sq, H, hd]``.
    """
    return _attention_quantized(q, k, v, q_scale * k_scale, v_scale)


@torch.library.custom_op("pneumam::attention_trimmed_quantized_warpy", mutates_args=())
def _attention_trimmed_quantized(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seqused_k: torch.Tensor,
    qk_scale: float,
    v_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    out, lse = _attend_quantized(q, k, v, qk_scale, v_scale, seqused_k)
    return out, lse.contiguous()


@_attention_trimmed_quantized.register_fake
def _(q, k, v, seqused_k, qk_scale, v_scale):
    return torch.empty(q.shape, dtype=torch.bfloat16, device=q.device), torch.empty(
        (q.shape[0], q.shape[2], q.shape[1]), dtype=torch.float32, device=q.device
    )


def attention_trimmed_quantized(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seqused_k: torch.Tensor,
    q_scale: float = 1.0,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``attention_quantized`` over the first ``seqused_k[b]`` keys of each batch item.

    Forward only.

    Args:
        q: FP8 queries accepted by ``fits_attention``, holding the true queries times
            ``q_scale``, shape ``[B, Sq, H, hd]``.
        k: FP8 keys times ``k_scale``, shape ``[B, Sk, H, hd]``.
        v: FP8 values times ``v_scale``, shape ``[B, Sk, H, hd]``.
        seqused_k: Valid key count per batch item, int32, shape ``[B]``.
        q_scale: Factor the queries were multiplied by before quantizing.
        k_scale: Factor the keys were multiplied by before quantizing.
        v_scale: Factor the values were multiplied by before quantizing.

    Returns:
        ``(out, lse)``: the bfloat16 output in true units ``[B, Sq, H, hd]`` and the float32
        natural-log softmax normalizer of the true logits ``[B, H, Sq]``. See ``ctx_scale``.
    """
    return _attention_trimmed_quantized(q, k, v, seqused_k, q_scale * k_scale, v_scale)


def attention_trimmed(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seqused_k: torch.Tensor,
    rescale: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FP8 attention over the first ``seqused_k[b]`` keys of each batch item. Forward only.

    Args:
        q: Queries accepted by ``fits_attention``, any floating dtype, shape ``[B, Sq, H, hd]``.
        k: Keys, shape ``[B, Sk, H, hd]``.
        v: Values, shape ``[B, Sk, H, hd]``.
        seqused_k: Valid key count per batch item, int32, shape ``[B]``.
        rescale: As in ``attention``.

    Returns:
        ``(out, lse)``: the bfloat16 output ``[B, Sq, H, hd]`` and the float32 natural-log softmax
        normalizer ``[B, H, Sq]``. See ``ctx_scale``.
    """
    return _attention_trimmed(q, k, v, seqused_k, rescale)


def fits_attention_bf16(q: torch.Tensor) -> bool:
    """Tell whether the bf16 attention kernels can run on inputs like ``q``.

    Args:
        q: Representative attention input, shape ``[B, S, H, hd]``.

    Returns:
        True if ``fits_attention`` holds and ``q`` is bf16.
    """
    return fits_attention(q) and q.dtype == torch.bfloat16


def _heads_first(*tensors: torch.Tensor) -> list[torch.Tensor]:
    return [t.detach().transpose(1, 2) for t in tensors]


@torch.library.custom_op("pneumam::attention_bf16_warpy", mutates_args=())
def _attention_bf16(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, need_lse: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty_like(q)
    _, lse, _ = _wk.attn_fwd(  # type: ignore
        *_heads_first(q, k, v), out=out.transpose(1, 2), return_lse=need_lse
    )
    return out, lse.contiguous() if need_lse else q.new_empty(0, dtype=torch.float32)


@_attention_bf16.register_fake
def _(q, k, v, need_lse):
    stats = (q.shape[0], q.shape[2], q.shape[1]) if need_lse else (0,)
    return torch.empty_like(q), q.new_empty(stats, dtype=torch.float32)


@torch.library.custom_op("pneumam::attention_bf16_bwd_warpy", mutates_args=())
def _attention_bf16_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    g: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    operands = [t.contiguous() for t in _heads_first(q, k, v, out)]
    dout = g.detach().transpose(1, 2).to(torch.bfloat16).contiguous()
    lse = lse.detach()
    # bf16 ignores pmax but its shape is still checked
    dq, dk, dv = _wk.attn_bwd(*operands, lse, lse, dout)  # type: ignore
    return tuple(t.transpose(1, 2).contiguous() for t in (dq, dk, dv))  # type: ignore


@_attention_bf16_bwd.register_fake
def _(q, k, v, out, lse, g):
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


def _attention_bf16_setup(ctx, inputs, output):
    ctx.save_for_backward(*inputs[:3], *output)


def _attention_bf16_backward(ctx, g, *unused):
    q, k, v, out, lse = ctx.saved_tensors
    return (*_attention_bf16_bwd(q, k, v, out, lse, g), None)


_attention_bf16.register_autograd(
    _attention_bf16_backward, setup_context=_attention_bf16_setup
)


def attention_bf16(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Full, non-causal multi-head attention in bf16. Differentiable.

    Args:
        q: Queries accepted by ``fits_attention_bf16``, shape ``[B, Sq, H, hd]``.
        k: Keys, bf16, shape ``[B, Sk, H, hd]``.
        v: Values, bf16, shape ``[B, Sk, H, hd]``.

    Returns:
        Attention output, bfloat16, shape ``[B, Sq, H, hd]``.
    """
    need_lse = torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, v))
    return _attention_bf16(q, k, v, need_lse)[0]


@torch.library.custom_op("pneumam::attention_trimmed_bf16_warpy", mutates_args=())
def _attention_trimmed_bf16(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, seqused_k: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty_like(q)
    _, lse, _ = _wk.attn_fwd(  # type: ignore
        *_heads_first(q, k, v),
        out=out.transpose(1, 2),
        seqused_k=seqused_k,
        lse_natural=True,
    )
    return out, lse.contiguous()


@_attention_trimmed_bf16.register_fake
def _(q, k, v, seqused_k):
    stats = (q.shape[0], q.shape[2], q.shape[1])
    return torch.empty_like(q), q.new_empty(stats, dtype=torch.float32)


def attention_trimmed_bf16(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, seqused_k: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """bf16 attention over the first ``seqused_k[b]`` keys of each batch item. Forward only.

    Args:
        q: Queries accepted by ``fits_attention_bf16``, shape ``[B, Sq, H, hd]``.
        k: Keys, bf16, shape ``[B, Sk, H, hd]``.
        v: Values, bf16, shape ``[B, Sk, H, hd]``.
        seqused_k: Valid key count per batch item, int32, shape ``[B]``.

    Returns:
        ``(out, lse)``: the bfloat16 output ``[B, Sq, H, hd]`` and the float32 natural-log softmax
        normalizer ``[B, H, Sq]``. See ``ctx_scale``.
    """
    return _attention_trimmed_bf16(q, k, v, seqused_k)


def warmup_attention(
    batch: int,
    heads: int,
    q_len: int,
    k_len: int,
    head_dim: int,
    device: torch.device | str,
    trimmed: bool = False,
    tune: bool = True,
    verbose: bool = False,
    bf16: bool = False,
    backward: bool = False,
) -> None:
    """Prepare one attention shape so that later calls with it start fast.

    The first tuning of a shape on a GPU takes a while. Its result is saved and reused by later
    processes.

    Args:
        batch: Batch size.
        heads: Number of heads.
        q_len: Query sequence length.
        k_len: Key sequence length.
        head_dim: Head width, 64 or 128.
        device: CUDA device the attention will run on.
        trimmed: Whether to prepare ``attention_trimmed`` instead of ``attention``.
        tune: Whether to tune the shape if no saved result exists for it.
        verbose: Whether to print progress while tuning.
        bf16: Whether to prepare the bf16 kernels instead of the FP8 ones.
        backward: Whether to also tune the backward kernel, for training. Ignored with
            ``trimmed``, which has no backward.
    """
    dtype = torch.bfloat16 if bf16 else FP8
    q = torch.zeros(batch, q_len, heads, head_dim, dtype=dtype, device=device)
    k = torch.zeros(batch, k_len, heads, head_dim, dtype=dtype, device=device)
    arch = _wk.device_arch(q)  # type: ignore
    kind = "fwd_bf16" if bf16 else "fwd"
    key = _wk.utils.tune_key(kind, head_dim, False, q_len, k_len, 1, arch)  # type: ignore
    if tune and not _wk.utils.tune_lookup(key):  # type: ignore
        qh, kh = q.transpose(1, 2), k.transpose(1, 2)
        _wk.attn.autotune_attn_fwd(qh, kh, kh, verbose=verbose)  # type: ignore
    used = torch.full((batch,), k_len, dtype=torch.int32, device=device)
    if bf16:
        if trimmed:
            attention_trimmed_bf16(q, k, k, used)
        else:
            _attention_bf16(q, k, k, False)
    else:
        _attend(q, k, k, False, used if trimmed else None)
    if backward and tune and not trimmed:
        _tune_attention_backward(q, k, arch, verbose)
    torch.cuda.synchronize(device)


def _tune_attention_backward(q, k, arch, verbose):
    kind = "bwd_bf16" if q.dtype == torch.bfloat16 else "bwd"
    key = _wk.utils.tune_key(kind, q.shape[-1], False, q.shape[1], k.shape[1], 1, arch)  # type: ignore
    if _wk.utils.tune_lookup(key):  # type: ignore
        return
    qh, kh = q.transpose(1, 2).contiguous(), k.transpose(1, 2).contiguous()
    out, lse, pmax = _wk.attn_fwd(qh, kh, kh, return_lse=True)  # type: ignore
    dout = torch.zeros_like(out)
    _wk.attn.autotune_attn_bwd(qh, kh, kh, out, lse, pmax, dout, verbose=verbose)  # type: ignore
