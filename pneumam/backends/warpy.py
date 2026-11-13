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


def _key(a: torch.Tensor, w: torch.Tensor, act: str) -> str:
    gpu = torch.cuda.get_device_name(a.device)
    m = max(a.shape[0] // _BM, 1) * _BM  # the rows of the main launch
    return f"{gpu}|{a.dtype}|m={m}|n={w.shape[0]}|k={w.shape[1]}|act={act}"


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


def _launch(a, w, act, out=None) -> torch.Tensor:
    y = _wk.gemm(a, w, out=out, act=act, **_tuned_config(_key(a, w, act)))  # type: ignore
    return y[0] if act == "gelu" else y  # type: ignore


def _run(a: torch.Tensor, w: torch.Tensor, act: str = "") -> torch.Tensor:
    m = a.shape[0]
    a = a.contiguous()
    whole = m // _BM * _BM
    if whole == m:
        return _launch(a, w, act)
    dtype = torch.bfloat16 if a.dtype == FP8 else a.dtype
    y = torch.empty(m, w.shape[0], dtype=dtype, device=a.device)
    if whole:
        _launch(a[:whole], w, act, y[:whole])  # the bulk runs in place, uncopied
    y[whole:] = _launch(_pad_rows(a[whole:]), w, act)[: m - whole]
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
    B, S, _ = q.shape
    y = fp8_gemm(q, w).view(B * S, -1)
    return y.mul_(s).mul_(w.s.view(1, -1)).view(B, S, -1)


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
    """
    q = torch.zeros(batch, q_len, heads, head_dim, dtype=torch.uint8, device=device)
    k = torch.zeros(batch, k_len, heads, head_dim, dtype=torch.uint8, device=device)
    key = _wk.utils.tune_key("fwd", head_dim, False, q_len, k_len, 1)  # type: ignore
    if tune and not _wk.utils.tune_lookup(key):  # type: ignore
        q8, k8 = q.view(FP8).transpose(1, 2), k.view(FP8).transpose(1, 2)
        _wk.attn.autotune_attn_fwd(q8, k8, k8, verbose=verbose)  # type: ignore
    used = torch.full((batch,), k_len, dtype=torch.int32, device=device)
    _attend(q.view(FP8), k.view(FP8), k.view(FP8), False, used if trimmed else None)
    torch.cuda.synchronize(device)
