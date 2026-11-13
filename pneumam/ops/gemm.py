import torch
import torch.nn.functional as F

from ..backends import quack, reference, triton, warpy
from ..backends._fp8 import FP8Weight, quantize_weight
from . import _dispatch

__all__ = [
    "FP8Weight",
    "fp8_linear",
    "fp8_linear_unscaled",
    "fp8_supported",
    "frozen_linear_fp8",
    "linear",
    "linear_gelu",
    "mlp",
    "quantize_weight",
    "rowquant",
]

_DATACENTER = ("sm90", "sm100")


def _needs_grad(*tensors: torch.Tensor) -> bool:
    return torch.is_grad_enabled() and any(t.requires_grad for t in tensors)


def linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Compute ``x @ weight.T``. Differentiable.

    Args:
        x: Activations, shape ``[..., K]``.
        weight: Weight with the dtype of ``x``, shape ``[N, K]``.

    Returns:
        Tensor with the dtype of ``x``, shape ``[..., N]``.

    Raises:
        RuntimeError: If ``x`` is not on a CUDA device.
    """
    if _dispatch.arch(x.device) == "sm120" and warpy.fits(x, weight):
        return warpy.gemm(x, weight)
    return F.linear(x, weight)


def linear_gelu(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Compute ``gelu(x @ weight.T)`` with the exact erf GELU and no bias. Differentiable.

    Args:
        x: Activations, shape ``[..., K]``.
        weight: Weight with the dtype of ``x``, shape ``[N, K]``.

    Returns:
        Tensor with the dtype of ``x``, shape ``[..., N]``.

    Raises:
        RuntimeError: If ``x`` is not on a CUDA device.
    """
    arch = _dispatch.arch(x.device)
    if arch == "sm120" and warpy.fits(x, weight) and not _needs_grad(x, weight):
        return warpy.linear_gelu(x, weight)
    if arch in _DATACENTER and quack.fits(x, weight):
        return quack.linear_gelu(x, weight)
    return reference.linear_gelu(x, weight)


def mlp(x: torch.Tensor, w_up: torch.Tensor, w_down: torch.Tensor) -> torch.Tensor:
    """Compute ``gelu(x @ w_up.T) @ w_down.T`` with the exact erf GELU and no biases.

    Differentiable.

    Args:
        x: Activations, shape ``[..., K]``.
        w_up: Weight with the dtype of ``x``, shape ``[H, K]``.
        w_down: Weight with the dtype of ``x``, shape ``[N, H]``.

    Returns:
        Tensor with the dtype of ``x``, shape ``[..., N]``.

    Raises:
        RuntimeError: If ``x`` is not on a CUDA device.
    """
    if _dispatch.arch(x.device) in _DATACENTER and quack.fits(x, w_up, w_down):
        return quack.mlp(x, w_up, w_down)
    return linear(linear_gelu(x, w_up), w_down)


def fp8_supported(device: torch.device) -> bool:
    """Tell whether FP8 products run as real FP8 kernels on a device.

    Args:
        device: Device to check.

    Returns:
        True if ``fp8_linear`` has an FP8 kernel there, or if ops are pinned to the reference
        path. If False it still works, but slowly.

    Raises:
        RuntimeError: If ``device`` is not a CUDA device.
    """
    arch = _dispatch.arch(device)
    return (
        arch == "reference"
        or (arch == "sm120" and warpy.available)
        or (arch in _DATACENTER and triton.available)
    )


def fp8_linear(
    q: torch.Tensor, s: torch.Tensor, w: FP8Weight, fast_accum: bool = False
) -> torch.Tensor:
    """Multiply FP8 tokens by an FP8 weight. Forward only.

    Args:
        q: FP8 tokens, shape ``[B, S, K]``.
        s: Token scales, shape ``[B * S, 1]``. See ``rowquant``.
        w: Weight from ``quantize_weight``, shape ``[N, K]``.
        fast_accum: Accumulate with reduced precision for extra speed, where the GPU offers it.

    Returns:
        bfloat16 result, shape ``[B, S, N]``.

    Raises:
        RuntimeError: If ``q`` is not on a CUDA device.
    """
    y, scales = fp8_linear_unscaled(q, s, w, fast_accum)
    if scales is None:
        return y
    B, S, N = y.shape
    if triton.available and _dispatch.arch(q.device) != "reference":
        return triton.scale_rows_cols_(y.view(B * S, N), *scales).view(B, S, N)
    return (y.float() * scales[0].view(B, S, 1) * scales[1]).to(y.dtype)


def fp8_linear_unscaled(
    q: torch.Tensor, s: torch.Tensor, w: FP8Weight, fast_accum: bool = False
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
    """``fp8_linear`` that may leave its scales for the next op to apply. Forward only.

    Pass the scales on to ``rowquant`` or ``residual_adaln_fp8``, which apply them for free.

    Args:
        q: FP8 tokens, shape ``[B, S, K]``.
        s: Token scales, shape ``[B * S, 1]``. See ``rowquant``.
        w: Weight from ``quantize_weight``, shape ``[N, K]``.
        fast_accum: Accumulate with reduced precision for extra speed, where the GPU offers it.

    Returns:
        ``(y, scales)`` with ``y`` bfloat16 of shape ``[B, S, N]``. If ``scales`` is None, ``y``
        is the product. Otherwise ``scales`` is float32 ``(rows, cols)`` of shapes
        ``[B * S, 1]`` and ``[N]``, and the product is ``y * rows * cols``.

    Raises:
        RuntimeError: If ``q`` is not on a CUDA device.
    """
    arch = _dispatch.arch(q.device)
    if arch == "sm120" and warpy.fits(q, w.q):
        return warpy.fp8_gemm(q, w), (s, w.s)
    if arch in _DATACENTER and triton.available:
        return triton.fp8_linear(q, s, w, fast_accum), None
    if arch == "reference":
        return reference.fp8_gemm(q, w), (s, w.s)
    return reference.fp8_linear(q, s, w, fast_accum), None


def rowquant(
    x: torch.Tensor,
    gelu: bool = False,
    scales: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize tokens to FP8 with a scale per token, optionally applying GELU first.

    Forward only.

    Args:
        x: Tokens, shape ``[B, S, K]``.
        gelu: Apply GELU before quantizing. It is within 1e-6 of the exact erf GELU.
        scales: Scales from ``fp8_linear_unscaled`` that ``x`` still has to be multiplied by.

    Returns:
        ``(q, s)``: the FP8 tokens ``[B, S, K]`` and their float32 scales ``[B * S, 1]``.
        See ``fp8_linear``.

    Raises:
        RuntimeError: If ``x`` is not on a CUDA device.
    """
    if triton.available and _dispatch.arch(x.device) != "reference":
        return triton.rowquant(x, gelu, scales)
    return reference.rowquant(x, gelu, scales)


@torch.library.custom_op("pneumam::frozen_linear_fp8", mutates_args=())
def _frozen_fp8(
    x: torch.Tensor,
    weight: torch.Tensor,
    wq: torch.Tensor,
    ws: torch.Tensor,
    tq: torch.Tensor | None,
    ts: torch.Tensor | None,
) -> torch.Tensor:
    q, s = rowquant(x[None])
    return fp8_linear(q, s, FP8Weight(wq, ws))[0].to(x.dtype)


@_frozen_fp8.register_fake
def _(x, weight, wq, ws, tq, ts):
    return x.new_empty(x.shape[0], weight.shape[0])


def _frozen_fp8_setup(ctx, inputs, output):
    _, weight, _, _, tq, ts = inputs
    ctx.fp8 = tq is not None and ts is not None
    ctx.save_for_backward(*((tq, ts) if ctx.fp8 else (weight,)))


def _frozen_fp8_backward(ctx, g):
    if not ctx.fp8:
        (weight,) = ctx.saved_tensors
        return g @ weight, None, None, None, None, None
    tq, ts = ctx.saved_tensors
    gq, gs = rowquant(g[None])
    dx = fp8_linear(gq, gs, FP8Weight(tq, ts))[0]
    return dx.to(g.dtype), None, None, None, None, None


_frozen_fp8.register_autograd(_frozen_fp8_backward, setup_context=_frozen_fp8_setup)


def frozen_linear_fp8(
    x: torch.Tensor,
    weight: torch.Tensor,
    w8: FP8Weight,
    w8_t: FP8Weight | None = None,
) -> torch.Tensor:
    """Compute ``x @ weight.T`` in FP8 for a weight that is not trained.

    Differentiable in ``x`` only. The gradient is that of the exact product, so quantization is
    treated as the identity.

    Args:
        x: Activations, shape ``[..., K]``.
        weight: The weight, shape ``[N, K]``. Used for the backward pass unless ``w8_t`` is
            given.
        w8: ``quantize_weight(weight)``.
        w8_t: ``quantize_weight(weight.T)``. If given, the backward pass also runs in FP8,
            which is faster and adds quantization noise to the gradient.

    Returns:
        Tensor with the dtype of ``x``, shape ``[..., N]``.

    Raises:
        RuntimeError: If ``x`` is not on a CUDA device.
    """
    flat = x.reshape(-1, x.shape[-1])
    tq, ts = (None, None) if w8_t is None else w8_t
    y = _frozen_fp8(flat, weight, w8.q, w8.s, tq, ts)
    return y.view(*x.shape[:-1], weight.shape[0])
