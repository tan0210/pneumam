import math

import torch
import torch.nn.functional as F

try:
    from quack.gemm_interface import gemm_act as _gemm_act
    from quack.gemm_interface import gemm_dact as _gemm_dact

    available = True
except Exception:  # noqa: BLE001  # cutlass can fail to import in ways other than ImportError
    _gemm_act = None
    _gemm_dact = None
    available = False

_DTYPES = (torch.bfloat16, torch.float16)


def fits(x: torch.Tensor, *weights: torch.Tensor) -> bool:
    """Tell whether this backend can run these operands.

    Args:
        x: Activations.
        *weights: The weights ``x`` is multiplied by.

    Returns:
        True if the library is installed and everything is bf16 or fp16 with one shared dtype.
    """
    return available and x.dtype in _DTYPES and all(w.dtype == x.dtype for w in weights)


@torch.library.custom_op("pneumam::linear_gelu_quack", mutates_args=())
def _linear_gelu_quack(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    _, post = _gemm_act(  # type: ignore
        x,
        weight.T,
        activation="gelu_erf",  # type: ignore
        store_preact=False,
    )
    return post


@_linear_gelu_quack.register_fake
def _(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return x.new_empty(x.shape[0], weight.shape[0])


@torch.library.custom_op("pneumam::linear_gelu_quack_train", mutates_args=())
def _linear_gelu_quack_train(
    x: torch.Tensor, weight: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    pre, post = _gemm_act(  # type: ignore
        x,
        weight.T,
        activation="gelu_erf",  # type: ignore
        store_preact=True,
    )
    return post, pre  # type: ignore


@_linear_gelu_quack_train.register_fake
def _(x: torch.Tensor, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n = weight.shape[0]
    return x.new_empty(x.shape[0], n), x.new_empty(x.shape[0], n)


def _setup_context(ctx, inputs, output) -> None:
    x, weight = inputs
    _, pre = output
    ctx.mark_non_differentiable(pre)
    ctx.save_for_backward(x, weight, pre)


def _backward(ctx, grad_post: torch.Tensor, grad_pre: torch.Tensor | None):
    x, weight, pre = ctx.saved_tensors
    # d/dh gelu(h) = Phi(h) + h * phi(h)
    h = pre.float()
    cdf = 0.5 * (1.0 + torch.erf(h * (1.0 / math.sqrt(2.0))))
    pdf = torch.exp(-0.5 * h * h) * (1.0 / math.sqrt(2.0 * math.pi))
    dh = (grad_post.float() * (cdf + h * pdf)).to(x.dtype)
    dx = dh @ weight if ctx.needs_input_grad[0] else None
    dw = dh.T @ x if ctx.needs_input_grad[1] else None
    return dx, dw


_linear_gelu_quack_train.register_autograd(_backward, setup_context=_setup_context)


def linear_gelu(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Compute ``gelu(x @ weight.T)`` with the exact erf GELU and no bias. Differentiable.

    Args:
        x: Activations accepted by ``fits``, shape ``[..., K]``.
        weight: Weight with the dtype of ``x``, shape ``[N, K]``.

    Returns:
        Tensor with the dtype of ``x``, shape ``[..., N]``.
    """
    x2 = x.reshape(-1, x.shape[-1])
    if torch.is_grad_enabled() and (x.requires_grad or weight.requires_grad):
        y, _ = _linear_gelu_quack_train(x2, weight)
    else:
        y = _linear_gelu_quack(x2, weight)
    return y.view(*x.shape[:-1], weight.shape[0])


@torch.library.custom_op("pneumam::mlp_quack_train", mutates_args=())
def _mlp_quack_train(
    x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    pre, post = _gemm_act(  # type: ignore
        x,
        w1.T,
        activation="gelu_erf",  # type: ignore
        store_preact=True,
    )
    return F.linear(post, w2), pre  # type: ignore


@_mlp_quack_train.register_fake
def _(x, w1, w2):
    return x.new_empty(x.shape[0], w2.shape[0]), x.new_empty(x.shape[0], w1.shape[0])


@torch.library.custom_op("pneumam::mlp_quack_bwd", mutates_args=())
def _mlp_quack_bwd(
    dout: torch.Tensor,
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    pre: torch.Tensor,
    need_dx: bool,
    need_dw1: bool,
    need_dw2: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dout = dout.contiguous()
    dpre, post = _gemm_dact(dout, w2, pre, activation="gelu_erf")  # type: ignore
    dx = dpre @ w1 if need_dx else x.new_empty(0)
    dw1 = dpre.T @ x if need_dw1 else w1.new_empty(0)
    dw2 = dout.T @ post if need_dw2 else w2.new_empty(0)
    return dx, dw1, dw2


@_mlp_quack_bwd.register_fake
def _(dout, x, w1, w2, pre, need_dx, need_dw1, need_dw2):
    return (
        x.new_empty(x.shape if need_dx else (0,)),
        w1.new_empty(w1.shape if need_dw1 else (0,)),
        w2.new_empty(w2.shape if need_dw2 else (0,)),
    )


def _mlp_setup_context(ctx, inputs, output) -> None:
    x, w1, w2 = inputs
    _, pre = output
    ctx.mark_non_differentiable(pre)
    ctx.save_for_backward(x, w1, w2, pre)


def _mlp_backward(ctx, dout: torch.Tensor, dpre: torch.Tensor | None):
    x, w1, w2, pre = ctx.saved_tensors
    need_x, need_w1, need_w2 = ctx.needs_input_grad
    dx, dw1, dw2 = _mlp_quack_bwd(dout, x, w1, w2, pre, need_x, need_w1, need_w2)
    return (dx if need_x else None, dw1 if need_w1 else None, dw2 if need_w2 else None)


_mlp_quack_train.register_autograd(_mlp_backward, setup_context=_mlp_setup_context)


def mlp(x: torch.Tensor, w_up: torch.Tensor, w_down: torch.Tensor) -> torch.Tensor:
    """Compute ``gelu(x @ w_up.T) @ w_down.T`` with the exact erf GELU and no biases.

    Differentiable.

    Args:
        x: Activations accepted by ``fits``, shape ``[..., K]``.
        w_up: Weight with the dtype of ``x``, shape ``[H, K]``.
        w_down: Weight with the dtype of ``x``, shape ``[N, H]``.

    Returns:
        Tensor with the dtype of ``x``, shape ``[..., N]``.
    """
    if torch.is_grad_enabled() and (
        x.requires_grad or w_up.requires_grad or w_down.requires_grad
    ):
        y, _ = _mlp_quack_train(x.reshape(-1, x.shape[-1]), w_up, w_down)
        return y.view(*x.shape[:-1], w_down.shape[0])
    return F.linear(linear_gelu(x, w_up), w_down)
