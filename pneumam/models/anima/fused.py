from typing import Any, cast

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ...adapters.lllite import LLLiteLinear
from ...ops import attention as A
from ...ops import gemm
from ...ops.adaln import adaln, adaln_fp8, residual_adaln, residual_adaln_fp8
from ...ops.gemm import (
    FP8Weight,
    fp8_linear,
    fp8_linear_unscaled,
    fp8_supported,
    linear,
    linear_gelu,
    mlp,
    quantize_weight,
    rowquant,
)
from ...ops.norm import head_norm, head_norm_fp8
from ...ops.rope import norm_scale, qkv_norm_rope, qkv_norm_rope_fp8
from ...runtime import Runtime
from ...utils import is_experimental_fp8_attention
from ..spec import Prepared
from .dit import FP8_LAYERS, AnimaDiT, Context


def _plain(layer: nn.Module) -> bool:
    # LLLite over a plain projection has an FP8 path of its own
    if isinstance(layer, LLLiteLinear):
        layer = layer.base
    return type(layer) is nn.Linear


def _adapted(model: AnimaDiT) -> bool:
    return not all(_plain(getattr(b, n)) for b in model.blocks for n in FP8_LAYERS)


def _fp8_proj(
    layer: Any, q: torch.Tensor, s: torch.Tensor, w: FP8Weight, fast: bool
) -> torch.Tensor:
    if isinstance(layer, LLLiteLinear):
        return layer.fast_fp8(q, s, w)
    return fp8_linear(q, s, w, fast)


def _attention_scales(model: AnimaDiT) -> list[dict[str, float]]:
    return [
        {n: norm_scale(getattr(b, f"{n}_norm")) for n in ("q", "k", "xq", "xk")}
        for b in model.blocks
    ]


def _proj(layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if type(layer) is nn.Linear and layer.bias is None:
        return linear(x, layer.weight)
    return layer(x)  # an adapter


def _mlp(b: nn.Module, hid: torch.Tensor) -> torch.Tensor:
    up, down = b.up, b.down
    if type(up) is nn.Linear and type(down) is nn.Linear:
        return mlp(hid, up.weight, down.weight)
    both = getattr(up, "fused_mlp", None)
    if both is not None:
        y = both(down, hid)
        if y is not None:
            return y
    if type(up) is nn.Linear:
        return down(linear_gelu(hid, up.weight))  # type: ignore
    return down(nn.functional.gelu(up(hid)))  # type: ignore


def _block(
    model: AnimaDiT,
    l: int,
    hid: torch.Tensor,
    z: torch.Tensor,
    mod: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    ctx: Context,
    scales: dict[str, float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    c, b = model.cfg, cast(Any, model.blocks[l])
    qkv = _proj(b.qkv, hid)
    q, k, v = qkv_norm_rope(qkv, b.q_norm, b.k_norm, cos, sin, c.heads, c.qk_eps)
    if scales is None:
        att = A.attention(q, k, v)
    else:
        att = A.attention_fp8_grad(q, k, v, scales["q"], scales["k"])
    hid, z = residual_adaln(
        _proj(b.o1, att),  # type: ignore
        z,
        mod[l, 0, 2],
        mod[l, 1, 0],
        mod[l, 1, 1],
    )
    q = head_norm(_proj(b.xq, hid), b.xq_norm, c.heads, c.qk_eps)  # type: ignore
    if scales is None:
        att = ctx.attend(l, q)
    else:
        att = A.attention_fp8_grad(q, *ctx.kv[l], scales["xq"], scales["xk"])
    hid, z = residual_adaln(
        _proj(b.o2, att),  # type: ignore
        z,
        mod[l, 1, 2],
        mod[l, 2, 0],
        mod[l, 2, 1],
    )
    y = _mlp(b, hid)
    if l + 1 < c.layers:
        return residual_adaln(y, z, mod[l, 2, 2], mod[l + 1, 0, 0], mod[l + 1, 0, 1])
    return y, z


def forward_fused(
    model: AnimaDiT,
    rt: Runtime,
    x: torch.Tensor,
    t: torch.Tensor,
    ctx: Context,
    mods: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Predict the velocity in the model's own precision. Differentiable.

    With autograd enabled and ``rt.checkpointing`` set, each block is recomputed during backward
    and only its input is stored. Adapters attached to the projections are applied.

    Args:
        model: The model.
        rt: Its runtime.
        x: Noisy latent, shape ``[B, H, W, C]``.
        t: Timesteps (sigmas), shape ``[B]``. Ignored if ``mods`` is given.
        ctx: Context from ``encode_context``, with a batch size matching ``x``.
        mods: Optional precomputed ``(mod, fin)`` from ``modulation``, with batch size ``B``.
            Broadcast views are fine.

    Returns:
        Predicted velocity, shape ``[B, H, W, C]``.
    """
    _, H, W, _ = x.shape
    cos, sin = model.tables(rt, H // 2, W // 2, x.device)
    z = model.patchify(x)
    mod, fin = mods if mods is not None else model.modulation(t, z.dtype)
    mod = mod.permute(1, 2, 3, 0, 4).contiguous()
    hid = adaln(z, mod[0, 0, 0], mod[0, 0, 1])
    remat = rt.checkpointing and torch.is_grad_enabled()
    capturing = x.is_cuda and torch.cuda.is_current_stream_capturing()
    scales = None
    if rt.fp8_attention_grad and torch.is_grad_enabled():
        if rt.fp8_scales is None:
            rt.fp8_scales = _attention_scales(model)
        scales = rt.fp8_scales
    for l in range(model.cfg.layers):
        args = (
            model,
            l,
            hid,
            z,
            mod,
            cos,
            sin,
            ctx,
            None if scales is None else scales[l],
        )
        if remat:
            hid, z = checkpoint(  # type: ignore
                _block,
                *args,
                use_reentrant=False,
                preserve_rng_state=not capturing,
            )
        else:
            hid, z = _block(*args)
    shift, scale = fin.chunk(2, dim=-1)
    hid, _ = residual_adaln(hid, z, mod[-1, 2, 2], shift, scale + 1)
    return model.unpatchify(model.out_proj(hid), H, W)


def enable_fp8(model: AnimaDiT, rt: Runtime, fast_accum: bool = False) -> None:
    """Switch inference to FP8 attention and MLP projections.

    This quantizes the current weights, so call it after merging any adapter and after the model
    is on its final device, and again after changing the weights. Training keeps using the
    original weights. Captured graphs are dropped.

    Args:
        model: The model.
        rt: Its runtime.
        fast_accum: Accumulate with reduced precision for extra speed, where the GPU offers it.

    Raises:
        RuntimeError: If the GPU has no FP8 matrix multiplication.
    """
    if not fp8_supported(model.patch.weight.device):
        raise RuntimeError("this GPU has no FP8 matrix multiplication")
    rt.fp8 = [
        {n: quantize_weight(getattr(b, n).weight) for n in FP8_LAYERS}
        for b in model.blocks
    ]
    rt.fp8_scales = _attention_scales(model)
    rt.fp8_fast_accum = fast_accum
    rt.graphs.clear()


def warmup(
    model: AnimaDiT,
    rt: Runtime,
    height: int,
    width: int,
    batch: int,
    context_length: int | None = None,
    tune: bool = True,
    verbose: bool = False,
) -> int:
    """Prepare the GPU kernels for one latent size and batch size, tuning them where possible.

    Tuned kernels are faster. Tuning a size for the first time on a GPU takes a while, and the
    results are saved for later processes. Call it after ``enable_fp8`` to cover FP8 as well.

    Args:
        model: The model, on its final device.
        rt: Its runtime.
        height: Latent height.
        width: Latent width.
        batch: Latents per model evaluation. Guidance doubles it.
        context_length: Padded text length, to also prepare FP8 cross-attention.
        tune: Whether to tune kernels that have no saved result yet.
        verbose: Whether to print progress while tuning.

    Returns:
        The number of kernels prepared. 0 means this GPU needs no preparation.
    """
    c, b = model.cfg, model.blocks[0]
    device = model.patch.weight.device
    tokens = (height // 2) * (width // 2)
    rows, done = batch * tokens, 0
    for name in FP8_LAYERS:
        act = "gelu" if name == "up" else ""
        done += gemm.warmup(rows, getattr(b, name).weight, act, tune, verbose)
        if rt.fp8 is not None:
            done += gemm.warmup(rows, rt.fp8[0][name].q, "", tune, verbose)
    shape = (batch, c.heads, tokens)
    done += A.warmup(*shape, tokens, c.head_dim, device, tune, verbose)
    if context_length is not None:
        done += A.warmup(*shape, context_length, c.head_dim, device, tune, verbose)
    return done


def disable_fp8(rt: Runtime) -> None:
    """Go back to the model's own precision and free the quantized weights.

    Args:
        rt: The runtime. Its captured graphs are dropped.
    """
    rt.fp8 = rt.fp8_scales = None
    rt.graphs.clear()


@torch.no_grad()
def forward_fp8(
    model: AnimaDiT,
    rt: Runtime,
    x: torch.Tensor,
    t: torch.Tensor,
    ctx: Context,
    mods: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Predict the velocity with FP8 projections. Forward only.

    Attention itself also runs in FP8 if the experimental FP8 attention path is switched on and
    ``ctx`` was encoded with it on. See ``is_experimental_fp8_attention``.

    Args:
        model: The model.
        rt: Its runtime, after ``enable_fp8``.
        x: Noisy latent, shape ``[B, H, W, C]``.
        t: Timesteps (sigmas), shape ``[B]``. Ignored if ``mods`` is given.
        ctx: Context from ``encode_context``, with a batch size matching ``x``.
        mods: Optional precomputed ``(mod, fin)`` from ``modulation``, with batch size ``B``.

    Returns:
        Predicted velocity, shape ``[B, H, W, C]``.

    Raises:
        RuntimeError: If ``enable_fp8`` has not been called.
    """
    if rt.fp8 is None or rt.fp8_scales is None:
        raise RuntimeError("call `enable_fp8()` first")
    c, fast = model.cfg, rt.fp8_fast_accum
    quantized = is_experimental_fp8_attention() and ctx.kv8 is not None
    _, H, W, _ = x.shape
    cos, sin = model.tables(rt, H // 2, W // 2, x.device)
    z = model.patchify(x)
    mod, fin = mods if mods is not None else model.modulation(t, z.dtype)
    mod = mod.permute(1, 2, 3, 0, 4).contiguous()
    hq, hs = adaln_fp8(z, mod[0, 0, 0], mod[0, 0, 1])
    y = z
    for l, b in enumerate(model.blocks):
        w, s = rt.fp8[l], rt.fp8_scales[l]
        qkv = _fp8_proj(b.qkv, hq, hs, w["qkv"], fast)
        rope = (b.q_norm, b.k_norm, cos, sin, c.heads)
        if quantized:
            q, k, v = qkv_norm_rope_fp8(qkv, *rope, s["q"], s["k"], 1.0, c.qk_eps)  # type: ignore
            att = A.attention_quantized(q, k, v, s["q"], s["k"])
        else:
            att = A.attention(*qkv_norm_rope(qkv, *rope, c.qk_eps))  # type: ignore
        o, sc = fp8_linear_unscaled(*rowquant(att), w["o1"], fast)
        hq, hs, z = residual_adaln_fp8(
            o, z, mod[l, 0, 2], mod[l, 1, 0], mod[l, 1, 1], x_scales=sc
        )
        xq = _fp8_proj(b.xq, hq, hs, w["xq"], fast)
        if quantized:
            q = head_norm_fp8(xq, b.xq_norm, c.heads, s["xq"], c.qk_eps)  # type: ignore
            att = ctx.attend_quantized(l, q, s["xq"])
        else:
            att = ctx.attend(l, head_norm(xq, b.xq_norm, c.heads, c.qk_eps))  # type: ignore
        o, sc = fp8_linear_unscaled(*rowquant(att), w["o2"], fast)
        hq, hs, z = residual_adaln_fp8(
            o, z, mod[l, 1, 2], mod[l, 2, 0], mod[l, 2, 1], x_scales=sc
        )
        if isinstance(b.up, LLLiteLinear):
            gq, gs = rowquant(b.up.fast_fp8(hq, hs, w["up"]), gelu=True)
        else:
            up, sc = fp8_linear_unscaled(hq, hs, w["up"], fast)
            gq, gs = rowquant(up, gelu=True, scales=sc)
        if l + 1 < c.layers:
            y, sc = fp8_linear_unscaled(gq, gs, w["down"], fast)
            hq, hs, z = residual_adaln_fp8(
                y, z, mod[l, 2, 2], mod[l + 1, 0, 0], mod[l + 1, 0, 1], x_scales=sc
            )
        else:
            y = fp8_linear(gq, gs, w["down"], fast)
    shift, scale = fin.chunk(2, dim=-1)
    hid, _ = residual_adaln(y, z, mod[-1, 2, 2], shift, scale + 1)
    return model.unpatchify(model.out_proj(hid), H, W)


def forward(
    model: AnimaDiT,
    rt: Runtime,
    x: torch.Tensor,
    t: torch.Tensor,
    ctx: Context,
    mods: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Predict the velocity for a noisy latent.

    This is ``forward_fp8`` when FP8 is enabled, autograd is disabled and no adapter is attached,
    and ``forward_fused`` otherwise.

    Args:
        model: The model.
        rt: Its runtime.
        x: Noisy latent, shape ``[B, H, W, C]``.
        t: Timesteps (sigmas), shape ``[B]``. Ignored if ``mods`` is given.
        ctx: Context from ``encode_context``, with a batch size matching ``x``.
        mods: Optional precomputed ``(mod, fin)`` from ``modulation``, with batch size ``B``.

    Returns:
        Predicted velocity, shape ``[B, H, W, C]``.
    """
    if rt.fp8 is not None and not torch.is_grad_enabled() and not _adapted(model):
        return forward_fp8(model, rt, x, t, ctx, mods)
    return forward_fused(model, rt, x, t, ctx, mods)


def prepare(
    model: AnimaDiT, rt: Runtime, sigmas: torch.Tensor, cond: torch.Tensor
) -> Prepared:
    """Ready the model for one sampling run.

    The text context and the modulation of every step are computed here, once.

    Args:
        model: The model.
        rt: Its runtime.
        sigmas: Noise level of each step, shape ``[steps]``.
        cond: Text embeddings for the whole batch, shape ``[B, S, text_dim]``, zero-padded along
            ``S``.

    Returns:
        The prepared run. Its ``run`` takes latents of shape ``[B, H, W, C]``.
    """
    dt, n = model.compute_dtype, cond.shape[0]
    ctx = model.encode_context(rt, cond.to(dt))
    mod_all, fin_all = model.modulation(sigmas, dt)
    t = sigmas[:1].expand(n)

    def run(
        x: torch.Tensor, mod: torch.Tensor, fin: torch.Tensor, *fixed: torch.Tensor
    ) -> torch.Tensor:
        mods = (mod.expand(n, -1, -1, -1, -1), fin.expand(n, -1))
        return forward(model, rt, x.to(dt), t, ctx.replace(fixed), mods).float()

    def per_step(i: int) -> tuple[torch.Tensor, ...]:
        return mod_all[i : i + 1], fin_all[i : i + 1]

    return Prepared(run, per_step, tuple(ctx.tensors()), ctx.key())
