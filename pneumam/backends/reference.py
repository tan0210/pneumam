import math

import torch
import torch.nn.functional as F

from ._fp8 import FP8, FP8_MAX, FP8Weight, to_fp8

available = True


def rms_silu(x: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    """Channel RMS norm followed by SiLU.

    Args:
        x: Activations with channels last, shape ``[..., C]``.
        gamma: Per-channel scale, shape ``[C]``.

    Returns:
        Tensor with the shape and dtype of ``x``.
    """
    s = x.float()
    rstd = x.shape[-1] ** 0.5 / s.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return F.silu(s * rstd * gamma.float()).to(x.dtype)


def add_rms_silu(
    x: torch.Tensor, residual: torch.Tensor, gamma: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Residual add followed by ``rms_silu`` of the sum.

    Args:
        x: Branch output with channels last, shape ``[..., C]``.
        residual: Skip connection, same shape as ``x``.
        gamma: Per-channel scale of the following norm, shape ``[C]``.

    Returns:
        ``(x + residual, rms_silu(x + residual, gamma))``.
    """
    s = (x.float() + residual.float()).to(x.dtype)
    return s, rms_silu(s, gamma)


def shuffle_rms_silu(
    u: torch.Tensor, gamma: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Depth-to-space by 2 followed by ``rms_silu``.

    Args:
        u: Sub-pixel activations, shape ``[B, H, W, 4 * C]``. Channel ``(2 * a + b) * C + c`` is the
            value of channel ``c`` at output row ``2 * i + a``, column ``2 * j + b``.
        gamma: Per-channel scale, shape ``[C]``.

    Returns:
        ``(x, rms_silu(x, gamma))`` with ``x`` of shape ``[B, 2 * H, 2 * W, C]``.
    """
    B, H, W, _ = u.shape
    C = gamma.numel()
    x = (
        u.reshape(B, H, W, 2, 2, C)
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(B, 2 * H, 2 * W, C)
    )
    return x, rms_silu(x, gamma)


def head_norm(
    x: torch.Tensor, weight: torch.Tensor, heads: int, eps: float = 1e-5
) -> torch.Tensor:
    """Split tokens into heads and RMS-normalize each head.

    Args:
        x: Tokens, shape ``[B, S, heads * hd]``.
        weight: Per-dimension scale, shape ``[hd]``.
        heads: Number of heads.
        eps: Epsilon added to the mean square.

    Returns:
        Tensor of shape ``[B, S, heads, hd]`` with the dtype of ``x``.
    """
    B, S, _ = x.shape
    h = x.reshape(B, S, heads, -1).float()
    rstd = torch.rsqrt(h.square().mean(-1, keepdim=True) + eps)
    return (h * rstd * weight.float()).to(x.dtype)


def _qkv(qkv, q_weight, k_weight, cos, sin, heads, eps):
    B, S, _ = qkv.shape
    q, k, v = qkv.reshape(B, S, 3, heads, -1).unbind(2)
    cos, sin = cos.float()[None, :, None, :], sin.float()[None, :, None, :]

    def norm_rotate(x, weight):
        # the kernel rounds the normalized value to the input dtype before rotating it
        a, b = head_norm(x.flatten(2), weight, heads, eps).float().chunk(2, dim=-1)
        return torch.cat((a * cos - b * sin, b * cos + a * sin), dim=-1)

    return norm_rotate(q, q_weight), norm_rotate(k, k_weight), v


def head_norm_fp8(
    x: torch.Tensor,
    weight: torch.Tensor,
    heads: int,
    scale: float,
    eps: float = 1e-5,
) -> torch.Tensor:
    """``head_norm`` whose output is FP8, multiplied by a fixed scale.

    Values beyond 448 in magnitude after scaling are clipped.

    Args:
        x: Tokens, shape ``[B, S, heads * hd]``.
        weight: Per-dimension scale, shape ``[hd]``.
        heads: Number of heads.
        scale: Factor applied before quantizing.
        eps: Epsilon added to the mean square.

    Returns:
        FP8 tensor of shape ``[B, S, heads, hd]``.
    """
    return to_fp8(head_norm(x, weight, heads, eps), scale)


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

    The rotation pairs dimension ``i`` with dimension ``i + hd // 2``.

    Args:
        qkv: Fused projection, shape ``[B, S, 3 * heads * hd]``.
        q_weight: Q norm scale, shape ``[hd]``.
        k_weight: K norm scale, shape ``[hd]``.
        cos: Rotary cosines, shape ``[S, hd // 2]``.
        sin: Rotary sines, shape ``[S, hd // 2]``.
        heads: Number of heads.
        eps: Epsilon added to the mean square.

    Returns:
        ``(q, k, v)``, each ``[B, S, heads, hd]`` with the dtype of ``qkv``.
    """
    q, k, v = _qkv(qkv, q_weight, k_weight, cos, sin, heads, eps)
    return q.to(qkv.dtype), k.to(qkv.dtype), v


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
    """``qkv_norm_rope`` whose outputs are FP8, each multiplied by a fixed scale.

    Values beyond 448 in magnitude after scaling are clipped.

    Args:
        qkv: Fused projection, shape ``[B, S, 3 * heads * hd]``.
        q_weight: Q norm scale, shape ``[hd]``.
        k_weight: K norm scale, shape ``[hd]``.
        cos: Rotary cosines, shape ``[S, hd // 2]``.
        sin: Rotary sines, shape ``[S, hd // 2]``.
        heads: Number of heads.
        q_scale: Factor applied to Q before quantizing.
        k_scale: Factor applied to K before quantizing.
        v_scale: Factor applied to V before quantizing.
        eps: Epsilon added to the mean square.

    Returns:
        ``(q, k, v)``, each FP8 of shape ``[B, S, heads, hd]``.
    """
    q, k, v = _qkv(qkv, q_weight, k_weight, cos, sin, heads, eps)
    return to_fp8(q, q_scale), to_fp8(k, k_scale), to_fp8(v, v_scale)


def _adaln(z, shift, scale, eps):
    zf = z.float()
    d = zf - zf.mean(-1, keepdim=True)
    rstd = torch.rsqrt(d.square().mean(-1, keepdim=True) + eps)
    return d * rstd * scale.float()[:, None] + shift.float()[:, None]


def _residual(x, residual, gate):
    prod = (gate.float()[:, None] * x.float()).to(residual.dtype)
    return (residual.float() + prod.float()).to(residual.dtype)


def adaln(
    z: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """LayerNorm without learned parameters, then a per-sample scale and shift.

    Args:
        z: Tokens, shape ``[B, S, D]``.
        shift: Shift, shape ``[B, D]``.
        scale: Scale (already including the +1), shape ``[B, D]``.
        eps: LayerNorm epsilon.

    Returns:
        Tensor with the shape and dtype of ``z``.
    """
    return _adaln(z, shift, scale, eps).to(z.dtype)


def residual_adaln(
    x: torch.Tensor,
    residual: torch.Tensor,
    gate: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gated residual add, then ``adaln`` of the updated residual stream.

    Args:
        x: Sub-layer output, shape ``[B, S, D]``.
        residual: Residual stream, same shape and dtype as ``x``.
        gate: Gate, shape ``[B, D]``.
        shift: Shift, shape ``[B, D]``.
        scale: Scale (already including the +1), shape ``[B, D]``.
        eps: LayerNorm epsilon.

    Returns:
        ``(h, z)``: ``z = residual + gate * x`` and ``h = adaln(z, shift, scale)``.
    """
    z = _residual(x, residual, gate)
    return _adaln(z, shift, scale, eps).to(z.dtype), z


def _rowquant(x):
    B, S, _ = x.shape
    s = x.abs().amax(-1, keepdim=True).clamp_min(1e-12) / FP8_MAX
    q = (x / s).clamp(-FP8_MAX, FP8_MAX).to(FP8)
    return q, s.reshape(B * S, 1)


def _scaled(x, scales):
    if scales is None:
        return x
    rows, cols = scales
    return x.float() * rows.view(*x.shape[:-1], 1) * cols


def rowquant(
    x: torch.Tensor,
    gelu: bool = False,
    scales: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize tokens to FP8 with a scale per token, optionally applying GELU first.

    Args:
        x: Tokens, shape ``[B, S, K]``.
        gelu: Apply the exact erf GELU before quantizing.
        scales: Float32 ``(rows, cols)`` of shapes ``[B * S, 1]`` and ``[K]``. If given, the
            tokens are ``x * rows * cols``.

    Returns:
        ``(q, s)``: the FP8 tokens ``[B, S, K]`` and their float32 scales ``[B * S, 1]``.
    """
    x = _scaled(x, scales).float()
    return _rowquant(F.gelu(x) if gelu else x)


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
    return _rowquant(_adaln(z, shift, scale, eps))


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
    """
    z = _residual(_scaled(x, x_scales), residual, gate)
    return (*_rowquant(_adaln(z, shift, scale, eps)), z)


def fp8_linear(
    q: torch.Tensor, s: torch.Tensor, w: FP8Weight, fast_accum: bool = False
) -> torch.Tensor:
    """Multiply FP8 tokens by an FP8 weight.

    Args:
        q: FP8 tokens, shape ``[B, S, K]``.
        s: Token scales, shape ``[B * S, 1]``.
        w: Quantized weight, shape ``[N, K]``.
        fast_accum: Ignored.

    Returns:
        bfloat16 result, shape ``[B, S, N]``.
    """
    B, S, K = q.shape
    a = q.float().view(B * S, K) * s
    return (a @ (w.q.float() * w.s[:, None]).T).to(torch.bfloat16).view(B, S, -1)


def fp8_gemm(q: torch.Tensor, w: FP8Weight) -> torch.Tensor:
    """Multiply FP8 tokens by an FP8 weight without applying either scale.

    Args:
        q: FP8 tokens, shape ``[B, S, K]``.
        w: Quantized weight, shape ``[N, K]``.

    Returns:
        bfloat16 product of the stored values, shape ``[B, S, N]``. Multiply it by the token
        scales and by ``w.s`` to get the true product.
    """
    return (q.float() @ w.q.float().T).to(torch.bfloat16)


def linear_gelu(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Compute ``gelu(x @ weight.T)`` with the exact erf GELU and no bias.

    Args:
        x: Activations, shape ``[..., K]``.
        weight: Weight, shape ``[N, K]``.

    Returns:
        Tensor with the dtype of ``x``, shape ``[..., N]``.
    """
    return F.gelu(F.linear(x, weight))


def mlp(x: torch.Tensor, w_up: torch.Tensor, w_down: torch.Tensor) -> torch.Tensor:
    """Compute ``gelu(x @ w_up.T) @ w_down.T`` with the exact erf GELU and no biases.

    Args:
        x: Activations, shape ``[..., K]``.
        w_up: Weight, shape ``[H, K]``.
        w_down: Weight, shape ``[N, H]``.

    Returns:
        Tensor with the dtype of ``x``, shape ``[..., N]``.
    """
    return F.linear(linear_gelu(x, w_up), w_down)


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Full, non-causal multi-head attention.

    Args:
        q: Queries, shape ``[B, Sq, H, hd]``.
        k: Keys, shape ``[B, Sk, H, hd]``.
        v: Values, shape ``[B, Sk, H, hd]``.

    Returns:
        Attention output, shape ``[B, Sq, H, hd]``, dtype of ``q``.
    """
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    )
    return out.transpose(1, 2)


def attention_trimmed(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, seqused_k: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Attention over the first ``seqused_k[b]`` keys of each batch item.

    Args:
        q: Queries, shape ``[B, Sq, H, hd]``.
        k: Keys, shape ``[B, Sk, H, hd]``.
        v: Values, shape ``[B, Sk, H, hd]``.
        seqused_k: Valid key count per batch item, shape ``[B]``.

    Returns:
        ``(out, lse)``: the output ``[B, Sq, H, hd]`` and the float32 natural-log softmax normalizer
        ``[B, H, Sq]``.
    """
    scores = torch.einsum("bqhd,bkhd->bhqk", q.float(), k.float()) / math.sqrt(
        q.shape[-1]
    )
    valid = torch.arange(k.shape[1], device=k.device)[None] < seqused_k[:, None]
    scores = scores.masked_fill(~valid[:, None, None, :], float("-inf"))
    lse = scores.logsumexp(-1)
    out = torch.einsum("bhqk,bkhd->bqhd", scores.softmax(-1), v.float())
    return out.to(q.dtype), lse


def attention_trimmed_biased(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seqused_k: torch.Tensor,
    ln_pad: torch.Tensor,
) -> torch.Tensor:
    """Attention over a trimmed context that stands in for the removed zero padding.

    The result equals attention over the first ``seqused_k[b]`` keys extended with ``P`` zero
    keys and zero values.

    Args:
        q: Queries, shape ``[B, Sq, H, hd]``.
        k: Keys, shape ``[B, Sk, H, hd]``.
        v: Values, shape ``[B, Sk, H, hd]``.
        seqused_k: Valid key count per batch item, shape ``[B]``.
        ln_pad: ``log(P)`` per batch item, ``-inf`` where nothing was trimmed, shape ``[B]``.

    Returns:
        Attention output, shape ``[B, Sq, H, hd]``, dtype of ``q``.
    """
    B, Sk = k.shape[:2]
    valid = torch.arange(Sk, device=k.device)[None] < seqused_k[:, None]
    bias = torch.zeros(B, Sk + 1, dtype=torch.float32, device=k.device)
    bias[:, :Sk].masked_fill_(~valid, float("-inf"))
    bias[:, Sk] = ln_pad.float()  # one zero key weighted like P of them
    one = (0, 0, 0, 0, 0, 1)
    kt, vt = F.pad(k, one).transpose(1, 2), F.pad(v, one).transpose(1, 2)
    mask = bias[:, None, None, :].to(q.dtype)
    out = F.scaled_dot_product_attention(q.transpose(1, 2), kt, vt, attn_mask=mask)
    return out.transpose(1, 2)


def ctx_scale(
    out: torch.Tensor, lse: torch.Tensor, ln_pad: torch.Tensor
) -> torch.Tensor:
    """Rescale trimmed attention so it matches attention over the zero-padded context.

    The result equals attention over the context extended with ``P`` zero keys and zero values.

    Args:
        out: Attention output over the trimmed context, shape ``[B, S, H, hd]``.
        lse: Its natural-log softmax normalizer, shape ``[B, H, S]``.
        ln_pad: ``log(P)`` per batch item, ``-inf`` where nothing was trimmed, shape ``[B]``.

    Returns:
        Tensor with the shape and dtype of ``out``.
    """
    gain = torch.sigmoid(lse.float() - ln_pad.float()[:, None, None])
    return (out.float() * gain.transpose(1, 2)[..., None]).to(out.dtype)
