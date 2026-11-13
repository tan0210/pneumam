import math
import os
from collections.abc import Hashable, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from ...ops import attention as A
from ...ops.rope import norm_scale
from ...runtime import Runtime
from ...utils import is_experimental_fp8_attention
from ..checkpoint import read_tensors, save
from ..spec import ModelSpec, Prepared
from .config import AnimaConfig
from .keys import PARTS

NORM_EPS = 1e-6
MODULATION_KEYS = ("t_lin1.", "t_lin2.", "t_norm", "mod_w1", "mod_w2", "out_w2")
FP8_LAYERS = ("qkv", "o1", "xq", "o2", "up", "down")


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal timestep embedding, cosine half first.

    Args:
        t: Timesteps, shape ``[B]``.
        dim: Embedding width (even).

    Returns:
        Float32 embedding, shape ``[B, dim]``.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=t.device, dtype=torch.float32)
        / half
    )
    args = t.float()[:, None] * freqs[None]
    return torch.cat([args.cos(), args.sin()], dim=-1)


def rope_tables(
    h: int,
    w: int,
    head_dim: int,
    device: str | torch.device,
    rope_scale: tuple = (1.0, 1.0),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build 3D (time, height, width) rotary tables for an ``h x w`` token grid.

    Args:
        h: Grid height in tokens.
        w: Grid width in tokens.
        head_dim: Attention head width.
        device: Device to build the tables on.
        rope_scale: NTK scaling of the (height, width) frequencies.

    Returns:
        ``(cos, sin)``, each float32 of shape ``[h * w, head_dim // 2]``.
    """
    dh = head_dim // 6 * 2
    dt = head_dim - 2 * dh
    ntk = dh / (dh - 2)
    ar = torch.arange(0, dh, 2, device=device, dtype=torch.float32) / dh
    inv_h = 1.0 / ((10000.0 * rope_scale[0] ** ntk) ** ar)
    inv_w = 1.0 / ((10000.0 * rope_scale[1] ** ntk) ** ar)
    rows = torch.arange(h, device=device, dtype=torch.float32)
    cols = torch.arange(w, device=device, dtype=torch.float32)
    ah = torch.outer(rows, inv_h)[:, None, :].expand(h, w, -1)
    aw = torch.outer(cols, inv_w)[None, :, :].expand(h, w, -1)
    at = torch.zeros(h, w, dt // 2, device=device)
    ang = torch.cat([at, ah, aw], dim=-1).reshape(h * w, -1)
    return ang.cos(), ang.sin()


class Context:
    """Precomputed per-layer cross-attention keys and values for one text context.

    Attributes:
        kv: Per-layer ``(k, v)`` pairs, each ``[B, S, H, hd]``.
        seqused: Valid context length per batch item (int32, ``[B]``) if trimmed, else None.
        ln_pad: Log of the number of trimmed padding keys per batch item (``[B]``) if trimmed,
            else None.
        kv8: Per-layer ``(k, v, k_scale, v_scale)`` with FP8 ``k`` and ``v``, or None. Only set
            for the experimental FP8 attention path.
    """

    def __init__(
        self,
        kv: list[tuple[torch.Tensor, torch.Tensor]],
        seqused: torch.Tensor | None = None,
        ln_pad: torch.Tensor | None = None,
        kv8: list[tuple[torch.Tensor, torch.Tensor, float, float]] | None = None,
    ) -> None:
        self.kv = kv
        self.seqused = seqused
        self.ln_pad = ln_pad
        self.kv8 = kv8

    def tensors(self) -> list[torch.Tensor]:
        """List every tensor the context holds, in a fixed order.

        Returns:
            The tensors. See ``replace``.
        """
        out = [t for pair in self.kv for t in pair]
        if self.seqused is not None and self.ln_pad is not None:
            out += [self.seqused, self.ln_pad]
        if self.kv8 is not None:
            out += [t for k, v, _, _ in self.kv8 for t in (k, v)]
        return out

    def replace(self, tensors: Sequence[torch.Tensor]) -> "Context":
        """Build a context of the same structure around other tensors.

        Args:
            tensors: Replacements for ``tensors()``, in the same order.

        Returns:
            The new context. This one is not modified.
        """
        it = iter(tensors)
        kv = [(next(it), next(it)) for _ in self.kv]
        trimmed = self.seqused is not None and self.ln_pad is not None
        seqused, ln_pad = (next(it), next(it)) if trimmed else (None, None)
        kv8 = None
        if self.kv8 is not None:
            kv8 = [(next(it), next(it), ks, vs) for _, _, ks, vs in self.kv8]
        return Context(kv, seqused, ln_pad, kv8)

    def key(self) -> Hashable:
        """Summarize what the context holds besides its tensors.

        Returns:
            A value that is equal for two contexts whenever one can stand in for the other after
            ``replace``.
        """
        scales = None
        if self.kv8 is not None:
            scales = tuple((ks, vs) for _, _, ks, vs in self.kv8)
        return (self.seqused is not None, scales)

    def attend(self, layer: int, q: torch.Tensor) -> torch.Tensor:
        """Cross-attend queries to this context at one layer.

        Args:
            layer: Block index.
            q: Normalized queries, shape ``[B, Sq, H, hd]``.

        Returns:
            Attention output, shape ``[B, Sq, H * hd]``.
        """
        k, v = self.kv[layer]
        if self.seqused is None or self.ln_pad is None:
            return A.attention(q, k, v)
        return A.cross_attention_trimmed(q, k, v, self.seqused, self.ln_pad)

    def attend_quantized(
        self, layer: int, q: torch.Tensor, q_scale: float
    ) -> torch.Tensor:
        """Cross-attend FP8 queries to the FP8 copy of this context at one layer.

        Args:
            layer: Block index.
            q: FP8 queries holding the normalized queries times ``q_scale``, shape
                ``[B, Sq, H, hd]``.
            q_scale: Factor the queries were multiplied by before quantizing.

        Returns:
            Attention output, bfloat16, shape ``[B, Sq, H * hd]``.

        Raises:
            RuntimeError: If the context has no FP8 copy.
        """
        if self.kv8 is None:
            raise RuntimeError("this context was encoded without an FP8 copy")
        k, v, k_scale, v_scale = self.kv8[layer]
        if self.seqused is None or self.ln_pad is None:
            return A.attention_quantized(q, k, v, q_scale, k_scale, v_scale)
        return A.cross_attention_trimmed_quantized(
            q, k, v, self.seqused, self.ln_pad, q_scale, k_scale, v_scale
        )


class Block(nn.Module):
    """Parameters of one transformer block: self-attention, cross-attention and MLP.

    A block has no forward of its own. See ``forward``.

    Args:
        c: Model config.
    """

    def __init__(self, c: AnimaConfig) -> None:
        super().__init__()
        D, hd = c.dim, c.head_dim
        self.qkv = nn.Linear(D, 3 * D, bias=False)
        self.q_norm = nn.Parameter(torch.ones(hd))
        self.k_norm = nn.Parameter(torch.ones(hd))
        self.o1 = nn.Linear(D, D, bias=False)
        self.xq = nn.Linear(D, D, bias=False)
        self.xkv = nn.Linear(c.text_dim, 2 * D, bias=False)
        self.xq_norm = nn.Parameter(torch.ones(hd))
        self.xk_norm = nn.Parameter(torch.ones(hd))
        self.o2 = nn.Linear(D, D, bias=False)
        self.up = nn.Linear(D, c.mlp_dim, bias=False)
        self.down = nn.Linear(c.mlp_dim, D, bias=False)


class AnimaDiT(nn.Module):
    """Parameters of the Anima diffusion transformer, with fused projections.

    The module holds weights and the small pieces around the blocks. Running it needs a
    ``Runtime``. See ``forward``.

    Args:
        cfg: Model config.
    """

    def __init__(self, cfg: AnimaConfig) -> None:
        super().__init__()
        self.cfg = cfg
        D, L, R = cfg.dim, cfg.layers, cfg.lora_dim
        self.patch = nn.Linear(4 * cfg.in_ch, D, bias=False)
        self.t_lin1 = nn.Linear(D, D, bias=False)
        self.t_lin2 = nn.Linear(D, 3 * D, bias=False)
        self.t_norm = nn.Parameter(torch.ones(D))
        self.mod_w1 = nn.Parameter(torch.zeros((3 * L + 1) * R, D))
        self.mod_w2 = nn.Parameter(torch.zeros(3 * L, 3 * D, R))
        self.out_w2 = nn.Parameter(torch.zeros(2 * D, R))
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(L))
        self.out_proj = nn.Linear(D, 4 * cfg.in_ch, bias=False)
        self.spec = ModelSpec(
            architecture="anima",
            latent_channels=cfg.in_ch,
            patch_size=2,
            shift=3.0,
            lora_targets=dict(PARTS),
        )

    @property
    def compute_dtype(self) -> torch.dtype:
        """Dtype of the activations the model runs in.

        Returns:
            The dtype latents and text embeddings should be cast to.
        """
        return self.patch.weight.dtype

    def prepare(
        self, rt: Runtime, sigmas: torch.Tensor, cond: torch.Tensor
    ) -> Prepared:
        """Ready the model for one sampling run.

        Args:
            rt: The model's runtime.
            sigmas: Noise level of each step, shape ``[steps]``.
            cond: Text embeddings for the whole batch, shape ``[B, S, text_dim]``, zero-padded
                along ``S``.

        Returns:
            The prepared run. Its ``run`` takes latents of shape ``[B, H, W, C]``.
        """
        from .fused import prepare

        return prepare(self, rt, sigmas, cond)

    def tables(
        self, rt: Runtime, h: int, w: int, device: str | torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the rotary tables for a token grid, building them on first use.

        Args:
            rt: Runtime that caches the tables.
            h: Grid height in tokens (latent height // 2).
            w: Grid width in tokens (latent width // 2).
            device: Target device.

        Returns:
            ``(cos, sin)``, each ``[h * w, head_dim // 2]``.
        """
        key = (h, w, device)
        if key not in rt.tables:
            c = self.cfg
            rt.tables[key] = rope_tables(h, w, c.head_dim, device, c.rope_scale)
        return rt.tables[key]

    def modulation(
        self, t: torch.Tensor, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute all adaLN modulation parameters from the timestep.

        Args:
            t: Timesteps (sigmas), shape ``[B]``.
            dtype: Output dtype.

        Returns:
            ``(mod, fin)``: ``mod`` of shape ``[B, L, 3, 3, D]`` indexed by layer, sub-layer and
            shift / scale+1 / gate, and the final-layer (shift, scale) ``fin`` of shape
            ``[B, 2 * D]``.
        """
        c = self.cfg
        B, L, D, R = t.shape[0], c.layers, c.dim, c.lora_dim
        ts = timestep_embedding(t, D).to(self.mod_w1.dtype)
        temb = self.t_lin2(F.silu(self.t_lin1(ts)))
        emb = F.rms_norm(ts, (D,), self.t_norm, NORM_EPS)
        h = (F.silu(emb) @ self.mod_w1.T).view(B, 3 * L + 1, R)
        low = torch.bmm(h[:, :-1].transpose(0, 1), self.mod_w2.transpose(1, 2))
        blk = low.transpose(0, 1) + temb[:, None]
        shift, scale, gate = blk.view(B, L, 3, 3, D).unbind(3)
        mod = torch.stack((shift, scale + 1, gate), dim=3).to(dtype)
        fin = (h[:, -1] @ self.out_w2.T + temb[:, : 2 * D]).to(dtype)
        return mod, fin

    def encode_context(self, rt: Runtime, ctx: torch.Tensor) -> Context:
        """Precompute the cross-attention keys and values of every layer for a text context.

        Call once per prompt batch and reuse across denoising steps.

        Args:
            rt: Runtime whose ``trim_context`` and FP8 state decide what is prepared.
            ctx: Text embeddings, shape ``[B, S, text_dim]``, zero-padded along ``S``.

        Returns:
            A ``Context`` holding per-layer keys and values.
        """
        c = self.cfg
        B, S, _ = ctx.shape
        fp8 = rt.fp8 is not None and is_experimental_fp8_attention()
        seqused = ln_pad = None
        if rt.trim_context and not torch.is_grad_enabled():
            index = torch.arange(1, S + 1, device=ctx.device)
            lengths = ((ctx != 0).any(-1) * index).amax(1).clamp_min(1)
            keep = min(S, (int(lengths.max()) + 7) // 8 * 8)
            ctx = ctx[:, :keep]
            seqused = lengths.to(torch.int32).contiguous()
            ln_pad = torch.log((S - lengths).float())
            S = keep
        kv, kv8 = [], []
        for b in self.blocks:
            k, v = b.xkv(ctx).view(B, S, 2, c.heads, c.head_dim).unbind(2)  # type: ignore
            k = F.rms_norm(k, (c.head_dim,), b.xk_norm.to(k.dtype), c.qk_eps)  # type: ignore
            kv.append((k.contiguous(), v.contiguous()))
            if fp8:
                ks = norm_scale(b.xk_norm)  # type: ignore
                top = max(float(v.abs().max()), 1e-12)
                vs = 2.0 ** math.floor(math.log2(448.0 / top))
                kv8.append((A.to_fp8(k, ks), A.to_fp8(v, vs), ks, vs))
        return Context(kv, seqused, ln_pad, kv8 if fp8 else None)

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """Split a latent into 2x2 patches and embed them.

        Args:
            x: Latent, shape ``[B, H, W, C]`` (H and W even).

        Returns:
            Token embeddings, shape ``[B, (H // 2) * (W // 2), D]``.
        """
        B, H, W, C = x.shape
        x = x.reshape(B, H // 2, 2, W // 2, 2, C).permute(0, 1, 3, 2, 4, 5)
        return self.patch(x.reshape(B, (H // 2) * (W // 2), 4 * C))

    def unpatchify(self, y: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Fold per-patch predictions back into a latent.

        Args:
            y: Per-patch outputs, shape ``[B, (H // 2) * (W // 2), 4 * C]``.
            H: Latent height.
            W: Latent width.

        Returns:
            Latent, shape ``[B, H, W, C]``.
        """
        B, C = y.shape[0], self.cfg.in_ch
        y = y.reshape(B, H // 2, W // 2, 2, 2, C).permute(0, 1, 3, 2, 4, 5)
        return y.reshape(B, H, W, C)


def group_dtype(
    key: str, dtype: torch.dtype, modulation_dtype: torch.dtype, norm_dtype: torch.dtype
) -> torch.dtype:
    """Pick the dtype a checkpoint tensor is loaded as, by name.

    Args:
        key: Tensor name in the checkpoint.
        dtype: Dtype for ordinary weights.
        modulation_dtype: Dtype for the timestep and modulation weights.
        norm_dtype: Dtype for the per-block q/k norm weights.

    Returns:
        The dtype for ``key``.
    """
    if key.startswith(MODULATION_KEYS):
        return modulation_dtype
    if key.startswith("blocks.") and key.endswith("_norm"):
        return norm_dtype
    return dtype


def load_dit(
    path: str | os.PathLike,
    config: str,
    device: str | torch.device,
    dtype: torch.dtype = torch.bfloat16,
    modulation_dtype: torch.dtype = torch.float32,
    norm_dtype: torch.dtype = torch.float32,
) -> AnimaDiT:
    """Build the model from a checkpoint's tensors for inference.

    Tensors of a VAE stored in the same file are ignored.

    Args:
        path: The checkpoint.
        config: Its serialized ``AnimaConfig``.
        device: Device to load the weights onto.
        dtype: Dtype of ordinary weights.
        modulation_dtype: Dtype of the modulation weights.
        norm_dtype: Dtype of the q/k norm weights.

    Returns:
        The model in eval mode with gradients disabled.
    """
    pick = lambda key: group_dtype(key, dtype, modulation_dtype, norm_dtype)
    sd = read_tensors(path, device, pick, skip=("vae.",))
    with torch.device("meta"):
        model = AnimaDiT(AnimaConfig.loads(config))
    model.load_state_dict(sd, strict=True, assign=True)
    model.requires_grad_(False)
    return model.eval()


def save_dit(model: AnimaDiT, path: str | os.PathLike) -> None:
    """Write the model's weights as a checkpoint ``load`` accepts.

    Args:
        model: The model, without adapters attached.
        path: Output path.
    """
    sd = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    save(path, sd, model.spec.architecture, model.cfg.dumps())
