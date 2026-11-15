import os
import re
from typing import Any, cast

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn

from ..ops.gemm import (
    FP8Weight,
    addmm_,
    fp8_addmm_,
    fp8_linear,
    gelu_addmm_,
    linear,
    quantize_weight,
)
from ..ops.lllite import lllite_hidden

MODULE = re.compile(
    r"lllite_dit_blocks_(\d+)_(self_attn_[qkv]_proj|cross_attn_q_proj|mlp_layer1)"
)
TARGETS = {
    "self_attn_q_proj": ("qkv", 3, 0),
    "self_attn_k_proj": ("qkv", 3, 1),
    "self_attn_v_proj": ("qkv", 3, 2),
    "cross_attn_q_proj": ("xq", 1, 0),
    "mlp_layer1": ("up", 1, 0),
}


def _group_norm(channels: int) -> nn.GroupNorm:
    groups = 8
    while groups > 1 and channels % groups:
        groups //= 2
    return nn.GroupNorm(groups, channels)


class _ResBlock(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.norm1, self.conv1 = _group_norm(ch), nn.Conv2d(ch, ch, 3, padding=1)
        self.norm2, self.conv2 = _group_norm(ch), nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        return x + self.conv2(F.silu(self.norm2(h)))


def _conv_norm_silu(conv: nn.Conv2d, ch: int) -> nn.Sequential:
    return nn.Sequential(conv, _group_norm(ch), nn.SiLU())


class _ASPP(nn.Module):
    def __init__(self, ch: int, dilations: tuple[int, ...]) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            _conv_norm_silu(
                nn.Conv2d(ch, ch, 1)
                if d == 1
                else nn.Conv2d(ch, ch, 3, padding=d, dilation=d),
                ch,
            )
            for d in dilations
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_conv = _conv_norm_silu(nn.Conv2d(ch, ch, 1), ch)
        self.proj = _conv_norm_silu(nn.Conv2d(ch * (len(dilations) + 1), ch, 1), ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = [b(x) for b in self.branches]
        pooled = self.global_conv(self.global_pool(x))
        outs.append(
            F.interpolate(
                pooled, size=x.shape[-2:], mode="bilinear", align_corners=False
            )
        )
        return self.proj(torch.cat(outs, dim=1))


class Conditioning(nn.Module):
    """Encodes a control image into one embedding per model token.

    Two stride-4 convolutions give one token per 16x16 pixels, which is one token per 2x2 patch
    of the 8x latent.

    Args:
        in_ch: Image channels: 3 for RGB, 4 for RGB and an inpainting mask.
        dim: Width of the convolutional trunk.
        emb_dim: Width of each token embedding.
        resblocks: Residual blocks after the downsampling.
        dilations: Dilations of the ASPP block, or empty for none.
    """

    def __init__(
        self,
        in_ch: int,
        dim: int,
        emb_dim: int,
        resblocks: int,
        dilations: tuple[int, ...],
    ) -> None:
        super().__init__()
        half = dim // 2
        self.conv1, self.norm1 = nn.Conv2d(in_ch, half, 4, stride=4), _group_norm(half)
        self.conv2, self.norm2 = nn.Conv2d(half, half, 3, padding=1), _group_norm(half)
        self.conv3, self.norm3 = nn.Conv2d(half, dim, 4, stride=4), _group_norm(dim)
        self.resblocks = nn.ModuleList(_ResBlock(dim) for _ in range(resblocks))
        self.aspp = _ASPP(dim, dilations) if dilations else None
        self.proj = nn.Conv2d(dim, emb_dim, 1)
        self.out_norm = nn.LayerNorm(emb_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Embed a control image.

        Args:
            image: Image in ``[-1, 1]``, shape ``[B, in_ch, H, W]`` with ``H`` and ``W``
                multiples of 16.

        Returns:
            Token embeddings, shape ``[B, (H // 16) * (W // 16), emb_dim]``.
        """
        x = F.silu(self.norm1(self.conv1(image)))
        x = F.silu(self.norm2(self.conv2(x)))
        x = F.silu(self.norm3(self.conv3(x)))
        for block in self.resblocks:
            x = block(x)
        if self.aspp is not None:
            x = self.aspp(x)
        return self.out_norm(self.proj(x).flatten(2).transpose(1, 2))


class Module(nn.Module):
    """The LLLite adapter of one projection input.

    Args:
        dim: Width of the projection input.
        emb_dim: Width of the control embeddings.
        mlp_dim: Width of the adapter.
    """

    def __init__(self, dim: int, emb_dim: int, mlp_dim: int) -> None:
        super().__init__()
        self.down = nn.Linear(dim, mlp_dim)
        self.mid = nn.Linear(mlp_dim + emb_dim, mlp_dim)
        self.cond_to_film = nn.Linear(emb_dim, 2 * mlp_dim)
        self.up = nn.Linear(mlp_dim, dim)
        self.depth_embed = nn.Parameter(torch.zeros(emb_dim))

    def hidden(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Compute the adapter's hidden state, before ``up``.

        Args:
            x: Projection input, shape ``[B, N, dim]``.
            cond: Control embeddings, shape ``[B, N, emb_dim]``.

        Returns:
            Shape ``[B, N, mlp_dim]``.
        """
        dtype = self.down.weight.dtype
        x, cond = x.to(dtype), cond.to(dtype) + self.depth_embed
        h = F.silu(self.down(x))
        gamma, beta = self.cond_to_film(cond).chunk(2, dim=-1)
        h = self.mid(torch.cat((cond, h), dim=-1))
        return F.silu(h * (1 + gamma) + beta)


class Control:
    """The control signal that the attached adapters read, with its embeddings and strength.

    Everything the adapters derive from it lives in fixed buffers that are updated in place, so
    CUDA graphs captured with a control stay valid while its image and strength change.

    Args:
        device: Device of the buffers.
    """

    def __init__(self, device: torch.device) -> None:
        self.cond: torch.Tensor | None = None
        self.strength = torch.zeros(1, device=device)
        self.layers: list[LLLiteLinear] = []

    def set_cond(self, cond: torch.Tensor | None) -> bool:
        """Set the control embeddings.

        Args:
            cond: Embeddings from ``LLLite.encode``, shape ``[B, N, emb_dim]``, or None to
                switch the control off.

        Returns:
            True if a buffer was replaced rather than refilled, which invalidates captured
            CUDA graphs.
        """
        if cond is not None and self.cond is not None and cond.shape == self.cond.shape:
            self.cond.copy_(cond)
            return self.refresh()
        self.cond = None if cond is None else cond.detach().clone()
        self.refresh()
        return True

    @torch.no_grad()
    def refresh(self) -> bool:
        """Recompute what the adapters derive from the embeddings and the projection weights.

        Call it after changing the weights of an adapted projection, for example by merging a
        LoRA.

        Returns:
            True if a buffer was replaced rather than refilled.
        """
        replaced = False
        for layer in self.layers:
            replaced |= layer.refresh()
        return replaced

    def set_strength(self, value: float) -> None:
        """Set the strength, for example to 0 outside the control's noise range.

        Args:
            value: Multiplier on every adapter's contribution.
        """
        self.strength.fill_(value)


class LLLiteLinear(nn.Module):
    """A projection whose input gets LLLite control on some output slices.

    ``W (x + s * up(h))`` is computed as ``W x + s * (h @ (W_i up_w).T + W_i up_b)`` for each
    slice ``W_i``, so the base product is not repeated.

    Args:
        base: The projection: an ``nn.Linear`` or another adapter with a ``weight``.
        parts: Equal output slices of the projection.
        modules: Adapter of each slice that has one, by slice index.
        control: The shared control signal.
    """

    def __init__(
        self, base: nn.Module, parts: int, modules: dict[int, Module], control: Control
    ) -> None:
        super().__init__()
        self.base, self.parts, self.control = base, parts, control
        self._mods: list[tuple[int, Module]] = sorted(modules.items())
        self.adapters = nn.ModuleDict({str(i): m for i, m in self._mods})
        self._fast: dict[str, Any] = {}
        control.layers.append(self)

    @property
    def weight(self) -> torch.Tensor:
        """Weight of the wrapped projection.

        Returns:
            Shape ``[out_features, in_features]``.
        """
        return cast(torch.Tensor, self.base.weight)

    def _plain(self) -> bool:
        return type(self.base) is nn.Linear and self.base.bias is None

    def _ready(self) -> bool:
        return bool(self._fast) and self.control.cond is not None and self._plain()

    @torch.no_grad()
    def refresh(self) -> bool:
        """Recompute the buffers of the inference path from the control and the weights.

        Returns:
            True if a buffer was replaced rather than refilled.
        """
        cond = self.control.cond
        if cond is None or not self._plain():
            replaced = bool(self._fast)
            self._fast = {}
            return replaced
        w = self.weight
        size, mods = w.shape[0] // self.parts, self._mods
        m, c = mods[0][1].down.weight.shape[0], cond.shape[-1]
        flat = cond.reshape(-1, c).float()
        cms, gammas, betas = [], [], []
        for _, mod in mods:
            local = flat + mod.depth_embed.float()
            g, b = (
                local @ mod.cond_to_film.weight.float().T
                + mod.cond_to_film.bias.float()
            ).chunk(2, -1)
            cms.append(local @ mod.mid.weight[:, :c].float().T + mod.mid.bias.float())
            gammas.append(g)
            betas.append(b)
        jm = len(mods) * m
        width = -(-(jm + 1) // 64) * 64
        rows = -(-jm // 128) * 128
        down = w.new_zeros(rows, w.shape[1])
        down[:jm] = torch.cat([mod.down.weight for _, mod in mods]).to(w.dtype)
        up = torch.zeros(w.shape[0], width, device=w.device)
        for j, (i, mod) in enumerate(mods):
            rows_i = slice(i * size, (i + 1) * size)
            wi = w[rows_i].float()
            up[rows_i, j * m : (j + 1) * m] = wi @ mod.up.weight.float()
            up[rows_i, jm] += wi @ mod.up.bias.float()
        new = {
            "down": down,
            "down_bias": torch.cat([mod.down.bias for _, mod in mods]).to(w.dtype),
            "mid": torch.stack([mod.mid.weight[:, c:] for _, mod in mods])
            .to(w.dtype)
            .contiguous(),
            "cond_mid": torch.cat(cms, -1).contiguous(),
            "gamma": torch.cat(gammas, -1).contiguous(),
            "beta": torch.cat(betas, -1).contiguous(),
            "up": up.to(w.dtype),
            "down8": quantize_weight(down),
        }
        replaced = new.keys() != self._fast.keys() or any(
            self._fast[k].shape != v.shape
            for k, v in new.items()
            if isinstance(v, torch.Tensor)
        )
        if replaced:
            self._fast = new
        else:
            for k, v in new.items():
                if isinstance(v, torch.Tensor):
                    self._fast[k].copy_(v)
                else:
                    for dst, src in zip(self._fast[k], v, strict=True):
                        dst.copy_(src)
        return replaced

    def _update(self, z: torch.Tensor) -> torch.Tensor:
        f = self._fast
        h = lllite_hidden(
            z,
            f["down_bias"],
            f["mid"],
            f["cond_mid"],
            f["gamma"],
            f["beta"],
            self.control.strength,
            self.control.cond.shape[1],  # type: ignore[union-attr]
            f["up"].shape[1],
        )
        return linear(h, f["up"])

    def _flat(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(-1, x.shape[-1]).contiguous()

    def fast_fp8(self, q: torch.Tensor, s: torch.Tensor, w8: FP8Weight) -> torch.Tensor:
        """Apply the projection with control to FP8 tokens. Forward only.

        Args:
            q: FP8 tokens, shape ``[B, N, in]``. See ``rowquant``.
            s: Their scales, shape ``[B * N, 1]``.
            w8: The projection weight from ``quantize_weight``.

        Returns:
            bfloat16 output, shape ``[B, N, out]``.
        """
        if not self._ready():
            return fp8_linear(q, s, w8)
        B, N, K = q.shape
        z = fp8_linear(q, s, self._fast["down8"]).reshape(B * N, -1)
        out = cast(
            torch.Tensor, fp8_addmm_(self._update(z), q.reshape(B * N, K), s, w8)
        )
        return out.view(B, N, -1)

    def fused_mlp(self, down: nn.Module, x: torch.Tensor) -> torch.Tensor | None:
        """Apply this projection, an exact erf GELU and ``down`` as one step. Forward only.

        Args:
            down: The plain linear layer after the GELU.
            x: Input, shape ``[..., in_features]``.

        Returns:
            ``down(gelu(self(x)))``, or None while autograd is on, without control, or if
            ``down`` is not a plain bias-free linear layer.
        """
        plain = type(down) is nn.Linear and down.bias is None
        if torch.is_grad_enabled() or not plain or not self._ready():
            return None
        flat = self._flat(x)
        update = self._update(linear(flat, self._fast["down"]))
        post, _ = gelu_addmm_(update, flat, self.weight)
        return linear(post, cast(torch.Tensor, down.weight)).view(*x.shape[:-1], -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the projection with control.

        Without autograd this runs fused kernels. With it, it runs plain differentiable ops.

        Args:
            x: Input, shape ``[B, N, in]`` or ``[B * N, in]`` with ``N`` control tokens.

        Returns:
            Output, shape ``[..., out]``.
        """
        if not torch.is_grad_enabled() and self._ready():
            flat = self._flat(x)
            update = self._update(linear(flat, self._fast["down"]))
            return addmm_(update, flat, self.weight).view(*x.shape[:-1], -1)
        out = self.base(x)
        cond = self.control.cond
        if cond is None:
            return out
        n = cond.shape[1]
        x3 = x.reshape(-1, n, x.shape[-1])
        if (
            cond.shape[0] != x3.shape[0]
        ):  # one control image for a guided or larger batch
            cond = cond.repeat(x3.shape[0] // cond.shape[0], 1, 1)
        w = self.weight
        size = w.shape[0] // self.parts
        out3 = out.reshape(x3.shape[0], n, -1)
        if torch.is_grad_enabled():
            out3 = out3.clone()  # the base output may be saved for its backward
        s = self.control.strength.to(out.dtype)
        for i, m in self._mods:
            wi = w[i * size : (i + 1) * size]
            h = m.hidden(x3, cond)
            folded = wi.to(h.dtype) @ m.up.weight.to(h.dtype)
            bias = wi.to(h.dtype) @ m.up.bias.to(h.dtype)
            out3[..., i * size : (i + 1) * size] += s * (h @ folded.T + bias)
        return out3.reshape(out.shape)


class LLLite(nn.Module):
    """A ControlNet-LLLite model for Anima, in the format of kohya's sd-scripts and ComfyUI.

    Args:
        sd: Its tensors.
        metadata: Its safetensors metadata.

    Raises:
        ValueError: If the file is not a version 2 LLLite model or has unknown modules.
    """

    def __init__(
        self, sd: dict[str, torch.Tensor], metadata: dict[str, str] | None = None
    ) -> None:
        super().__init__()
        metadata = metadata or {}
        version = metadata.get("lllite.version", "2")
        if version != "2":
            raise ValueError(f"unsupported LLLite version {version!r}; expected '2'")
        prefix = "lllite_conditioning1."
        in_ch = sd[prefix + "conv1.weight"].shape[1]
        dim = sd[prefix + "conv3.weight"].shape[0]
        emb_dim = sd[prefix + "proj.weight"].shape[0]
        blocks = {
            int(k.split(".")[2]) for k in sd if k.startswith(prefix + "resblocks.")
        }
        dilations: tuple[int, ...] = ()
        if any(k.startswith(prefix + "aspp.") for k in sd):
            text = metadata.get("lllite.aspp_dilations", "1,2,4,8")
            dilations = tuple(int(v) for v in text.split(",") if v.strip())
        self.in_ch = in_ch
        self.metadata = dict(metadata)
        self.masked_input = (
            metadata.get("lllite.inpaint_masked_input", "false").lower() == "true"
        )
        self.conditioning = Conditioning(
            in_ch, dim, emb_dim, max(blocks, default=-1) + 1, dilations
        )
        self.targets: dict[int, dict[str, Module]] = {}
        mods = {}
        for name in sorted(
            {k.split(".", 1)[0] for k in sd if k.startswith("lllite_dit_blocks_")}
        ):
            match = MODULE.fullmatch(name)
            if match is None:
                raise ValueError(f"unknown LLLite module {name}")
            mlp_dim, in_dim = sd[f"{name}.down.weight"].shape
            mods[name] = Module(in_dim, emb_dim, mlp_dim)
            self.targets.setdefault(int(match.group(1)), {})[match.group(2)] = mods[
                name
            ]
        self.adapters = nn.ModuleDict(mods)
        own = {
            k.replace(prefix, "conditioning.", 1)
            if k.startswith(prefix)
            else "adapters." + k: v
            for k, v in sd.items()
        }
        self.load_state_dict(own)

    def encode(
        self, image: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Embed a control image.

        Args:
            image: Image in ``[0, 1]``, shape ``[B, 3, H, W]`` at the generated size.
            mask: Inpainting mask in ``[0, 1]``, shape ``[B, 1, H, W]``, where 1 marks the area to
                generate. Required by 4-channel models, ignored by others.

        Returns:
            Control embeddings, shape ``[B, (H // 16) * (W // 16), emb_dim]``.

        Raises:
            ValueError: If the model needs a mask and none is given.
        """
        p = next(self.parameters())
        x = image.to(p.device, p.dtype).clamp(0, 1) * 2 - 1
        if self.in_ch == 4:
            if mask is None:
                raise ValueError("this LLLite model is for inpainting and needs a mask")
            m = (mask.to(p.device) >= 0.5).to(p.dtype)
            if self.masked_input:
                x = x * (1 - m)
            x = torch.cat((x, m * 2 - 1), dim=1)
        return self.conditioning(x)


def new_lllite(
    dim: int,
    layers: int,
    targets: tuple[str, ...] = (
        "self_attn_q_proj",
        "self_attn_k_proj",
        "self_attn_v_proj",
        "mlp_layer1",
    ),
    in_ch: int = 3,
    cond_dim: int = 64,
    emb_dim: int = 32,
    mlp_dim: int = 64,
    resblocks: int = 4,
    masked_input: bool = False,
    seed: int = 0,
) -> LLLite:
    """Create an untrained ControlNet-LLLite model for Anima.

    Every adapter starts with a zero ``up`` projection, so the model is unchanged until
    training moves it.

    Args:
        dim: Width of the model.
        layers: Number of blocks to adapt, from the first.
        targets: Projection inputs to adapt. See ``TARGETS``.
        in_ch: Image channels: 3 for RGB, 4 for inpainting with a mask.
        cond_dim: Width of the image encoder trunk.
        emb_dim: Width of the control embeddings.
        mlp_dim: Width of each adapter.
        resblocks: Residual blocks in the image encoder.
        masked_input: For inpainting, zero the image under the mask before encoding it.
        seed: Seed of the initial weights.

    Returns:
        The model in float32 on the CPU, with trainable weights.

    Raises:
        ValueError: If a target is unknown.
    """
    unknown = [t for t in targets if t not in TARGETS]
    if unknown:
        raise ValueError(
            f"unknown LLLite targets {unknown}; expected some of {tuple(TARGETS)}"
        )
    torch.manual_seed(seed)
    cond = Conditioning(in_ch, cond_dim, emb_dim, resblocks, ())
    sd = {"lllite_conditioning1." + k: v for k, v in cond.state_dict().items()}
    for i in range(layers):
        for t in targets:
            mod = Module(dim, emb_dim, mlp_dim)
            nn.init.zeros_(mod.up.weight)
            nn.init.zeros_(mod.up.bias)
            for k, v in mod.state_dict().items():
                sd[f"lllite_dit_blocks_{i}_{t}.{k}"] = v
    metadata = {
        "lllite.version": "2",
        "lllite.cond_in_channels": str(in_ch),
        "lllite.cond_dim": str(cond_dim),
        "lllite.cond_emb_dim": str(emb_dim),
        "lllite.mlp_dim": str(mlp_dim),
        "lllite.cond_resblocks": str(resblocks),
        "lllite.use_aspp": "false",
        "lllite.inpaint_masked_input": str(masked_input).lower(),
    }
    return LLLite(sd, metadata).requires_grad_(True)


def save_lllite(
    lllite: LLLite, path: str | os.PathLike, dtype: torch.dtype = torch.bfloat16
) -> None:
    """Write an LLLite model in the format of kohya's sd-scripts, which ComfyUI also reads.

    Args:
        lllite: The model.
        path: Output file.
        dtype: Dtype of the written weights.
    """
    sd = {}
    for k, v in lllite.state_dict().items():
        if k.startswith("conditioning."):
            k = "lllite_conditioning1." + k[len("conditioning.") :]
        elif k.startswith("adapters."):
            k = k[len("adapters.") :]
        sd[k] = v.detach().to("cpu", dtype).contiguous()
    save_file(sd, os.fspath(path), metadata=lllite.metadata)


def load_lllite(
    path: str | os.PathLike,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> LLLite:
    """Load a ControlNet-LLLite file for Anima.

    Args:
        path: A safetensors file from kohya's sd-scripts.
        device: Device to load it onto.
        dtype: Dtype of its weights.

    Returns:
        The model in eval mode with frozen weights.
    """
    with safe_open(os.fspath(path), "pt") as f:
        names = list(f.keys())
        sd = {k: f.get_tensor(k) for k in names}
        metadata = f.metadata() or {}
    return LLLite(sd, metadata).to(device, dtype).requires_grad_(False).eval()


def attach_lllite(model: nn.Module, lllite: LLLite) -> Control:
    """Attach an LLLite model's adapters to the model's projections.

    Args:
        model: A model from ``load``. It may already carry LoRA adapters.
        lllite: The LLLite model.

    Returns:
        The control the adapters read. They do nothing until its embeddings are set.

    Raises:
        ValueError: If the model already carries LLLite adapters.
    """
    if any(isinstance(m, LLLiteLinear) for m in model.modules()):
        raise ValueError("the model already has LLLite adapters")
    control = Control(next(lllite.parameters()).device)
    for i, targets in lllite.targets.items():
        block = model.blocks[i]  # type: ignore[index]
        by_layer: dict[str, tuple[int, dict[int, Module]]] = {}
        for target, mod in targets.items():
            layer, parts, part = TARGETS[target]
            by_layer.setdefault(layer, (parts, {}))[1][part] = mod
        for layer, (parts, mods) in by_layer.items():
            setattr(
                block, layer, LLLiteLinear(getattr(block, layer), parts, mods, control)
            )
    return control


def detach_lllite(model: nn.Module) -> None:
    """Remove all LLLite adapters from the model.

    Args:
        model: The model.
    """
    for block in model.blocks:  # type: ignore[attr-defined]
        for name, child in list(block.named_children()):
            if isinstance(child, LLLiteLinear):
                setattr(block, name, child.base)
