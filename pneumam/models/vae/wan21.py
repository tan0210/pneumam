import os
from typing import Any, cast

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn

from ...graphs import Graphed
from ...ops.attention import wide_attention
from ...ops.norm import add_rms_silu, rms_silu, shuffle_rms_silu
from .latents import denormalize, from_bchw, normalize, to_bchw

FORMAT = "pneumam-vae-1"
ARCHITECTURE = "wan21"
_LEGACY_PREFIX = "vae."  # where older model checkpoints keep their decoder

# row a of _PHASE[phase] folds the taps of a 3x3 kernel applied after a nearest 2x upsample
# onto the taps of a 3x3 kernel over the original grid, for outputs of parity phase
_PHASE = torch.tensor(
    [[[1.0, 0, 0], [0, 1, 1], [0, 0, 0]], [[0, 0, 0], [1, 1, 0], [0, 0, 1]]]
)


def _p(t: torch.Tensor) -> nn.Parameter:
    t = t.detach().clone()
    if t.ndim == 4:
        t = t.contiguous(memory_format=torch.channels_last)
    return nn.Parameter(t)


def _nhwc(t: torch.Tensor) -> torch.Tensor:
    return t.permute(0, 2, 3, 1)


def _nchw(t: torch.Tensor) -> torch.Tensor:
    return t.permute(0, 3, 1, 2)


def _conv2d_weight(conv: Any) -> torch.Tensor:
    w = conv.weight
    return w[:, :, -1] if w.ndim == 5 else w  # a single frame only meets the last tap


def _put3d(conv: Any, w: torch.Tensor, tap: int) -> None:
    with torch.no_grad():
        conv.weight[:, :, tap] = w.reshape(conv.weight[:, :, tap].shape).to(conv.weight)


def _put(dst: torch.Tensor, src: torch.Tensor) -> None:
    with torch.no_grad():
        dst.copy_(src.reshape(dst.shape).to(dst))


def _check_layout(vae: Any) -> None:
    patched = getattr(vae.config, "patch_size", None) not in (None, 1)
    if patched or getattr(vae.config, "is_residual", False):
        raise ValueError("only the Wan 2.1 VAE layout is supported")


def _stats_tensors(vae: Any) -> dict[str, torch.Tensor]:
    return {
        "mean": torch.tensor(vae.config.latents_mean, dtype=torch.float32),
        "std": torch.tensor(vae.config.latents_std, dtype=torch.float32),
    }


def _res_tensors(b: Any, prefix: str) -> dict[str, torch.Tensor]:
    out = {
        prefix + "g1": b.norm1.gamma.reshape(-1),
        prefix + "w1": _conv2d_weight(b.conv1),
        prefix + "b1": b.conv1.bias,
        prefix + "g2": b.norm2.gamma.reshape(-1),
        prefix + "w2": _conv2d_weight(b.conv2),
        prefix + "b2": b.conv2.bias,
    }
    if not isinstance(b.conv_shortcut, nn.Identity):
        out[prefix + "sw"] = b.conv_shortcut.weight[:, :, 0]
        out[prefix + "sb"] = b.conv_shortcut.bias
    return out


def _attn_tensors(a: Any, names: tuple[str, ...]) -> dict[str, torch.Tensor]:
    values = (
        a.norm.gamma.reshape(-1),
        a.to_qkv.weight,
        a.to_qkv.bias,
        a.proj.weight,
        a.proj.bias,
    )
    return dict(zip(names, values, strict=True))


def _attention_branch(
    x: torch.Tensor,
    g: torch.Tensor,
    qkv_w: torch.Tensor,
    qkv_b: torch.Tensor,
    proj_w: torch.Tensor,
    proj_b: torch.Tensor,
) -> torch.Tensor:
    B, C, H, W = x.shape
    t = _nhwc(x).reshape(B, 1, H * W, C)  # a view when x is channels_last
    n = F.normalize(t.float(), dim=-1).to(x.dtype)
    w = qkv_w.reshape(3 * C, C) * (g * C**0.5)  # fold the norm gain into the weights
    wq, wk, wv = w.chunk(3)
    bq, bk, bv = qkv_b.chunk(3)
    q, k, v = F.linear(n, wq, bq), F.linear(n, wk, bk), F.linear(n, wv, bv)
    o = F.linear(wide_attention(q, k, v), proj_w.reshape(C, C), proj_b)
    return _nchw(o.reshape(B, H, W, C))  # channels_last, without the residual


def _attention(
    x: torch.Tensor,
    g: torch.Tensor,
    qkv_w: torch.Tensor,
    qkv_b: torch.Tensor,
    proj_w: torch.Tensor,
    proj_b: torch.Tensor,
) -> torch.Tensor:
    return _attention_branch(x, g, qkv_w, qkv_b, proj_w, proj_b) + x


class _Res(nn.Module):
    def __init__(self, sd: dict[str, torch.Tensor], prefix: str) -> None:
        super().__init__()
        self.g1 = _p(sd[prefix + "g1"])
        self.w1, self.b1 = _p(sd[prefix + "w1"]), _p(sd[prefix + "b1"])
        self.g2 = _p(sd[prefix + "g2"])
        self.w2, self.b2 = _p(sd[prefix + "w2"]), _p(sd[prefix + "b2"])
        self.has_shortcut = prefix + "sw" in sd
        if self.has_shortcut:
            self.sw, self.sb = _p(sd[prefix + "sw"]), _p(sd[prefix + "sb"])

    def to_diffusers(self, b: Any) -> None:
        _put(b.norm1.gamma, self.g1)
        _put3d(b.conv1, self.w1, 2)
        _put(b.conv1.bias, self.b1)
        _put(b.norm2.gamma, self.g2)
        _put3d(b.conv2, self.w2, 2)
        _put(b.conv2.bias, self.b2)
        if self.has_shortcut:
            _put3d(b.conv_shortcut, self.sw, 0)
            _put(b.conv_shortcut.bias, self.sb)

    def branch(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = F.conv2d(x, self.sw, self.sb) if self.has_shortcut else x
        c = F.conv2d(y, self.w1, self.b1, padding=1)
        z = _nchw(rms_silu(_nhwc(c), self.g2))
        return F.conv2d(z, self.w2, self.b2, padding=1), h


class _Down(nn.Module):
    def __init__(self, sd: dict[str, torch.Tensor], prefix: str) -> None:
        super().__init__()
        self.w, self.b = _p(sd[prefix + "w"]), _p(sd[prefix + "b"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(F.pad(x, (0, 1, 0, 1)), self.w, self.b, stride=2)


class _Attn(nn.Module):
    def __init__(self, sd: dict[str, torch.Tensor], prefix: str) -> None:
        super().__init__()
        self.g = _p(sd[prefix + "g"])
        self.qkv_w, self.qkv_b = _p(sd[prefix + "qkv_w"]), _p(sd[prefix + "qkv_b"])
        self.proj_w, self.proj_b = _p(sd[prefix + "proj_w"]), _p(sd[prefix + "proj_b"])

    def branch(self, x: torch.Tensor) -> torch.Tensor:
        return _attention_branch(
            x, self.g, self.qkv_w, self.qkv_b, self.proj_w, self.proj_b
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _attention(x, self.g, self.qkv_w, self.qkv_b, self.proj_w, self.proj_b)


class _Half(nn.Module):
    mean: torch.Tensor
    std: torch.Tensor
    scale_factor: int = 8

    def __init__(self, sd: dict[str, torch.Tensor]) -> None:
        super().__init__()
        self.register_buffer("mean", sd["mean"].detach().clone().float())
        self.register_buffer("std", sd["std"].detach().clone().float())

    def _apply(self, fn, recurse=True):  # type: ignore
        mean, std = self.mean, self.std
        super()._apply(fn, recurse)
        device = self.mean.device
        self.mean, self.std = (
            mean.to(device),
            std.to(device),
        )  # stay float32 under casts
        return self


class Wan21Decoder(_Half):
    """Decoder of the Wan 2.1 VAE, which Qwen-Image also uses, for single images.

    It is differentiable with respect to the latent and to all decoder weights.

    Args:
        sd: Tensors as returned by ``state_dict()``, including ``mean`` and ``std``. They are
            copied.

    Attributes:
        mean: Per-channel mean of the raw latents, float32.
        std: Per-channel standard deviation of the raw latents, float32.
        scale_factor: Image pixels per latent cell along each side.
        graphs: CUDA graphs captured by ``decode``. Moving or casting the module clears them.

    Raises:
        KeyError: If ``sd`` lacks a tensor the decoder needs.
    """

    def __init__(self, sd: dict[str, torch.Tensor]) -> None:
        super().__init__(sd)
        self.graphs: dict[Any, Graphed] = {}
        self.register_buffer("phase", _PHASE.clone(), persistent=False)
        self.post_w, self.post_b = _p(sd["post_w"]), _p(sd["post_b"])
        self.in_w, self.in_b = _p(sd["in_w"]), _p(sd["in_b"])
        self.mid0, self.mid1 = _Res(sd, "mid0."), _Res(sd, "mid1.")
        self.attn_g = _p(sd["attn_g"])
        self.qkv_w, self.qkv_b = _p(sd["qkv_w"]), _p(sd["qkv_b"])
        self.proj_w, self.proj_b = _p(sd["proj_w"]), _p(sd["proj_b"])
        self.blocks = nn.ModuleList()
        i = 0
        while f"blocks.{i}.0.g1" in sd:
            res, j = [], 0
            while f"blocks.{i}.{j}.g1" in sd:
                res.append(_Res(sd, f"blocks.{i}.{j}."))
                j += 1
            self.blocks.append(nn.ModuleList(res))
            i += 1
        self.up_w, self.up_b = nn.ParameterList(), nn.ParameterList()
        k = 0
        while f"up_w.{k}" in sd:
            self.up_w.append(_p(sd[f"up_w.{k}"]))
            self.up_b.append(_p(sd[f"up_b.{k}"]))
            k += 1
        self.out_g = _p(sd["out_g"])
        self.out_w, self.out_b = _p(sd["out_w"]), _p(sd["out_b"])

    @classmethod
    def from_diffusers(cls, vae: Any) -> "Wan21Decoder":
        """Build the decoder from a Diffusers VAE.

        Args:
            vae: An ``AutoencoderKLQwenImage`` or ``AutoencoderKLWan`` in the 2.1 layout, without
                patchify. Its weights are copied.

        Returns:
            The decoder, with the device and dtype of ``vae``.

        Raises:
            ValueError: If the VAE uses the 2.2 residual layout or patchified input.
        """
        _check_layout(vae)
        dec, mid = vae.decoder, vae.decoder.mid_block
        sd = _stats_tensors(vae)
        sd["post_w"] = vae.post_quant_conv.weight[:, :, 0]
        sd["post_b"] = vae.post_quant_conv.bias
        sd["in_w"], sd["in_b"] = _conv2d_weight(dec.conv_in), dec.conv_in.bias
        sd.update(_res_tensors(mid.resnets[0], "mid0."))
        sd.update(_res_tensors(mid.resnets[1], "mid1."))
        names = ("attn_g", "qkv_w", "qkv_b", "proj_w", "proj_b")
        sd.update(_attn_tensors(mid.attentions[0], names))
        k = 0
        for i, ub in enumerate(dec.up_blocks):
            for j, r in enumerate(ub.resnets):
                sd.update(_res_tensors(r, f"blocks.{i}.{j}."))
            if ub.upsamplers is not None:
                conv = ub.upsamplers[0].resample[1]
                sd[f"up_w.{k}"], sd[f"up_b.{k}"] = conv.weight, conv.bias
                k += 1
        sd["out_g"] = dec.norm_out.gamma.reshape(-1)
        sd["out_w"], sd["out_b"] = _conv2d_weight(dec.conv_out), dec.conv_out.bias
        return cls(sd)

    def to_diffusers(self, vae: Any) -> None:
        """Write the decoder weights into a Diffusers VAE of the same architecture.

        Only what single frames use is overwritten, so the result decodes images like this
        module. Its behavior on video changes if the weights were trained here.

        Args:
            vae: The target ``AutoencoderKLQwenImage`` or ``AutoencoderKLWan``, modified in place.
        """
        dec, mid = vae.decoder, vae.decoder.mid_block
        _put3d(vae.post_quant_conv, self.post_w, 0)
        _put(vae.post_quant_conv.bias, self.post_b)
        _put3d(dec.conv_in, self.in_w, 2)
        _put(dec.conv_in.bias, self.in_b)
        self.mid0.to_diffusers(mid.resnets[0])
        self.mid1.to_diffusers(mid.resnets[1])
        attn = mid.attentions[0]
        _put(attn.norm.gamma, self.attn_g)
        _put(attn.to_qkv.weight, self.qkv_w)
        _put(attn.to_qkv.bias, self.qkv_b)
        _put(attn.proj.weight, self.proj_w)
        _put(attn.proj.bias, self.proj_b)
        k = 0
        for ub, block in zip(dec.up_blocks, self.blocks, strict=True):
            for target, res in zip(ub.resnets, cast(nn.ModuleList, block), strict=True):
                cast(_Res, res).to_diffusers(target)
            if ub.upsamplers is not None:
                conv = ub.upsamplers[0].resample[1]
                _put(conv.weight, self.up_w[k])
                _put(conv.bias, self.up_b[k])
                k += 1
        _put(dec.norm_out.gamma, self.out_g)
        _put3d(dec.conv_out, self.out_w, 2)
        _put(dec.conv_out.bias, self.out_b)

    def _subpixel(
        self, w: torch.Tensor, b: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        A = cast(torch.Tensor, self.phase).to(w.dtype)
        w4 = torch.einsum(
            "aij,bkl,ocjl->abocik", A, A, w
        )  # one kernel per output parity
        co, ci = w.shape[:2]
        w4 = w4.reshape(4 * co, ci, 3, 3).contiguous(memory_format=torch.channels_last)
        return w4, b.repeat(4)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode raw latents to images.

        Args:
            z: Raw latents in the weights' dtype, shape ``[B, C, H, W]``. See ``denormalize``.

        Returns:
            Images of shape ``[B, 3, 8 * H, 8 * W]``, nominally in ``[-1, 1]`` but not clamped.

        Raises:
            RuntimeError: If ``z`` is not on a CUDA device.
        """
        z = z.contiguous(memory_format=torch.channels_last)
        x = F.conv2d(z, self.post_w, self.post_b)
        x = F.conv2d(x, self.in_w, self.in_b, padding=1)
        y = _nchw(rms_silu(_nhwc(x), self.mid0.g1))
        c, h = self.mid0.branch(x, y)
        x = c + h
        o = _attention_branch(
            x, self.attn_g, self.qkv_w, self.qkv_b, self.proj_w, self.proj_b
        )
        s, y = add_rms_silu(
            _nhwc(o), _nhwc(x), self.mid1.g1
        )  # residual and norm in one pass
        x, y = _nchw(s), _nchw(y)
        c, h = self.mid1.branch(x, y)
        stages = [
            (cast(_Res, res), i)
            for i, block in enumerate(self.blocks)
            for res in cast(nn.ModuleList, block)
        ]
        upsample = 0
        for k, (res, i) in enumerate(stages):
            if k == 0:
                s, y = add_rms_silu(_nhwc(c), _nhwc(h), res.g1)
                x, y = _nchw(s), _nchw(y)
            c, h = res.branch(x, y)
            if k + 1 == len(stages):
                s, y = add_rms_silu(_nhwc(c), _nhwc(h), self.out_g)
            elif stages[k + 1][1] == i:
                s, y = add_rms_silu(_nhwc(c), _nhwc(h), stages[k + 1][0].g1)
            else:
                w4, b4 = self._subpixel(self.up_w[upsample], self.up_b[upsample])
                upsample += 1
                u = F.conv2d(c + h, w4, b4, padding=1)  # upsample and conv in one pass
                s, y = shuffle_rms_silu(_nhwc(u), stages[k + 1][0].g1)
            x, y = _nchw(s), _nchw(y)
        return F.conv2d(y, self.out_w, self.out_b, padding=1)

    def _apply(self, fn, recurse=True):  # type: ignore
        self.graphs.clear()
        return super()._apply(fn, recurse)

    def _decode(self, x: torch.Tensor, layout: str) -> torch.Tensor:
        z = denormalize(to_bchw(x, layout), self.mean, self.std)
        return self(z.to(self.in_w.dtype)).clamp(-1, 1)

    @torch.no_grad()
    def decode(
        self, x: torch.Tensor, layout: str = "BHWC", graph: bool = False
    ) -> torch.Tensor:
        """Decode a diffusion model's latents to images.

        Args:
            x: Normalized latents in ``layout``, any floating dtype.
            layout: Axis order of ``x``: ``"BHWC"``, ``"BCHW"``, or ``"BCTHW"`` with one frame.
            graph: Replay the decode as a CUDA graph. The first call for each shape, dtype and
                layout captures a graph, at the cost of about three extra decodes, and later
                calls reuse it. Changes to the weights are not tracked, so clear ``graphs``
                after changing them.

        Returns:
            Images of shape ``[B, 3, 8 * H, 8 * W]``, clamped to ``[-1, 1]``, in the weights'
            dtype.

        Raises:
            RuntimeError: If ``x`` is not on a CUDA device.
            ValueError: If ``layout`` is unknown or ``x`` has more than one frame.
        """
        if not graph:
            return self._decode(x, layout)
        key = (tuple(x.shape), x.dtype, layout)
        runner = self.graphs.get(key)
        if runner is None:
            runner = Graphed(lambda t: self._decode(t, layout), (x,))
            self.graphs[key] = runner
        return runner(x)


class Wan21Encoder(_Half):
    """Encoder of the Wan 2.1 VAE, which Qwen-Image also uses, for single images.

    It returns the mean of the latent distribution, not a sample from it. Its weights start
    frozen.

    Args:
        sd: Tensors as returned by ``state_dict()``, including ``mean`` and ``std``. They are
            copied.

    Attributes:
        mean: Per-channel mean of the raw latents, float32.
        std: Per-channel standard deviation of the raw latents, float32.
        scale_factor: Image pixels per latent cell along each side.

    Raises:
        KeyError: If ``sd`` lacks a tensor the encoder needs.
    """

    def __init__(self, sd: dict[str, torch.Tensor]) -> None:
        super().__init__(sd)
        self.in_w, self.in_b = _p(sd["in_w"]), _p(sd["in_b"])
        self.layers = nn.ModuleList()
        i = 0
        while True:
            prefix = f"layers.{i}."
            if prefix + "g1" in sd:
                self.layers.append(_Res(sd, prefix))
            elif prefix + "qkv_w" in sd:
                self.layers.append(_Attn(sd, prefix))
            elif prefix + "w" in sd:
                self.layers.append(_Down(sd, prefix))
            else:
                break
            i += 1
        self.out_g = _p(sd["out_g"])
        self.out_w, self.out_b = _p(sd["out_w"]), _p(sd["out_b"])
        self.requires_grad_(False)

    @classmethod
    def from_diffusers(cls, vae: Any) -> "Wan21Encoder":
        """Build the encoder from a Diffusers VAE.

        Args:
            vae: An ``AutoencoderKLQwenImage`` or ``AutoencoderKLWan`` in the 2.1 layout, without
                patchify. Its weights are copied.

        Returns:
            The encoder, with the device and dtype of ``vae``.

        Raises:
            ValueError: If the VAE uses the 2.2 residual layout, patchified input, or an unknown
                encoder layer.
        """
        _check_layout(vae)
        enc, mid = vae.encoder, vae.encoder.mid_block
        sd = _stats_tensors(vae)
        sd["in_w"], sd["in_b"] = _conv2d_weight(enc.conv_in), enc.conv_in.bias
        names = ("g", "qkv_w", "qkv_b", "proj_w", "proj_b")
        chain = [*enc.down_blocks, mid.resnets[0], mid.attentions[0], mid.resnets[1]]
        for i, layer in enumerate(chain):
            prefix = f"layers.{i}."
            if hasattr(layer, "conv1"):
                sd.update(_res_tensors(layer, prefix))
            elif hasattr(layer, "resample"):
                conv = layer.resample[1]
                sd[prefix + "w"], sd[prefix + "b"] = conv.weight, conv.bias
            elif hasattr(layer, "to_qkv"):
                sd.update(_attn_tensors(layer, tuple(prefix + n for n in names)))
            else:
                raise ValueError(f"unsupported encoder layer {type(layer).__name__}")
        sd["out_g"] = enc.norm_out.gamma.reshape(-1)
        z = len(vae.config.latents_mean)
        wq = vae.quant_conv.weight[:z, :, 0, 0, 0]  # only the mean half is needed
        wc = _conv2d_weight(enc.conv_out)
        sd["out_w"] = torch.einsum(
            "oc,cikl->oikl", wq, wc
        )  # fold the 1x1 conv into the 3x3
        sd["out_b"] = wq @ enc.conv_out.bias + vae.quant_conv.bias[:z]
        return cls(sd)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Encode images to raw latents.

        Args:
            img: Images in ``[-1, 1]`` in the weights' dtype, shape ``[B, 3, H, W]`` with ``H``
                and ``W`` multiples of 8.

        Returns:
            Raw latent means, shape ``[B, C, H / 8, W / 8]``. See ``normalize``.

        Raises:
            RuntimeError: If ``img`` is not on a CUDA device.
        """
        x = img.contiguous(memory_format=torch.channels_last)
        x = F.conv2d(x, self.in_w, self.in_b, padding=1)
        y = None
        n = len(self.layers)
        for k, layer in enumerate(self.layers):
            nxt = self.layers[k + 1] if k + 1 < n else None
            if isinstance(layer, _Attn) and isinstance(nxt, _Res):
                s, y = add_rms_silu(_nhwc(layer.branch(x)), _nhwc(x), nxt.g1)
                x, y = _nchw(s), _nchw(y)
                continue
            if not isinstance(layer, _Res):
                x, y = layer(x), None
                continue
            if y is None:
                y = _nchw(rms_silu(_nhwc(x), layer.g1))
            c, h = layer.branch(x, y)
            if nxt is not None and not isinstance(nxt, _Res):
                x, y = c + h, None
                continue
            gamma = self.out_g if nxt is None else nxt.g1
            s, y = add_rms_silu(_nhwc(c), _nhwc(h), gamma)
            x, y = _nchw(s), _nchw(y)
        assert y is not None
        return F.conv2d(y, self.out_w, self.out_b, padding=1)

    @torch.no_grad()
    def encode(self, img: torch.Tensor, layout: str = "BHWC") -> torch.Tensor:
        """Encode images to a diffusion model's latents. The inverse of ``Wan21Decoder.decode``.

        Args:
            img: Images in ``[-1, 1]``, any floating dtype, shape ``[B, 3, H, W]`` with ``H`` and
                ``W`` multiples of 8.
            layout: Axis order of the result: ``"BHWC"``, ``"BCHW"``, or ``"BCTHW"``.

        Returns:
            Normalized latents in float32, in ``layout``, with spatial size ``H / 8`` by
            ``W / 8``.

        Raises:
            RuntimeError: If ``img`` is not on a CUDA device.
            ValueError: If ``layout`` is unknown.
        """
        z = normalize(self(img.to(self.in_w.dtype)), self.mean, self.std)
        return from_bchw(z, layout)


def save_vae(
    path: str | os.PathLike,
    decoder: Wan21Decoder | None = None,
    encoder: Wan21Encoder | None = None,
) -> None:
    """Write a VAE file holding a decoder, an encoder, or both.

    Args:
        path: Output path.
        decoder: The decoder to store, if any.
        encoder: The encoder to store, if any.

    Raises:
        ValueError: If neither is given.
    """
    if decoder is None and encoder is None:
        raise ValueError("nothing to save: pass a decoder, an encoder, or both")
    tensors = {}
    for part, module in (("decoder", decoder), ("encoder", encoder)):
        if module is not None:
            for k, v in module.state_dict().items():
                tensors[f"{part}.{k}"] = v.detach().contiguous()
    save_file(tensors, path, metadata={"format": FORMAT, "architecture": ARCHITECTURE})


def _read(path: str | os.PathLike, part: str) -> dict[str, torch.Tensor]:
    with safe_open(path, framework="pt", device="cpu") as f:
        meta = f.metadata() or {}
        prefix = part + "."
        if meta.get("format") != FORMAT:
            prefix = _LEGACY_PREFIX if part == "decoder" else "\0"
        elif meta.get("architecture") != ARCHITECTURE:
            got = meta.get("architecture")
            raise ValueError(f"{path} holds a {got!r} VAE, not {ARCHITECTURE!r}")
        keys = [k for k in f.keys() if k.startswith(prefix)]  # noqa: SIM118
        sd = {k[len(prefix) :]: f.get_tensor(k) for k in keys}
    if not sd and meta.get("format") != FORMAT:
        raise ValueError(
            f"{path} is not a {FORMAT} file (format: {meta.get('format')!r})"
        )
    if not sd:
        raise ValueError(f"{path} holds no {part}")
    return sd


def load_decoder(
    path: str | os.PathLike,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> Wan21Decoder:
    """Load the decoder from a VAE file, or from an older checkpoint that embeds one.

    Args:
        path: A file written by ``save_vae``, or an older model checkpoint that embeds a decoder.
        device: Device to load the weights onto.
        dtype: Dtype of the decoder weights.

    Returns:
        The decoder in eval mode with frozen weights.

    Raises:
        ValueError: If the file is not a VAE file for this architecture or holds no decoder.
    """
    decoder = Wan21Decoder(_read(path, "decoder")).requires_grad_(False)
    return decoder.to(device, dtype).eval()


def load_encoder(
    path: str | os.PathLike,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> Wan21Encoder:
    """Load the encoder from a VAE file.

    Args:
        path: A file written by ``save_vae``.
        device: Device to load the weights onto.
        dtype: Dtype of the encoder weights.

    Returns:
        The encoder in eval mode with frozen weights.

    Raises:
        ValueError: If the file is not a VAE file for this architecture or holds no encoder.
    """
    return Wan21Encoder(_read(path, "encoder")).to(device, dtype).eval()
