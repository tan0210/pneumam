import types

import torch

from pneumam.models.anima.config import AnimaConfig
from pneumam.models.anima.dit import AnimaDiT


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm()).item()


def maxerr(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def tiny_config(**overrides) -> AnimaConfig:
    base = {
        "layers": 3,
        "heads": 2,
        "head_dim": 64,
        "text_dim": 32,
        "lora_dim": 8,
        "mlp_dim": 256,
        "in_ch": 4,
    }
    return AnimaConfig(**(base | overrides))


def tiny_model(cfg: AnimaConfig | None = None, seed: int = 0) -> AnimaDiT:
    torch.manual_seed(seed)
    m = AnimaDiT(cfg or tiny_config())
    for name, p in m.named_parameters():
        if name.endswith("_norm"):
            torch.nn.init.uniform_(p, 0.7, 1.3)
        else:
            torch.nn.init.normal_(p, std=0.5 / p.shape[-1] ** 0.5)
    return m.requires_grad_(False).eval()


def padded_text(
    batch: int, length: int, dim: int, used: tuple[int, ...]
) -> torch.Tensor:
    text = torch.zeros(batch, length, dim)
    for b, n in enumerate(used):
        text[b, :n] = torch.randn(n, dim)
    return text


def tiny_cosmos():
    import diffusers.models.transformers.transformer_cosmos as tc

    if not hasattr(getattr(tc, "transforms", None), "functional"):
        # without torchvision; the padding mask is all zeros, so any resize will do
        tc.transforms = types.SimpleNamespace(
            functional=types.SimpleNamespace(
                resize=lambda m, size, **kw: torch.nn.functional.interpolate(
                    m, size=tuple(size)
                )
            ),
            InterpolationMode=types.SimpleNamespace(NEAREST=None),
        )
    torch.manual_seed(0)
    tr = tc.CosmosTransformer3DModel(
        in_channels=4,
        out_channels=4,
        num_attention_heads=2,
        attention_head_dim=64,
        num_layers=3,
        mlp_ratio=2.0,
        text_embed_dim=32,
        adaln_lora_dim=8,
        max_size=(8, 32, 32),
        rope_scale=(1.0, 4.0, 4.0),
        extra_pos_embed_type=None,
    ).eval()
    for p in tr.parameters():  # zero-initialised layers would hide errors
        if p.ndim > 1:
            torch.nn.init.normal_(p, std=0.6 / p.shape[-1] ** 0.5)
        else:
            torch.nn.init.uniform_(p, 0.7, 1.3)
    return tr


def tiny_wan_vae(attn_scales: tuple[float, ...] = ()):
    from diffusers import AutoencoderKLWan

    torch.manual_seed(0)
    vae = AutoencoderKLWan(
        base_dim=32,
        z_dim=16,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=list(attn_scales),
        temperal_downsample=[False, True, True],
    )
    with torch.no_grad():
        for name, p in vae.named_parameters():
            if name.endswith("gamma"):
                p.uniform_(0.7, 1.3)
            elif p.ndim == 1:
                p.normal_(std=0.05)
    return vae.eval()
