import pytest
import torch
from helpers import rel, tiny_cosmos

from pneumam import models
from pneumam.models.anima import fused
from pneumam.models.anima.convert import (
    config_from_diffusers,
    convert_checkpoint,
    convert_lora,
    from_diffusers,
    merge_lora,
    to_diffusers,
)
from pneumam.runtime import Runtime

pytest.importorskip("diffusers")


@pytest.fixture(scope="module")
def cosmos():
    return tiny_cosmos()


@pytest.fixture
def sd(cosmos):
    return {k: v.detach() for k, v in cosmos.state_dict().items()}


@pytest.fixture
def cfg(cosmos, sd):
    return config_from_diffusers(sd, dict(cosmos.config))


def test_config_is_read_from_the_checkpoint(cfg):
    assert (cfg.layers, cfg.heads, cfg.head_dim) == (3, 2, 64)
    assert (cfg.text_dim, cfg.lora_dim, cfg.mlp_dim, cfg.in_ch) == (32, 8, 256, 4)
    assert cfg.rope_scale == (4.0, 4.0)


def test_weights_round_trip(sd, cfg):
    back = to_diffusers(from_diffusers(sd, cfg), cfg)
    patch = "patch_embed.proj.weight"
    assert set(back) == set(sd)
    assert all(torch.equal(sd[k], back[k]) for k in sd if k != patch)
    C = cfg.in_ch
    a, b = sd[patch].view(-1, C + 1, 2, 2), back[patch].view(-1, C + 1, 2, 2)
    assert torch.equal(a[:, :C], b[:, :C])  # the padding-mask channel is not kept


def test_converted_checkpoint_matches_the_diffusers_forward(cosmos, cfg, tmp_path):
    src, dst = tmp_path / "diffusers", tmp_path / "anima.safetensors"
    cosmos.save_pretrained(src)
    assert convert_checkpoint(src, dst) == cfg
    m = models.load(dst, "cpu", torch.float32)
    B, C, H, W, S = 2, cfg.in_ch, 12, 16, 10
    x, t = torch.randn(B, C, 1, H, W), torch.tensor([0.25, 0.8])
    text = torch.zeros(B, S, cfg.text_dim)
    text[:, :4] = torch.randn(B, 4, cfg.text_dim)
    with torch.no_grad():
        ref = cosmos(
            hidden_states=x,
            timestep=t,
            encoder_hidden_states=text,
            padding_mask=torch.zeros(1, 1, H * 8, W * 8),
            return_dict=False,
        )[0]
    rt = Runtime()
    ours = x[:, :, 0].permute(0, 2, 3, 1).contiguous()
    out = fused.forward(m, rt, ours, t, m.encode_context(rt, text))
    assert rel(out, ref[:, :, 0].permute(0, 2, 3, 1)) < 1e-5


def test_converted_lora_merges_like_each_projection_s_own_delta(sd, cfg):
    flat = from_diffusers(sd, cfg)
    r = 4
    lora, want = {}, {k: v.clone() for k, v in flat.items()}
    cases = (
        ("attn1.to_q", 8.0),
        ("attn1.to_v", None),
        ("attn2.to_k", 2.0),
        ("ff.net.2", None),
    )
    for name, alpha in cases:
        stem = f"transformer_blocks.1.{name}"
        base = sd[f"{stem}.weight"]
        a, b = torch.randn(r, base.shape[1]), torch.randn(base.shape[0], r)
        lora[f"transformer.{stem}.lora_A.weight"] = a
        lora[f"transformer.{stem}.lora_B.weight"] = b
        if alpha is not None:
            lora[f"transformer.{stem}.alpha"] = torch.tensor(alpha)
        changed = dict(sd)
        changed[f"{stem}.weight"] = base + (alpha or r) / r * (b @ a)
        for k, v in from_diffusers(changed, cfg).items():
            want[k] = want[k] + (v - flat[k])
    lora["unrelated.key"] = torch.zeros(1)
    conv, skipped = convert_lora(lora)
    assert skipped == ["unrelated.key"]
    assert {k.split(".", 2)[2] for k in conv} == {
        f"{t}.lora_{ab}" for t in ("qkv", "xkv", "down") for ab in "AB"
    }
    merged = merge_lora(flat, conv)
    assert max((merged[k] - want[k]).abs().max().item() for k in flat) < 1e-5
    half = merge_lora(flat, conv, 0.5)
    k = "blocks.1.qkv.weight"
    assert torch.allclose(half[k] - flat[k], (merged[k] - flat[k]) / 2, atol=1e-6)
    assert merged is not flat and torch.equal(flat[k], from_diffusers(sd, cfg)[k])
