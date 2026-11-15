import pytest
import torch
import torch.nn.functional as F
from helpers import rel, tiny_model
from safetensors.torch import save_file

from pneumam.adapters.lllite import (
    Conditioning,
    LLLiteLinear,
    Module,
    attach_lllite,
    detach_lllite,
    load_lllite,
)
from pneumam.models.anima import fused
from pneumam.runtime import Runtime

TARGETS = (
    "self_attn_q_proj",
    "self_attn_k_proj",
    "self_attn_v_proj",
    "cross_attn_q_proj",
    "mlp_layer1",
)


def lllite_file(path, in_ch=3, targets=TARGETS, layers=(0, 2), dim=128, mlp=16, emb=8):
    torch.manual_seed(3)
    sd = {
        "lllite_conditioning1." + k: v
        for k, v in Conditioning(in_ch, 16, emb, 1, ()).state_dict().items()
    }
    for i in layers:
        for t in targets:
            mod = Module(dim, emb, mlp)
            with torch.no_grad():
                mod.up.weight.normal_(std=0.2)
                mod.up.bias.normal_(std=0.05)
                mod.depth_embed.normal_()
            for k, v in mod.state_dict().items():
                sd[f"lllite_dit_blocks_{i}_{t}.{k}"] = v
    meta = {
        "lllite.version": "2",
        "lllite.inpaint_masked_input": "true" if in_ch == 4 else "false",
    }
    save_file({k: v.contiguous() for k, v in sd.items()}, path, metadata=meta)
    return path


@pytest.fixture
def controlled(tmp_path):
    m = tiny_model()
    lllite = load_lllite(lllite_file(tmp_path / "l.safetensors"), "cpu", torch.float32)
    control = attach_lllite(m, lllite)
    control.set_strength(0.8)
    control.set_cond(lllite.encode(torch.rand(1, 3, 64, 64)))
    return m, control


def test_loads_and_encodes_one_embedding_per_token(tmp_path):
    lllite = load_lllite(lllite_file(tmp_path / "l.safetensors"), "cpu", torch.float32)
    assert lllite.encode(torch.rand(2, 3, 64, 96)).shape == (2, 4 * 6, 8)
    assert sorted(lllite.targets) == [0, 2]


def test_inpainting_models_need_a_mask(tmp_path):
    lllite = load_lllite(
        lllite_file(tmp_path / "l.safetensors", in_ch=4), "cpu", torch.float32
    )
    with pytest.raises(ValueError, match="mask"):
        lllite.encode(torch.rand(1, 3, 64, 64))
    assert lllite.encode(torch.rand(1, 3, 64, 64), torch.ones(1, 1, 64, 64)).shape == (
        1,
        16,
        8,
    )


def test_attaches_to_the_fused_projections(controlled):
    m, _ = controlled
    for i in (0, 2):
        for name in ("qkv", "xq", "up"):
            assert isinstance(getattr(m.blocks[i], name), LLLiteLinear)
    assert type(m.blocks[1].qkv) is torch.nn.Linear


def test_fast_path_matches_the_reference_with_a_guided_batch(controlled):
    m, _ = controlled
    layer = m.blocks[0].qkv
    x = torch.randn(2, 16, 128)
    want = layer(x)  # autograd is on in tests, so this is the plain path
    with torch.no_grad():
        got = layer(x)
    plain = x @ layer.base.weight.T
    assert rel(want, plain) > 0.05  # the control changes the output
    assert rel(got, want) < 1e-5


def test_reference_matches_adding_the_residual_to_the_input(controlled):
    m, control = controlled
    layer = m.blocks[2].xq
    x = torch.randn(1, 16, 128)
    mod = layer.adapters["0"]
    r = mod.up(mod.hidden(x, control.cond)) * 0.8
    assert rel(layer(x), (x + r) @ layer.base.weight.T) < 1e-5


def test_fused_mlp_matches_the_separate_layers(controlled):
    m, _ = controlled
    b = m.blocks[0]
    x = torch.randn(1, 16, 128)
    want = b.down(F.gelu(b.up(x)))
    with torch.no_grad():
        got = b.up.fused_mlp(b.down, x)
    assert got is not None and rel(got, want) < 1e-5
    assert b.up.fused_mlp(b.down, x) is None  # autograd on


def test_strength_zero_and_refilled_control(controlled):
    m, control = controlled
    layer = m.blocks[0].qkv
    x = torch.randn(1, 16, 128)
    control.set_strength(0.0)
    with torch.no_grad():
        assert rel(layer(x), x @ layer.base.weight.T) < 1e-6
    control.set_strength(1.0)
    buffer = layer._fast["gamma"]
    assert (
        control.set_cond(torch.randn(1, 16, 8)) is False
    )  # same shape refills in place
    assert layer._fast["gamma"] is buffer
    with torch.no_grad():
        got = layer(x)
    assert rel(got, layer(x)) < 1e-5


def test_model_forward_uses_the_control_and_detach_restores_it(controlled):
    m, _ = controlled
    rt = Runtime()
    torch.manual_seed(1)
    x, t, text = (
        torch.randn(2, 8, 8, 4),
        torch.tensor([0.3, 0.7]),
        torch.randn(2, 6, 32),
    )
    with torch.no_grad():
        on = fused.forward(m, rt, x, t, m.encode_context(rt, text))
        detach_lllite(m)
        off = fused.forward(m, rt, x, t, m.encode_context(rt, text))
    assert rel(on, off) > 1e-3
    assert all(
        type(getattr(b, n)) is torch.nn.Linear
        for b in m.blocks
        for n in ("qkv", "xq", "up")
    )
