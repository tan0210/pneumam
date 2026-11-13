import pytest
import torch
from helpers import padded_text, rel, tiny_model

from pneumam.models.anima import fused
from pneumam.runtime import Runtime

B, H, W, S = 2, 8, 12, 16


@pytest.fixture
def inputs():
    x = torch.randn(B, H, W, 4)
    return x, torch.tensor([0.3, 0.8]), padded_text(B, S, 32, (5, 11))


def run(m, rt, inputs, dtype=torch.float32):
    x, t, text = inputs
    return fused.forward(m, rt, x.to(dtype), t, m.encode_context(rt, text.to(dtype)))


def test_spec_describes_the_model():
    spec = tiny_model().spec
    assert spec.architecture == "anima"
    assert spec.latent_channels == 4 and spec.patch_size == 2
    assert spec.layout == "BHWC"
    assert spec.lora_targets["qkv"] == 3


def test_forward_returns_a_velocity_shaped_like_the_latent(inputs):
    m = tiny_model().to(torch.bfloat16)
    out = run(m, Runtime(), inputs, torch.bfloat16)
    assert out.shape == (B, H, W, 4) and out.dtype == torch.bfloat16
    assert out.isfinite().all()


def test_bf16_tracks_float32(inputs):
    m = tiny_model()
    want = run(m, Runtime(), inputs)
    got = run(m.to(torch.bfloat16), Runtime(), inputs, torch.bfloat16)
    assert rel(got, want) < 0.05


def test_samples_do_not_depend_on_the_rest_of_the_batch(inputs):
    m, (x, t, text) = tiny_model(), inputs
    both = run(m, Runtime(), inputs)
    one = run(m, Runtime(), (x[:1], t[:1], text[:1]))
    assert rel(one, both[:1]) < 1e-5


def test_context_trimming_does_not_change_the_result(inputs):
    m = tiny_model()
    want = run(m, Runtime(trim_context=False), inputs)
    assert rel(run(m, Runtime(trim_context=True), inputs), want) < 1e-5


def test_fp8_tracks_bf16(inputs):
    m, rt = tiny_model().to(torch.bfloat16), Runtime()
    want = run(m, rt, inputs, torch.bfloat16)
    fused.enable_fp8(m, rt)
    assert rt.fp8 is not None
    assert rel(run(m, rt, inputs, torch.bfloat16), want) < 0.1
    fused.disable_fp8(rt)
    assert torch.equal(run(m, rt, inputs, torch.bfloat16), want)


def test_experimental_fp8_attention(inputs, monkeypatch):
    m, rt = tiny_model().to(torch.bfloat16), Runtime()
    x, t, text = (a.to(torch.bfloat16) if a.ndim > 1 else a for a in inputs)
    want = run(m, rt, inputs, torch.bfloat16)
    fused.enable_fp8(m, rt)
    plain = m.encode_context(rt, text)
    assert plain.kv8 is None
    monkeypatch.setenv("PNEUMAM_EXPERIMENTAL_FP8_ATTENTION", "1")
    quantized = m.encode_context(rt, text)
    assert quantized.kv8 is not None
    assert rel(fused.forward(m, rt, x, t, quantized), want) < 0.15
    assert rel(fused.forward(m, rt, x, t, plain), want) < 0.15  # falls back per context


def test_block_checkpointing_does_not_change_gradients(inputs):
    m = tiny_model().requires_grad_(True)
    grads = []
    for ck in (True, False):
        m.zero_grad()
        run(m, Runtime(checkpointing=ck), inputs).square().sum().backward()
        grads.append(m.blocks[1].qkv.weight.grad.clone())
    assert grads[0].abs().sum() > 0
    assert rel(grads[0], grads[1]) < 1e-5


def test_rope_tables_are_cached_per_size(inputs):
    m, rt = tiny_model(), Runtime()
    run(m, rt, inputs)
    run(m, rt, inputs)
    assert len(rt.tables) == 1


def test_patchify_round_trip():
    m = tiny_model()
    x = torch.randn(B, H, W, 4)
    n = (H // 2) * (W // 2)
    assert m.patchify(x).shape == (B, n, m.cfg.dim)
    y = torch.randn(B, n, 16)
    out = m.unpatchify(y, H, W)
    assert out.shape == (B, H, W, 4)
    first = out[0, :2, :2].flatten().tolist()  # the first token's 2x2 patch
    assert sorted(first) == sorted(y[0, 0].tolist())
