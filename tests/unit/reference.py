import pytest
import torch
import torch.nn.functional as F
from helpers import maxerr, rel

from pneumam.backends import _fp8
from pneumam.backends import reference as R

B, S, D, H, hd = 2, 12, 64, 4, 16


@pytest.fixture
def x():
    return torch.randn(B, S, D)


@pytest.fixture
def mod():
    return torch.randn(3, B, D).unbind(0)


def test_adaln_is_a_modulated_layer_norm(x, mod):
    sh, sc, _ = mod
    want = F.layer_norm(x, (D,), eps=1e-6) * sc[:, None] + sh[:, None]
    assert maxerr(R.adaln(x, sh, sc), want) < 1e-5


def test_residual_adaln(x, mod):
    sh, sc, g = mod
    z = torch.randn(B, S, D)
    h, out = R.residual_adaln(x, z, g, sh, sc)
    assert maxerr(out, z + g[:, None] * x) < 1e-5
    assert maxerr(h, R.adaln(z + g[:, None] * x, sh, sc)) < 1e-5


def test_head_norm_is_rms_norm_per_head(x):
    w = torch.rand(hd) + 0.5
    want = F.rms_norm(x.view(B, S, H, hd), (hd,), w, 1e-5)
    assert maxerr(R.head_norm(x, w, H), want) < 1e-5


def test_rms_silu(x):
    g = torch.rand(D) + 0.5
    want = F.silu(F.normalize(x, dim=-1) * D**0.5 * g)
    assert maxerr(R.rms_silu(x, g), want) < 1e-5


def test_add_rms_silu(x):
    g, r = torch.rand(D) + 0.5, torch.randn(B, S, D)
    s, y = R.add_rms_silu(x, r, g)
    assert maxerr(s, x + r) < 1e-6
    assert maxerr(y, R.rms_silu(x + r, g)) < 1e-5


def test_shuffle_rms_silu_is_a_pixel_shuffle():
    C = 8
    u, g = torch.randn(B, 3, 5, 4 * C), torch.rand(C) + 0.5
    s, y = R.shuffle_rms_silu(u, g)
    nchw = u.view(B, 3, 5, 4, C).permute(0, 4, 3, 1, 2).reshape(B, 4 * C, 3, 5)
    want = F.pixel_shuffle(nchw, 2).permute(0, 2, 3, 1)
    assert maxerr(s, want) < 1e-6
    assert maxerr(y, R.rms_silu(want, g)) < 1e-5


def test_qkv_norm_rope_rotates_normalized_q_and_leaves_v():
    w = torch.rand(hd) + 0.5
    qkv, ang = torch.randn(B, S, 3 * H * hd), torch.rand(S, hd // 2) * 6
    q, k, v = R.qkv_norm_rope(qkv, w, w, ang.cos(), ang.sin(), H)
    qn = R.head_norm(qkv[..., : H * hd].contiguous(), w, H)
    rot = torch.polar(torch.ones_like(ang), ang)[None, :, None]
    c = torch.complex(qn[..., : hd // 2], qn[..., hd // 2 :]) * rot
    assert maxerr(q, torch.cat((c.real, c.imag), -1)) < 1e-5
    assert maxerr(q.norm(dim=-1), qn.norm(dim=-1)) < 1e-4
    assert torch.equal(v, qkv.view(B, S, 3, H, hd)[:, :, 2])
    assert k.shape == (B, S, H, hd)


def test_qkv_norm_rope_fp8_with_a_safe_scale_never_clips():
    w = torch.rand(hd) + 0.5
    qkv, ang = torch.randn(B, S, 3 * H * hd), torch.rand(S, hd // 2) * 6
    s = _fp8.norm_scale(w)
    q, k, v = R.qkv_norm_rope(qkv, w, w, ang.cos(), ang.sin(), H)
    q8, k8, v8 = R.qkv_norm_rope_fp8(qkv, w, w, ang.cos(), ang.sin(), H, s, s)
    assert q8.dtype == k8.dtype == v8.dtype == torch.float8_e4m3fn
    assert q8.float().abs().max() < _fp8.FP8_MAX
    assert rel(q8.float() / s, q) < 0.05
    assert rel(k8.float() / s, k) < 0.05
    assert rel(v8.float(), v) < 0.05


def test_norm_scale_is_a_power_of_two_within_the_bound():
    w = torch.rand(hd) + 0.5
    s = _fp8.norm_scale(w)
    bound = _fp8.FP8_MAX / (hd**0.5 * w.abs().max().item())
    assert s <= bound < 2 * s
    assert torch.log2(torch.tensor(s)).frac() == 0


def test_trimmed_attention_matches_attention_over_zero_padding():
    Sk, full = 8, 20
    q = torch.randn(B, S, H, hd)
    used = torch.tensor([5, 8], dtype=torch.int32)
    keep = (torch.arange(Sk)[None] < used[:, None])[:, :, None, None]
    k, v = torch.randn(B, Sk, H, hd) * keep, torch.randn(B, Sk, H, hd) * keep
    pad = (0, 0, 0, 0, 0, full - Sk)
    out, lse = R.attention_trimmed(q, k, v, used)
    got = R.ctx_scale(out, lse, torch.log((full - used).float()))
    assert maxerr(got, R.attention(q, F.pad(k, pad), F.pad(v, pad))) < 1e-5


def test_biased_trimmed_attention_matches_attention_over_zero_padding():
    Sk, full = 8, 20
    q = torch.randn(B, S, H, hd)
    used = torch.tensor([5, 8], dtype=torch.int32)
    keep = (torch.arange(Sk)[None] < used[:, None])[:, :, None, None]
    k, v = torch.randn(B, Sk, H, hd) * keep, torch.randn(B, Sk, H, hd) * keep
    pad = (0, 0, 0, 0, 0, full - Sk)
    want = R.attention(q, F.pad(k, pad), F.pad(v, pad))
    ln_pad = torch.log((full - used).float())
    assert maxerr(R.attention_trimmed_biased(q, k, v, used, ln_pad), want) < 1e-5
    junk = torch.randn(B, Sk, H, hd)  # keys past the valid count must not matter
    got = R.attention_trimmed_biased(q, torch.where(keep, k, junk), v, used, ln_pad)
    assert maxerr(got, want) < 1e-5
    whole = torch.tensor([Sk, Sk], dtype=torch.int32)
    k, v = torch.randn(2, B, Sk, H, hd).unbind(0)
    got = R.attention_trimmed_biased(q, k, v, whole, torch.log(torch.zeros(B)))
    assert maxerr(got, R.attention(q, k, v)) < 1e-5
    half = R.attention_trimmed_biased(
        q.bfloat16(), k.bfloat16(), v.bfloat16(), whole, ln_pad
    )
    assert half.dtype == torch.bfloat16 and half.isfinite().all()


def test_ctx_scale_without_padding_is_the_identity():
    q, k, v = torch.randn(3, B, S, H, hd).unbind(0)
    out, lse = R.attention_trimmed(q, k, v, torch.tensor([S, S], dtype=torch.int32))
    got = R.ctx_scale(out, lse, torch.log(torch.zeros(B)))
    assert maxerr(got, R.attention(q, k, v)) < 1e-5


def test_mlp_and_linear_gelu(x):
    w1, w2 = torch.randn(128, D), torch.randn(D, 128)
    assert maxerr(R.linear_gelu(x, w1), F.gelu(x @ w1.T)) < 1e-4
    assert maxerr(R.mlp(x, w1, w2), F.gelu(x @ w1.T) @ w2.T) < 1e-3


def test_rowquant_round_trips_within_fp8_precision(x):
    q, s = R.rowquant(x)
    assert q.dtype == torch.float8_e4m3fn
    assert rel(q.float() * s.view(B, S, 1), x) < 0.05
    qg, sg = R.rowquant(x, gelu=True)
    assert rel(qg.float() * sg.view(B, S, 1), F.gelu(x)) < 0.05


def test_fp8_linear_approximates_the_dense_product(x):
    w = torch.randn(32, D)
    q, s = R.rowquant(x)
    assert rel(R.fp8_linear(q, s, _fp8.quantize_weight(w)), x @ w.T) < 0.05


def test_adaln_fp8_approximates_adaln(x, mod):
    sh, sc, g = mod
    q, s = R.adaln_fp8(x, sh, sc)
    assert rel(q.float() * s.view(B, S, 1), R.adaln(x, sh, sc)) < 0.05
    q, s, z = R.residual_adaln_fp8(x, x, g, sh, sc)
    h, want = R.residual_adaln(x, x, g, sh, sc)
    assert rel(q.float() * s.view(B, S, 1), h) < 0.05
    assert maxerr(z, want) < 1e-5


def test_to_fp8_clips_at_the_format_maximum():
    t = _fp8.to_fp8(torch.tensor([1e4, -1e4, 1.0]), 2.0)
    assert t.float().tolist() == [448.0, -448.0, 2.0]
