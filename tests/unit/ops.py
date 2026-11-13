import pytest
import torch
import torch.nn.functional as F
from helpers import maxerr, rel

from pneumam.backends import reference as R
from pneumam.backends import warpy
from pneumam.ops import adaln, attention, gemm, lora, norm, rope

B, S, D, H, hd = 2, 12, 64, 4, 16


@pytest.fixture
def x():
    return torch.randn(B, S, D)


@pytest.fixture
def qkv():
    w, ang = torch.rand(hd) + 0.5, torch.rand(S, hd // 2)
    return rope.qkv_norm_rope(torch.randn(B, S, 3 * D), w, w, ang.cos(), ang.sin(), H)


def test_ops_agree_with_the_reference_backend(x):
    sh, sc, g = torch.randn(3, B, D).unbind(0)
    w = torch.rand(hd) + 0.5
    assert torch.equal(adaln.adaln(x, sh, sc), R.adaln(x, sh, sc))
    assert torch.equal(norm.head_norm(x, w, H), R.head_norm(x, w, H))
    assert torch.equal(norm.rms_silu(x, w.repeat(H)), R.rms_silu(x, w.repeat(H)))
    _, z = adaln.residual_adaln(x, x, g, sh, sc)
    assert maxerr(z, x + g[:, None] * x) < 1e-5


def test_attention_merges_heads(qkv):
    q, k, v = qkv
    t = [a.transpose(1, 2) for a in (q, k, v)]
    want = F.scaled_dot_product_attention(*t).transpose(1, 2).flatten(2)
    out = attention.attention(q, k, v)
    assert out.shape == (B, S, D)
    assert maxerr(out, want) < 1e-5


def test_fp8_attention_falls_back_without_a_kernel(qkv):
    q, k, v = qkv
    want = attention.attention(q, k, v)
    assert torch.equal(attention.attention_fp8(q, k, v, rescale=False), want)
    assert not attention.trims_fast(q) and not attention.trims_fast(q, fp8=True)


def test_quantized_attention_recovers_true_units(qkv):
    q, k, v = qkv
    sq, sk, sv = 4.0, 8.0, 2.0
    q8, k8, v8 = (attention.to_fp8(t, s) for t, s in ((q, sq), (k, sk), (v, sv)))
    out = attention.attention_quantized(q8, k8, v8, sq, sk, sv)
    assert out.dtype == torch.bfloat16
    assert rel(out, attention.attention(q, k, v)) < 0.1


def test_trimmed_cross_attention(qkv):
    q = qkv[0]
    Sk, full = 8, 20
    used = torch.tensor([5, 8], dtype=torch.int32)
    keep = (torch.arange(Sk)[None] < used[:, None])[:, :, None, None]
    k, v = torch.randn(B, Sk, H, hd) * keep, torch.randn(B, Sk, H, hd) * keep
    pad = (0, 0, 0, 0, 0, full - Sk)
    ln_pad = torch.log((full - used).float())
    want = attention.attention(q, F.pad(k, pad), F.pad(v, pad))
    assert maxerr(attention.cross_attention_trimmed(q, k, v, used, ln_pad), want) < 1e-5
    k8, v8 = attention.to_fp8(k, 4.0), attention.to_fp8(v, 4.0)
    q8 = attention.to_fp8(q, 4.0)
    got = attention.cross_attention_trimmed_quantized(
        q8, k8, v8, used, ln_pad, 4.0, 4.0, 4.0
    )
    assert rel(got, want) < 0.1


def test_wide_attention():
    q = torch.randn(B, 1, S, 384)
    want = F.scaled_dot_product_attention(q, q, q)
    assert maxerr(attention.wide_attention(q, q, q), want) < 1e-6


def test_gemm_ops(x):
    w1, w2 = torch.randn(128, D), torch.randn(D, 128)
    assert maxerr(gemm.linear(x, w1), x @ w1.T) < 1e-4
    assert maxerr(gemm.linear_gelu(x, w1), F.gelu(x @ w1.T)) < 1e-4
    assert maxerr(gemm.mlp(x, w1, w2), F.gelu(x @ w1.T) @ w2.T) < 1e-3
    fw = gemm.quantize_weight(w1)
    assert rel(gemm.fp8_linear(*gemm.rowquant(x), fw), x @ w1.T) < 0.05


def test_linear_is_differentiable(x):
    w = torch.randn(32, D, requires_grad=True)
    xg = x.clone().requires_grad_()
    got = torch.autograd.grad(gemm.linear(xg, w).square().sum(), (xg, w))
    want = torch.autograd.grad((xg @ w.T).square().sum(), (xg, w))
    assert all(maxerr(a, b) < 1e-3 for a, b in zip(got, want, strict=True))


@pytest.mark.parametrize("parts", [1, 2, 3])
def test_lora_add_matches_the_dense_formula_and_its_gradients(parts):
    n, i, o, r, scale = 10, 24, 8, 4, 0.5
    x = torch.randn(n, i, requires_grad=True)
    base = torch.randn(n, parts * o, requires_grad=True)
    A = torch.randn(parts * r, i, requires_grad=True)
    Bm = torch.randn(parts, o, r, requires_grad=True)
    out = lora.lora_add(base, x, A, Bm, A.detach(), (Bm * scale).detach(), scale)
    dense = base + scale * x @ (torch.block_diag(*Bm.unbind(0)) @ A).T
    assert maxerr(out, dense) < 1e-4
    cot = torch.randn_like(out)
    got = torch.autograd.grad((out * cot).sum(), (base, x, A, Bm))
    want = torch.autograd.grad((dense * cot).sum(), (base, x, A, Bm))
    assert all(maxerr(a, b) < 1e-4 for a, b in zip(got, want, strict=True))


def test_warpy_pads_rows_to_whole_blocks():
    a = torch.randn(130, 64).to(torch.bfloat16)
    p = warpy._pad_rows(a)
    assert p.shape == (256, 64) and p.dtype == a.dtype
    assert torch.equal(p[:130], a) and not p[130:].any()
    q = R.rowquant(torch.randn(1, 24, 64))[0].reshape(24, 64)
    p8 = warpy._pad_rows(q)
    assert p8.dtype == q.dtype and p8.shape == (128, 64)
    assert torch.equal(p8[:24].float(), q.float())
    full = torch.randn(128, 64).to(torch.bfloat16)
    assert warpy._pad_rows(full).shape == (128, 64)
