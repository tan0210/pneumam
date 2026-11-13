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


def test_trimmed_cross_attention_without_a_trimming_kernel(qkv, monkeypatch):
    q = qkv[0]
    Sk, full = 8, 20
    used = torch.tensor([5, 8], dtype=torch.int32)
    keep = (torch.arange(Sk)[None] < used[:, None])[:, :, None, None]
    k, v = torch.randn(B, Sk, H, hd) * keep, torch.randn(B, Sk, H, hd) * keep
    pad = (0, 0, 0, 0, 0, full - Sk)
    ln_pad = torch.log((full - used).float())
    want = attention.attention(q, F.pad(k, pad), F.pad(v, pad))
    monkeypatch.setattr(attention._dispatch, "arch", lambda device: "sm89")
    monkeypatch.setattr(attention.flash_attn, "fits", lambda q: False)
    got = attention.cross_attention_trimmed(q, k, v, used, ln_pad)
    assert got.shape == want.shape and maxerr(got, want) < 1e-5


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


def test_unscaled_fp8_product_carries_its_scales(x):
    w = torch.randn(128, D)
    fw = gemm.quantize_weight(w)
    q, s = gemm.rowquant(x)
    y, scales = gemm.fp8_linear_unscaled(q, s, fw)
    assert scales is not None
    true = y.float() * scales[0].view(B, S, 1) * scales[1]
    assert rel(true, x @ w.T) < 0.05
    assert rel(gemm.fp8_linear(q, s, fw), true) < 1e-2
    plain = gemm.rowquant(true, gelu=True)
    folded = gemm.rowquant(y, gelu=True, scales=scales)
    assert (
        rel(
            folded[0].float() * folded[1].view(B, S, 1),
            plain[0].float() * plain[1].view(B, S, 1),
        )
        < 1e-2
    )


def test_residual_adaln_fp8_applies_pending_scales(x):
    sh, sc, g = torch.randn(3, B, D).unbind(0)
    rows, cols = torch.rand(B * S, 1) + 0.5, torch.rand(D) + 0.5
    want = adaln.residual_adaln_fp8(x * rows.view(B, S, 1) * cols, x, g, sh, sc)
    got = adaln.residual_adaln_fp8(x, x, g, sh, sc, x_scales=(rows, cols))
    assert maxerr(got[2], want[2]) < 1e-5
    assert (
        rel(
            got[0].float() * got[1].view(B, S, 1),
            want[0].float() * want[1].view(B, S, 1),
        )
        < 1e-2
    )


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


@pytest.mark.parametrize("rows", [128, 300, 100, 384])
def test_warpy_runs_whole_blocks_in_place_and_pads_only_the_tail(rows, monkeypatch):
    launches = []

    def fake_gemm(a, w, out=None, act="", **config):
        assert a.shape[0] % 128 == 0 and a.is_contiguous()
        launches.append((a.shape[0], out is not None))
        y = a.float() @ w.float().T
        if out is None:
            out = torch.empty(a.shape[0], w.shape[0], dtype=a.dtype)
        out.copy_(torch.nn.functional.gelu(y) if act == "gelu" else y)
        return (out, None) if act == "gelu" else out

    monkeypatch.setattr(warpy, "_wk", type("W", (), {"gemm": staticmethod(fake_gemm)}))
    monkeypatch.setattr(warpy, "_key", lambda a, w, act: "k")
    a, w = (
        torch.randn(rows, 64).to(torch.bfloat16),
        torch.randn(128, 64).to(torch.bfloat16),
    )
    for act in ("", "gelu"):
        launches.clear()
        y = a.float() @ w.float().T
        want = torch.nn.functional.gelu(y) if act == "gelu" else y
        got = warpy._run(a, w, act)
        assert got.shape == (rows, 128) and maxerr(got, want.to(torch.bfloat16)) < 1e-6
        whole, tail = rows // 128 * 128, rows % 128
        expect = ([(whole, tail > 0)] if whole else []) + (
            [(128, False)] if tail else []
        )
        assert launches == expect


@pytest.mark.parametrize("parts", [1, 2, 3])
@pytest.mark.parametrize("fp8", ["off", "forward", "both"])
def test_lora_linear_matches_the_dense_formula_and_its_gradients(parts, fp8):
    n, i, o, r, scale = 10, 64, 128, 4, 0.5
    x = torch.randn(n, i, requires_grad=True)
    w = torch.randn(parts * o, i)
    A = torch.randn(parts * r, i, requires_grad=True)
    Bm = torch.randn(parts, o, r, requires_grad=True)
    w8 = gemm.quantize_weight(w) if fp8 != "off" else None
    w8_t = gemm.quantize_weight(w.T.contiguous()) if fp8 == "both" else None
    args = (x, w, A, Bm, A.detach(), (Bm * scale).detach(), scale, w8, w8_t)
    out = lora.lora_linear(*args)
    dense = x @ w.T + scale * x @ (torch.block_diag(*Bm.unbind(0)) @ A).T
    cot = torch.randn_like(out)
    got = torch.autograd.grad((out * cot).sum(), (x, A, Bm))
    want = torch.autograd.grad((dense * cot).sum(), (x, A, Bm))
    tol = 1e-4 if fp8 == "off" else 0.05
    assert rel(out, dense) < tol
    assert rel(got[0], want[0]) < (1e-4 if fp8 != "both" else 0.05)
    assert rel(got[1], want[1]) < 1e-4 and rel(got[2], want[2]) < 1e-4
    assert w.grad is None


def test_warpy_mlp_gradients_with_a_stand_in_kernel(monkeypatch):
    gelu = torch.nn.functional.gelu

    def fake_gemm(
        a, b, out=None, act="", store_pre=False, pre=None, b_trans=False, **config
    ):
        assert a.is_contiguous() and not a.requires_grad and not b.requires_grad
        prod = a.float() @ (b.float() if b_trans else b.float().T)
        if act == "gelu":
            return gelu(prod).to(a.dtype), (prod.to(a.dtype) if store_pre else None)
        if act == "dgelu":
            h = pre.float()
            density = torch.exp(-0.5 * h * h) / (2 * torch.pi) ** 0.5
            slope = 0.5 * (1 + torch.erf(h / 2**0.5)) + h * density
            prod = prod * slope
        if out is not None:
            out.copy_(prod)
            return out
        return prod.to(a.dtype)

    monkeypatch.setattr(warpy, "_wk", type("W", (), {"gemm": staticmethod(fake_gemm)}))
    monkeypatch.setattr(warpy, "_key", lambda a, w, act: "k")
    monkeypatch.setattr(warpy, "available", True)
    monkeypatch.setattr(gemm._dispatch, "arch", lambda device: "sm120")
    x = torch.randn(2, 64, 64, requires_grad=True)
    w1 = torch.randn(128, 64, requires_grad=True)
    w2 = torch.randn(128, 128, requires_grad=True)
    cast = lambda t: t.to(torch.bfloat16)
    assert warpy.fits_mlp_grad(cast(x), cast(w1), cast(w2))
    assert not warpy.fits_mlp_grad(cast(x)[:, :50], cast(w1), cast(w2))
    out = gemm.mlp(cast(x), cast(w1), cast(w2))
    dense = gelu(x @ w1.T) @ w2.T
    assert out.dtype == torch.bfloat16 and rel(out, dense) < 0.02
    cot = torch.randn_like(dense)
    got = torch.autograd.grad((out.float() * cot).sum(), (x, w1, w2))
    want = torch.autograd.grad((dense * cot).sum(), (x, w1, w2))
    assert all(rel(a, b) < 0.03 for a, b in zip(got, want, strict=True))
    frozen = gemm.mlp(cast(x), cast(w1).detach(), cast(w2).detach())
    (dx,) = torch.autograd.grad((frozen.float() * cot).sum(), x)
    assert rel(dx, want[0]) < 0.03


def lora_mlp_case():
    n, i, h, o, r = 128, 128, 256, 128, 4
    x = torch.randn(n, i, requires_grad=True)
    w1, w2 = torch.randn(h, i) * 0.2, torch.randn(o, h) * 0.2
    leaves = [
        torch.randn(s, requires_grad=True)
        for s in ((r, i), (1, h, r), (r, h), (1, o, r))
    ]
    A1, B1, A2, B2 = leaves
    s1, s2 = 0.5, 2.0
    up = (w1, A1, B1, A1.detach(), (B1 * s1).detach(), s1)
    down = (w2, A2, B2, A2.detach(), (B2 * s2).detach(), s2)
    pre = x @ w1.T + s1 * x @ (B1[0] @ A1).T
    post = torch.nn.functional.gelu(pre)
    dense = post @ w2.T + s2 * post @ (B2[0] @ A2).T
    return x, up, down, dense, (x, *leaves)


def test_lora_mlp_matches_the_dense_formula_and_its_gradients():
    x, up, down, dense, leaves = lora_mlp_case()
    out = lora.lora_mlp(x, up, down)
    assert rel(out, dense) < 1e-5
    cot = torch.randn_like(dense)
    got = torch.autograd.grad((out * cot).sum(), leaves)
    want = torch.autograd.grad((dense * cot).sum(), leaves)
    assert all(rel(a, b) < 1e-4 for a, b in zip(got, want, strict=True))


def test_accumulating_products_with_a_stand_in_kernel(monkeypatch):
    gelu = torch.nn.functional.gelu
    calls = []

    def fake_gemm(
        a,
        b,
        out=None,
        act="",
        store_pre=False,
        pre=None,
        b_trans=False,
        accum=False,
        **cfg,
    ):
        assert accum and out is not None and out.is_contiguous()
        assert a.shape[0] % 128 == 0 and not a.requires_grad
        calls.append((act, b_trans))
        prod = a.float() @ (b.float() if b_trans else b.float().T) + out.float()
        if act == "gelu":
            out.copy_(gelu(prod))
            return out, prod.to(a.dtype)
        if act == "dgelu":
            prod = prod * gemm._gelu_slope(pre)
        out.copy_(prod)
        return out

    monkeypatch.setattr(warpy, "_wk", type("W", (), {"gemm": staticmethod(fake_gemm)}))
    monkeypatch.setattr(warpy, "_key", lambda a, w, act: "k")
    monkeypatch.setattr(warpy, "_ACCUM", True)
    monkeypatch.setattr(gemm._dispatch, "arch", lambda device: "sm120")
    x, up, down, dense, leaves = lora_mlp_case()
    bf = lambda t: t.to(torch.bfloat16) if isinstance(t, torch.Tensor) else t
    xb = bf(x)
    out = lora.lora_mlp(
        xb,
        tuple(bf(t) for t in up[:1]) + up[1:3] + tuple(bf(t) for t in up[3:5]) + up[5:],
        tuple(bf(t) for t in down[:1])
        + down[1:3]
        + tuple(bf(t) for t in down[3:5])
        + down[5:],
    )
    assert rel(out, dense) < 0.03
    cot = torch.randn_like(dense)
    got = torch.autograd.grad((out.float() * cot).sum(), leaves)
    want = torch.autograd.grad((dense * cot).sum(), leaves)
    assert all(rel(a, b) < 0.05 for a, b in zip(got, want, strict=True))
    assert calls == [("gelu", False), ("", False), ("dgelu", True), ("", True)]
    assert not warpy.fits_accum(xb[:100], bf(up[0]))
    monkeypatch.setattr(warpy, "_ACCUM", False)
    assert not warpy.fits_accum(xb, bf(up[0]))
