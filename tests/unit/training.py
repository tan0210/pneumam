import pytest
import torch
from helpers import padded_text, tiny_model

from pneumam.adapters import lora as L
from pneumam.models.anima import fused
from pneumam.runtime import Runtime
from pneumam.training.config import TrainConfig
from pneumam.training.objectives.flow import flow_loss, sample_sigmas
from pneumam.training.trainer import Trainer, learning_rate

B, H, W, S = 2, 8, 8, 12


def data(seed=0):
    g = torch.Generator().manual_seed(seed)
    while True:
        latents = torch.randn(B, H, W, 4, generator=g)
        text = torch.zeros(B, S, 32)
        text[0, :5] = torch.randn(5, 32, generator=g)
        text[1, :9] = torch.randn(9, 32, generator=g)
        yield latents, text


def make(cfg, seed=0, loss_fn=None):
    m = tiny_model(seed=seed)
    params = L.add_lora(m, rank=2)
    uncond = padded_text(1, S, 32, (2,))
    if loss_fn is not None:
        return m, Trainer(
            m, Runtime(), params, cfg, uncond, lambda a, b: loss_fn(m, a, b)
        )
    return m, Trainer(m, Runtime(), params, cfg, uncond)


class EagerStep:
    def __init__(self, loss_fn, params):
        self.loss_fn, self.params, self.calls = loss_fn, params, 0

    def __call__(self, *inputs):
        self.calls += 1
        for p in self.params:
            p.grad = None  # like a replay, this overwrites the gradients
        loss = self.loss_fn(*inputs)
        loss.backward()
        return loss.detach()


@pytest.mark.parametrize("dist", ["logit_normal", "uniform"])
def test_sigmas_are_noise_levels_and_shift_raises_them(dist):
    g = torch.Generator().manual_seed(0)
    low = sample_sigmas(4000, 1.0, dist, generator=g)
    high = sample_sigmas(4000, 3.0, dist, generator=g)
    assert low.shape == (4000,) and low.dtype == torch.float32
    assert 0 < low.min() and high.max() < 1
    assert abs(low.mean().item() - 0.5) < 0.03
    assert high.mean() > low.mean() + 0.1


def test_unknown_distribution_is_rejected():
    with pytest.raises(ValueError, match="unknown dist"):
        sample_sigmas(4, 3.0, "normal")


def test_flow_loss_is_the_velocity_error():
    m, rt = tiny_model(), Runtime()
    latents, text = torch.randn(B, H, W, 4), padded_text(B, S, 32, (5, 9))
    g = torch.Generator().manual_seed(3)
    loss = flow_loss(m, rt, latents, text, caption_dropout=0.0, generator=g)
    g = torch.Generator().manual_seed(3)
    eps = torch.randn(latents.shape, generator=g)
    sigma = sample_sigmas(B, m.spec.shift, generator=g)
    s = sigma.view(B, 1, 1, 1)
    v = fused.forward(
        m, rt, (1 - s) * latents + s * eps, sigma, m.encode_context(rt, text)
    )
    assert loss.ndim == 0 and loss.dtype == torch.float32
    assert torch.allclose(loss, (v - (eps - latents)).square().mean(), rtol=1e-5)


def test_caption_dropout_swaps_in_the_empty_prompt():
    m, rt = tiny_model(), Runtime()
    latents, text = torch.randn(B, H, W, 4), padded_text(B, S, 32, (5, 9))
    uncond = padded_text(1, S, 32, (2,))
    seeded = lambda: torch.Generator().manual_seed(5)
    dropped = flow_loss(
        m, rt, latents, text, uncond, caption_dropout=1.0, generator=seeded()
    )
    g = seeded()
    empty = flow_loss(
        m, rt, latents, uncond.expand(B, -1, -1), caption_dropout=0.0, generator=g
    )
    kept = flow_loss(
        m, rt, latents, text, uncond, caption_dropout=0.0, generator=seeded()
    )
    assert torch.allclose(dropped, empty, rtol=1e-5)
    assert not torch.allclose(dropped, kept, rtol=1e-5)
    with pytest.raises(ValueError, match="empty-prompt"):
        flow_loss(m, rt, latents, text, caption_dropout=0.5)


def test_loss_reaches_only_the_adapters():
    m, rt = tiny_model(), Runtime()
    params = L.add_lora(m, rank=2)
    with torch.no_grad():
        for p in params[1::2]:
            p.normal_(std=0.05)
    latents, text = torch.randn(B, H, W, 4), padded_text(B, S, 32, (5, 9))
    flow_loss(m, rt, latents, text, caption_dropout=0.0).backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in params)
    assert m.blocks[0].qkv.base.weight.grad is None


def test_learning_rate_warms_up_then_decays():
    cfg = TrainConfig(steps=100, lr=1.0, warmup=10, min_lr_ratio=0.1)
    rates = [learning_rate(i, cfg) for i in range(100)]
    assert rates[0] == pytest.approx(0.1) and rates[9] == pytest.approx(1.0)
    assert rates[10:] == sorted(rates[10:], reverse=True)
    assert learning_rate(1000, cfg) == pytest.approx(0.1)


def test_training_lowers_the_loss_on_a_fixed_batch():
    cfg = TrainConfig(steps=30, lr=2e-3, warmup=2, caption_dropout=0.0)
    m, trainer = make(cfg)
    batch = next(data())
    fixed = iter(lambda: batch, None)
    g = lambda: torch.Generator().manual_seed(9)
    before = flow_loss(m, trainer.rt, *batch, caption_dropout=0.0, generator=g()).item()
    seen = []
    trainer.fit(fixed, on_step=lambda step, loss: seen.append((step, loss)))
    after = flow_loss(m, trainer.rt, *batch, caption_dropout=0.0, generator=g()).item()
    assert trainer.step == 30 and [s for s, _ in seen] == list(range(1, 31))
    assert after < before
    assert all(p.grad is None for p in trainer.params)


def test_accumulation_averages_the_micro_batches():
    one = TrainConfig(steps=1, accum=1, caption_dropout=0.0, max_grad_norm=0.0)
    two = TrainConfig(steps=1, accum=2, caption_dropout=0.0, max_grad_norm=0.0)
    batch = next(data())

    def same_noise(m, latents, context):
        g = torch.Generator().manual_seed(1)
        return flow_loss(
            m, Runtime(), latents, context, caption_dropout=0.0, generator=g
        )

    def gradients(cfg):
        _, trainer = make(cfg, loss_fn=same_noise)
        with torch.no_grad():
            for p in trainer.params[1::2]:
                p.fill_(0.01)
        captured = []
        step = trainer.optimizer.step

        def spy():
            captured.extend(p.grad.clone() for p in trainer.params)
            step()

        trainer.optimizer.step = spy
        trainer.train_step(iter(lambda: batch, None))
        return captured

    pairs = zip(gradients(one), gradients(two), strict=True)
    assert all(torch.allclose(a, b, rtol=1e-4, atol=1e-7) for a, b in pairs)


def test_saved_run_resumes_exactly(tmp_path):
    cfg = TrainConfig(steps=6, lr=1e-3, warmup=2, ema=0.9)
    _, straight = make(cfg)
    straight.fit(data())
    _, first = make(cfg)
    stream = data()
    for _ in range(3):
        first.train_step(stream)
    first.save(tmp_path / "run.pt")
    _, second = make(cfg)
    with torch.no_grad():
        for p in second.params:
            p.normal_()  # whatever was here must be replaced by the file
    second.load(tmp_path / "run.pt")
    assert second.step == 3
    second.fit(stream)
    assert all(
        torch.equal(a, b) for a, b in zip(second.params, straight.params, strict=True)
    )
    assert all(torch.equal(a, b) for a, b in zip(second.ema, straight.ema, strict=True))


def test_load_rejects_other_parameters(tmp_path):
    cfg = TrainConfig(steps=1)
    _, trainer = make(cfg)
    trainer.save(tmp_path / "run.pt")
    m = tiny_model()
    other = Trainer(m, Runtime(), L.add_lora(m, rank=4), cfg)
    with pytest.raises(ValueError, match="different trainable"):
        other.load(tmp_path / "run.pt")


def test_average_tracks_the_weights_and_can_be_swapped_in():
    cfg = TrainConfig(steps=5, lr=5e-3, warmup=1, ema=0.5)
    _, trainer = make(cfg)
    trainer.fit(data())
    trained = [p.detach().clone() for p in trainer.params]
    assert any(not torch.equal(a, p) for a, p in zip(trainer.ema, trained, strict=True))
    with trainer.ema_weights():
        assert all(
            torch.equal(p, a) for p, a in zip(trainer.params, trainer.ema, strict=True)
        )
    assert all(torch.equal(p, t) for p, t in zip(trainer.params, trained, strict=True))


def test_config_round_trips_through_json():
    cfg = TrainConfig(steps=7, shift=2.5, ema=0.99)
    assert TrainConfig.loads(cfg.dumps()) == cfg


@pytest.mark.parametrize("accum", [1, 3])
def test_graphed_training_matches_eager_training(monkeypatch, accum):
    from pneumam.training import trainer as T

    monkeypatch.setattr(T, "GraphedStep", EagerStep)
    kw = {"steps": 4, "lr": 1e-3, "warmup": 1, "accum": accum, "ema": 0.9}
    _, eager = make(TrainConfig(**kw))
    eager.fit(data())
    _, graphed = make(TrainConfig(graph=True, **kw))
    graphed.fit(data())
    assert graphed._graphed.calls == 4 * accum
    pairs = zip(graphed.params, eager.params, strict=True)
    assert all(torch.allclose(a, b, rtol=1e-4, atol=1e-6) for a, b in pairs)


def test_flow_batch_and_velocity_loss_compose_to_flow_loss():
    from pneumam.training.objectives.flow import flow_batch, velocity_loss

    m, rt = tiny_model(), Runtime()
    latents, text = torch.randn(B, H, W, 4), padded_text(B, S, 32, (5, 9))
    uncond = padded_text(1, S, 32, (2,))
    seeded = lambda: torch.Generator().manual_seed(8)
    noisy, sigma, context, target = flow_batch(
        m, latents, text, uncond, generator=seeded()
    )
    assert (
        noisy.shape == latents.shape
        and sigma.shape == (B,)
        and target.dtype == torch.float32
    )
    whole = flow_loss(m, rt, latents, text, uncond, generator=seeded())
    assert torch.equal(velocity_loss(m, rt, noisy, sigma, context, target), whole)


def test_adapter_follows_in_place_updates_that_skip_version_counters():
    m = tiny_model()
    params = L.add_lora(m, rank=2)
    layer = m.blocks[0].qkv
    x = torch.randn(3, layer.base.in_features)
    before = layer(x)
    opt = torch.optim.AdamW(params, lr=0.05, fused=True)
    version = layer.lora_B._version
    layer(x).square().sum().backward()
    opt.step()
    assert layer.lora_B._version == version  # the fused update is invisible to autograd
    with torch.no_grad():
        want = layer.base(x) + x @ layer.delta_weight().T
    assert not torch.allclose(layer(x), before)
    assert torch.allclose(layer(x), want, atol=1e-5)


def test_frozen_fp8_product_and_its_gradients():
    from pneumam.ops import gemm

    x = torch.randn(2, 6, 64, requires_grad=True)
    w = torch.randn(128, 64)
    w8, w8_t = gemm.quantize_weight(w), gemm.quantize_weight(w.T.contiguous())
    cot = torch.randn(2, 6, 128)
    want = torch.autograd.grad(((x @ w.T) * cot).sum(), x)[0]
    y = gemm.frozen_linear_fp8(x, w, w8)
    assert y.shape == (2, 6, 128) and y.dtype == x.dtype
    assert ((y - x @ w.T).norm() / (x @ w.T).norm()) < 0.05
    exact = torch.autograd.grad((y * cot).sum(), x)[0]
    assert torch.allclose(exact, want, rtol=1e-4, atol=1e-5)  # straight-through
    y = gemm.frozen_linear_fp8(x, w, w8, w8_t)
    noisy = torch.autograd.grad((y * cot).sum(), x)[0]
    assert ((noisy - want).norm() / want.norm()) < 0.05
    assert not torch.equal(noisy, want)


def test_fp8_base_keeps_adapters_trainable():
    m, rt = tiny_model(), Runtime()
    params = L.add_lora(m, rank=2)
    with torch.no_grad():
        for p in params[1::2]:
            p.normal_(std=0.05)
    latents, text = torch.randn(B, H, W, 4), padded_text(B, S, 32, (5, 9))
    seeded = lambda: torch.Generator().manual_seed(2)
    loss = flow_loss(m, rt, latents, text, caption_dropout=0.0, generator=seeded())
    want = torch.autograd.grad(loss, params)
    for backward in (False, True):
        L.enable_fp8_base(m, backward)
        assert (m.blocks[0].qkv.fp8[1] is not None) == backward
        fp8 = flow_loss(m, rt, latents, text, caption_dropout=0.0, generator=seeded())
        got = torch.autograd.grad(fp8, params)
        assert abs(fp8.item() - loss.item()) / loss.item() < 0.1
        a = torch.cat([g.flatten() for g in got])
        b = torch.cat([g.flatten() for g in want])
        assert torch.nn.functional.cosine_similarity(a, b, dim=0) > 0.9
    L.disable_fp8_base(m)
    assert m.blocks[0].qkv.fp8 is None
    again = flow_loss(m, rt, latents, text, caption_dropout=0.0, generator=seeded())
    assert torch.equal(again, loss)


def test_trainer_switches_on_the_fp8_options():
    cfg = TrainConfig(
        steps=3, lr=1e-3, warmup=1, fp8_base=True, fp8_backward=True, fp8_attention=True
    )
    m, trainer = make(cfg)
    assert m.blocks[0].qkv.fp8 is not None and trainer.rt.fp8_attention_grad
    trainer.fit(data())
    assert trainer.step == 3 and trainer.rt.fp8_scales is not None
    assert all(torch.isfinite(p).all() for p in trainer.params)


def test_fp8_attention_grad_falls_back_without_a_kernel():
    from pneumam.ops import attention as A

    q, k, v = (torch.randn(2, 12, 4, 16, requires_grad=True) for _ in range(3))
    out = A.attention_fp8_grad(q, k, v, 4.0, 4.0)
    assert torch.equal(out, A.attention(q, k, v))
    assert all(g is not None for g in torch.autograd.grad(out.sum(), (q, k, v)))
