from typing import Any

import pytest
import torch
from helpers import padded_text, rel, tiny_model

from pneumam.models.anima import fused
from pneumam.runtime import Runtime
from pneumam.sampling.loop import sample
from pneumam.sampling.schedules import noise, sigmas

B, H, W, S, STEPS, SCALE = 2, 8, 12, 10, 6, 4.0


@pytest.fixture
def setup():
    m = tiny_model()
    cond, unc = padded_text(B, S, 32, (6, 6)), padded_text(B, S, 32, (3, 3))
    return m, cond, unc, noise(B, H, W, 4, 7, "cpu")


def test_euler_matches_a_hand_written_guided_loop(setup):
    m, cond, unc, x = setup
    rt, sg = Runtime(), sigmas(STEPS, m.spec.shift)
    cc, cu = m.encode_context(rt, cond), m.encode_context(rt, unc)
    z = x.clone()
    for i in range(STEPS):
        t = sg[i].expand(B)
        vc, vu = fused.forward(m, rt, z, t, cc), fused.forward(m, rt, z, t, cu)
        z = z + (sg[i + 1] - sg[i]) * (vu + SCALE * (vc - vu))
    out = sample(m, rt, cond, unc, x, STEPS, SCALE)
    assert out.shape == x.shape and out.dtype == torch.float32
    assert rel(out, z) < 1e-5


def test_no_negative_prompt_equals_a_guidance_scale_of_one(setup):
    m, cond, unc, x = setup
    rt = Runtime()
    one = sample(m, rt, cond, unc, x, STEPS, scale=1.0)
    assert rel(sample(m, rt, cond, None, x, STEPS), one) < 1e-5


def test_shift_overrides_the_model_default(setup):
    m, cond, unc, x = setup
    rt = Runtime()
    default = sample(m, rt, cond, unc, x, STEPS, SCALE)
    assert torch.equal(sample(m, rt, cond, unc, x, STEPS, SCALE, m.spec.shift), default)
    assert not torch.equal(sample(m, rt, cond, unc, x, STEPS, SCALE, 1.0), default)


def test_er_sde_ode_limit_stays_close_to_euler(setup):
    m, cond, unc, x = setup
    rt = Runtime()
    want = sample(m, rt, cond, unc, x, STEPS, SCALE)
    kw: dict[str, Any] = {"sampler": "er_sde", "noise_scale": "ode"}
    assert (
        rel(sample(m, rt, cond, unc, x, STEPS, SCALE, max_stage=1, **kw), want) < 1e-2
    )
    high = sample(m, rt, cond, unc, x, STEPS, SCALE, **kw)
    assert high.isfinite().all()


def test_er_sde_seeds_make_samples_repeatable_and_batch_independent(setup):
    m, cond, unc, x = setup
    rt = Runtime()
    kw: dict[str, Any] = {"sampler": "er_sde"}
    a = sample(m, rt, cond, unc, x, STEPS, SCALE, seeds=[1, 2], **kw)
    assert torch.equal(a, sample(m, rt, cond, unc, x, STEPS, SCALE, seeds=[1, 2], **kw))
    c = sample(m, rt, cond[:1], unc[:1], x[:1], STEPS, SCALE, seeds=[1], **kw)
    assert rel(c[0], a[0]) < 1e-4


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sampler": "x"},
        {"sampler": "er_sde", "max_stage": 4},
        {"sampler": "er_sde", "seeds": [1]},
        {"sampler": "er_sde", "noise_scale": "x"},
    ],
)
def test_bad_arguments_are_rejected(setup, kwargs):
    m, cond, unc, x = setup
    with pytest.raises(ValueError):
        sample(m, Runtime(), cond, unc, x, STEPS, **kwargs)


def test_prepared_run_lists_every_tensor_it_reads(setup):
    m, cond, unc, x = setup
    rt, sg = Runtime(), sigmas(STEPS, 3.0)
    prep = m.prepare(rt, sg[:STEPS], torch.cat([cond, unc]))
    z = torch.cat([x, x])
    a = prep.run(z, *prep.per_step(2), *prep.fixed)
    assert a.shape == z.shape and a.dtype == torch.float32
    assert not torch.equal(a, prep.run(z, *prep.per_step(3), *prep.fixed))
    hash(prep.key)
    other = m.prepare(rt, sg[:STEPS], torch.cat([unc, cond]))
    b = prep.run(z, *prep.per_step(2), *other.fixed)
    assert torch.equal(b, other.run(z, *other.per_step(2), *other.fixed))


def test_fp8_sampling_tracks_bf16(setup, monkeypatch):
    m, cond, unc, x = setup
    m, rt = m.to(torch.bfloat16), Runtime()
    want = sample(m, rt, cond, unc, x, STEPS, SCALE)
    fused.enable_fp8(m, rt)
    assert rel(sample(m, rt, cond, unc, x, STEPS, SCALE), want) < 0.3
    monkeypatch.setenv("PNEUMAM_EXPERIMENTAL_FP8_ATTENTION", "1")
    assert rel(sample(m, rt, cond, unc, x, STEPS, SCALE), want) < 0.3
