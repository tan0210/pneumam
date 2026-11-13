import pytest
import torch
from helpers import rel

from pneumam.sampling.guidance import guided
from pneumam.sampling.samplers.er_sde import er_sde, er_sde_plan
from pneumam.sampling.samplers.euler import euler
from pneumam.sampling.schedules import sigmas

TARGET = torch.tensor([[1.0, -2.0, 0.5], [0.0, 3.0, -1.0]])


def exact(i, x, sg):
    return (x - TARGET) / sg[i]  # a flow whose clean latent is always TARGET


def test_euler_reaches_the_target_of_an_exact_flow():
    sg = sigmas(8, 3.0)
    out = euler(lambda i, x: exact(i, x, sg), torch.randn(2, 3), sg)
    assert rel(out, TARGET) < 1e-5


@pytest.mark.parametrize("noise_scale", ["er", "ode"])
@pytest.mark.parametrize("max_stage", [1, 2, 3])
def test_er_sde_reaches_the_target_of_an_exact_flow(noise_scale, max_stage):
    sg = sigmas(8, 3.0)
    out = er_sde(
        lambda i, x: exact(i, x, sg),
        torch.randn(2, 3),
        sg,
        max_stage,
        seeds=[1, 2],
        noise_scale=noise_scale,
    )
    assert rel(out, TARGET) < 1e-4


def test_first_order_ode_solver_is_euler():
    sg = sigmas(12, 3.0)
    v = lambda i, x: torch.sin(x) + sg[i]
    x = torch.randn(2, 3)
    got = er_sde(v, x, sg, max_stage=1, noise_scale="ode")
    assert rel(got, euler(v, x, sg)) < 1e-3


def test_plan_ends_on_the_clean_prediction_and_ode_adds_no_noise():
    sg = sigmas(6, 3.0)
    plan = er_sde_plan(sg)
    assert len(plan) == 6
    assert plan[-1] == (0.0, (1.0, 0.0, 0.0), 0.0)
    assert all(std > 0 for _, _, std in plan[:-1])
    assert plan[0][1][1:] == (0.0, 0.0)  # no history on the first step
    assert er_sde_plan(sg, s_noise=0.0)[0][2] == 0.0
    from pneumam.sampling.samplers.er_sde import _ode_phi

    assert all(std == 0 for _, _, std in er_sde_plan(sg, phi=_ode_phi))


def test_seeded_noise_is_repeatable_and_independent_of_the_batch():
    sg = sigmas(6, 3.0)
    v = lambda i, x: x * 0.3
    x = torch.randn(2, 3)
    a = er_sde(v, x, sg, seeds=[1, 2])
    assert torch.equal(a, er_sde(v, x, sg, seeds=[1, 2]))
    assert torch.equal(a[:1], er_sde(v, x[:1], sg, seeds=[1]))
    assert not torch.equal(a, er_sde(v, x, sg, seeds=[3, 4]))


@pytest.mark.parametrize(
    "kwargs",
    [{"max_stage": 4}, {"seeds": [1]}, {"noise_scale": "x"}],
)
def test_er_sde_rejects_bad_arguments(kwargs):
    sg = sigmas(4, 3.0)
    with pytest.raises(ValueError):
        er_sde(lambda i, x: x, torch.randn(2, 3), sg, **kwargs)


def test_guidance_combines_the_two_halves():
    def run(x, bias):
        out = x + bias
        out[x.shape[0] // 2 :] *= -1  # the negative half predicts the opposite
        return out

    z, bias = torch.randn(2, 3), torch.ones(3)
    vc = z + bias
    assert torch.allclose(guided(run, 1.0)(z, bias), vc)
    assert torch.allclose(guided(run, 4.0)(z, bias), -vc + 4.0 * (2 * vc))
