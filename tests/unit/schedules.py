import numpy as np
import pytest
import torch

from pneumam.sampling.schedules import noise, sigmas


def test_sigmas_shape_and_ends():
    sg = sigmas(20, 3.0)
    assert sg.shape == (21,) and sg.dtype == torch.float32
    assert sg[0] == 1 and sg[-1] == 0
    assert (sg[1:] < sg[:-1]).all()


def test_shift_of_one_is_linear():
    assert torch.allclose(sigmas(4, 1.0), torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0]))


def test_larger_shift_keeps_noise_higher():
    assert (sigmas(10, 5.0)[1:-1] > sigmas(10, 2.0)[1:-1]).all()


def test_sigmas_match_the_diffusers_scheduler():
    diffusers = pytest.importorskip("diffusers")
    sch = diffusers.FlowMatchEulerDiscreteScheduler(shift=3.0)
    sch.set_timesteps(sigmas=np.linspace(1.0, 1 / 20, 20), device="cpu")
    assert (sch.sigmas.float() - sigmas(20, 3.0)).abs().max() < 1e-6


def test_noise_is_reproducible():
    a, b = noise(2, 4, 6, 16, 7, "cpu"), noise(2, 4, 6, 16, 7, "cpu")
    assert a.shape == (2, 4, 6, 16) and a.dtype == torch.float32
    assert torch.equal(a, b)
    assert not torch.equal(a, noise(2, 4, 6, 16, 8, "cpu"))
