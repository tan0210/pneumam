import pytest
import torch

from pneumam import utils
from pneumam.ops import _dispatch, adaln, gemm


def test_forced_path_is_reported():
    assert _dispatch.arch(torch.device("cpu")) == "reference"
    _dispatch.force("triton")
    assert _dispatch.arch(torch.device("cpu")) == "triton"


def test_cpu_is_rejected_when_nothing_is_forced():
    _dispatch.force(None)
    x = torch.randn(2, 4, 8)
    with pytest.raises(RuntimeError, match="CUDA"):
        adaln.adaln(x, x[:, 0], x[:, 0])


def test_unknown_path_is_rejected():
    with pytest.raises(ValueError, match="unknown path"):
        _dispatch.force("nope")


def test_fp8_is_supported_on_the_reference_path():
    assert gemm.fp8_supported(torch.device("cpu"))


@pytest.mark.parametrize("value", ["", "0", "false", "No", " off "])
def test_experimental_flag_off(monkeypatch, value):
    monkeypatch.setenv("PNEUMAM_EXPERIMENTAL_FP8_ATTENTION", value)
    assert not utils.is_experimental_fp8_attention()


@pytest.mark.parametrize("value", ["1", "true", "yes"])
def test_experimental_flag_on(monkeypatch, value):
    monkeypatch.setenv("PNEUMAM_EXPERIMENTAL_FP8_ATTENTION", value)
    assert utils.is_experimental_fp8_attention()


def test_experimental_flag_defaults_off():
    assert not utils.is_experimental_fp8_attention()
