import pytest
import torch

from pneumam.ops import _dispatch


@pytest.fixture(autouse=True)
def reference_path(monkeypatch):
    monkeypatch.delenv("PNEUMAM_EXPERIMENTAL_FP8_ATTENTION", raising=False)
    torch.manual_seed(0)
    _dispatch.force("reference")  # the only path that runs without a GPU
    yield
    _dispatch.force(None)
