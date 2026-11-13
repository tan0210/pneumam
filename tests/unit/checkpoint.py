import pytest
import torch
from helpers import tiny_config, tiny_model
from safetensors.torch import save_file

from pneumam import models
from pneumam.models import checkpoint
from pneumam.models.anima import fused
from pneumam.models.anima.config import AnimaConfig
from pneumam.models.anima.dit import save_dit
from pneumam.ops import _dispatch
from pneumam.runtime import Runtime


@pytest.fixture
def model():
    return tiny_model(tiny_config(layers=2, mlp_dim=128))


@pytest.fixture
def saved(model, tmp_path):
    path = tmp_path / "new.safetensors"
    save_dit(model, path)
    return path


@pytest.fixture
def legacy(model, tmp_path):
    path = tmp_path / "old.safetensors"
    sd = {k: v.clone() for k, v in model.state_dict().items()}
    meta = {"format": "anima-flat-1", "config": model.cfg.dumps()}
    save_file({**sd, "vae.x": torch.zeros(3)}, path, metadata=meta)
    return path


def run(m):
    rt = Runtime()
    torch.manual_seed(1)
    x, t, text = torch.randn(1, 8, 8, 4), torch.tensor([0.5]), torch.randn(1, 6, 32)
    return fused.forward(m, rt, x, t, m.encode_context(rt, text))


def test_config_round_trips_through_json():
    cfg = tiny_config(rope_scale=(2.0, 3.0))
    back = AnimaConfig.loads(cfg.dumps())
    assert back == cfg and isinstance(back.rope_scale, tuple)
    assert cfg.dim == 128


def test_header_names_the_architecture(saved, legacy):
    new, old = checkpoint.read_header(saved), checkpoint.read_header(legacy)
    assert (new.architecture, new.format) == ("anima", "pneumam-1")
    assert (old.architecture, old.format) == ("anima", "anima-flat-1")


def test_loaded_model_reproduces_the_saved_one(model, saved, legacy):
    want = run(model)
    for path in (saved, legacy):
        m = models.load(path, "cpu", torch.float32)
        assert torch.equal(run(m), want)
        assert not m.training
        assert not any(p.requires_grad for p in m.parameters())


def test_default_dtypes_keep_sensitive_weights_in_float32(saved):
    m = models.load(saved, "cpu")
    assert m.patch.weight.dtype == torch.bfloat16
    assert m.mod_w1.dtype == torch.float32
    assert m.blocks[0].q_norm.dtype == torch.float32
    assert m.compute_dtype == torch.bfloat16


def test_load_requires_cuda_unless_a_path_is_forced(saved):
    _dispatch.force(None)
    with pytest.raises(RuntimeError, match="CUDA"):
        models.load(saved, "cpu")


def test_foreign_files_are_rejected(model, tmp_path):
    sd = {k: v.clone() for k, v in model.state_dict().items()}
    bad, odd, bare = (tmp_path / n for n in ("bad.st", "odd.st", "bare.st"))
    save_file(sd, bad, metadata={"format": "something-else"})
    meta = {"format": "pneumam-1", "architecture": "wan", "config": "{}"}
    save_file(sd, odd, metadata=meta)
    save_file(sd, bare, metadata={"format": "pneumam-1"})
    with pytest.raises(ValueError, match="not a pneumam-1 checkpoint"):
        models.load(bad, "cpu")
    with pytest.raises(ValueError, match="unknown architecture"):
        models.load(odd, "cpu")
    with pytest.raises(ValueError, match="does not name"):
        checkpoint.read_header(bare)


def test_read_tensors_skips_prefixes_and_casts(legacy):
    sd = checkpoint.read_tensors(legacy, "cpu", lambda k: torch.float16, ("vae.",))
    assert sd and not any(k.startswith("vae.") for k in sd)
    assert all(v.dtype == torch.float16 for v in sd.values())
    assert "vae.x" in checkpoint.read_tensors(legacy, "cpu")
