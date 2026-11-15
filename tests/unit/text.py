import pytest
import torch
from helpers import rel

from pneumam.models.anima.text import Conditioner, Qwen3

transformers = pytest.importorskip("transformers")
conditioner_mod = pytest.importorskip(
    "diffusers.models.condition_embedders.condition_embedder_anima"
)

QWEN = {
    "hidden_size": 64,
    "head_dim": 16,
    "intermediate_size": 96,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "num_hidden_layers": 2,
    "rms_norm_eps": 1e-6,
    "vocab_size": 100,
    "rope_parameters": {"rope_theta": 1e6, "rope_type": "default"},
}
COND = {
    "source_dim": 64,
    "target_dim": 64,
    "model_dim": 64,
    "num_layers": 2,
    "num_attention_heads": 4,
    "mlp_ratio": 4.0,
    "target_vocab_size": 50,
    "use_self_attention": True,
    "use_layer_norm": False,
    "min_sequence_length": 12,
}


@pytest.fixture(autouse=True)
def no_capture_query(monkeypatch):
    if (
        not torch.cuda.is_available()
    ):  # transformers' capture query crashes on broken CUDA
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)


def _randomize(module):
    torch.manual_seed(0)
    with torch.no_grad():
        for p in module.parameters():
            p.normal_(std=0.3)
    return module


def test_qwen3_matches_transformers_on_right_padded_batches():
    cfg = transformers.Qwen3Config(
        **QWEN, attention_bias=False, max_position_embeddings=64
    )
    ref = _randomize(transformers.Qwen3Model(cfg)).eval()
    ours = Qwen3(QWEN)
    ours.load(ref.state_dict())
    ids = torch.randint(0, 100, (2, 9))
    mask = torch.ones(2, 9, dtype=torch.long)
    mask[1, 5:] = 0
    want = ref(input_ids=ids, attention_mask=mask).last_hidden_state
    got = ours(ids)
    assert rel(got[0], want[0]) < 1e-5
    assert (
        rel(got[1, :5], want[1, :5]) < 1e-5
    )  # padding after a prompt does not change it


def test_conditioner_matches_diffusers():
    ref = _randomize(conditioner_mod.AnimaTextConditioner(**COND)).eval()
    ours = Conditioner(COND)
    ours.load(ref.state_dict())
    source = torch.randn(2, 7, 64)
    source_mask = torch.ones(2, 7, dtype=torch.bool)
    source_mask[1, 4:] = False
    source = source * source_mask[..., None]
    ids = torch.randint(0, 50, (2, 6))
    mask = torch.ones(2, 6, dtype=torch.bool)
    mask[0, 3:] = False
    want = ref(source, ids, mask.long(), source_mask.long())
    got = ours(source, source_mask, ids, mask)
    assert got.shape == want.shape == (2, 12, 64)
    assert rel(got, want) < 1e-5
    assert got[0, 3:].abs().max() == 0
