import pytest
import torch
from helpers import rel, tiny_model
from safetensors.torch import save_file

from pneumam.adapters import lora as L
from pneumam.models.anima import fused
from pneumam.models.anima.convert import convert_lora
from pneumam.runtime import Runtime


def run(m):
    rt = Runtime()
    torch.manual_seed(1)
    x, t, text = (
        torch.randn(2, 8, 8, 4),
        torch.tensor([0.3, 0.7]),
        torch.randn(2, 6, 32),
    )
    return fused.forward(m, rt, x, t, m.encode_context(rt, text))


@pytest.fixture
def adapted():
    m = tiny_model()
    params = L.add_lora(m, rank=4, alpha=8.0)
    torch.manual_seed(2)
    with torch.no_grad():
        for p in params[1::2]:
            p.normal_(
                std=0.05
            )  # B starts at zero, which would make every check trivial
    return m


@pytest.fixture
def saved(adapted, tmp_path):
    path = tmp_path / "a.safetensors"
    L.save_lora(adapted, path)
    return path


def test_fresh_adapter_is_a_no_op_and_only_it_trains():
    m = tiny_model()
    base = run(m)
    assert not L.has_lora(m)
    params = L.add_lora(m, rank=4, alpha=8.0)
    assert L.has_lora(m)
    assert rel(run(m), base) < 1e-6
    trainable = [p for p in m.parameters() if p.requires_grad]
    assert len(trainable) == len(params) == len(L.lora_parameters(m))
    assert all(a is b for a, b in zip(trainable, params, strict=True))


def test_gradients_reach_the_adapter_and_not_the_base(adapted):
    assert rel(run(adapted), run(tiny_model())) > 1e-3
    run(adapted).square().sum().backward()
    assert all(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in L.lora_parameters(adapted)
    )
    assert adapted.blocks[0].qkv.base.weight.grad is None


def test_layers_and_targets_limit_where_adapters_attach():
    m = tiny_model()
    L.add_lora(m, rank=2, targets=("qkv", "down"), layers=[1])
    sd = L.lora_state_dict(m)
    assert set(sd) == {
        f"blocks.1.{t}.lora_{ab}" for t in ("qkv", "down") for ab in "AB"
    }
    assert sd["blocks.1.qkv.lora_A"].shape[0] == 3 * 2  # one rank-2 adapter per slice


def test_saved_file_records_what_it_targets(saved):
    sd, meta = L.read_lora(saved, "anima")
    assert meta == {
        "format": "pneumam-lora-1",
        "architecture": "anima",
        "rank": "4",
        "alpha": "8.0",
    }
    assert len(sd) == 2 * 7 * 3


def test_merging_the_file_equals_the_attached_adapter(adapted, saved):
    sd, _ = L.read_lora(saved)
    m = tiny_model()
    backup = L.merge_lora(m, [(sd, 1.0)])
    assert len(backup) == 7 * 3
    assert rel(run(m), run(adapted)) < 1e-5


def test_stacked_merge_sums_deltas_and_replaces_the_earlier_set(saved):
    sd, _ = L.read_lora(saved)
    m = tiny_model()
    orig = {k: v.clone() for k, v in m.state_dict().items()}
    backup = L.merge_lora(m, [(sd, 1.0)])
    other = {k: torch.randn_like(v) * 0.05 for k, v in sd.items() if ".0." in k}
    L.merge_lora(m, [(sd, 0.5), (other, 2.0)], backup)
    want = dict(orig)
    for s, w in ((sd, 0.5), (other, 2.0)):
        for k in s:
            if k.endswith(".lora_A"):
                b = k[:-7]
                want[b + ".weight"] = want[b + ".weight"] + w * (
                    s[b + ".lora_B"] @ s[k]
                )
    got = m.state_dict()
    assert max((got[k] - want[k]).abs().max().item() for k in want) < 1e-6
    L.unmerge_lora(m, backup)
    assert all(torch.equal(v, orig[k]) for k, v in m.state_dict().items())
    assert len(backup) == 0


def test_unmerge_is_exact_in_bf16(saved):
    sd, _ = L.read_lora(saved)
    m = tiny_model().to(torch.bfloat16)
    orig = {k: v.clone() for k, v in m.state_dict().items()}
    backup = L.merge_lora(m, [(sd, 1.0)])
    assert sum(not torch.equal(v, orig[k]) for k, v in m.state_dict().items()) == 7 * 3
    L.unmerge_lora(m, backup)
    assert all(torch.equal(v, orig[k]) for k, v in m.state_dict().items())


def test_load_then_remove(adapted, saved):
    want = run(adapted)
    m = tiny_model()
    L.load_lora(m, saved)
    assert m.blocks[0].qkv.alpha == 8.0
    assert torch.equal(run(m), want)
    L.remove_lora(m)
    assert not L.has_lora(m)
    assert rel(run(m), want) < 1e-5
    L.load_lora(m, saved)
    L.remove_lora(m, merge=False)
    assert rel(run(m), want) < 1e-5  # the first adapter stays merged


def test_converted_third_party_lora_touches_only_its_projection(tmp_path):
    a, b = torch.randn(4, 128), torch.randn(128, 4) * 0.05
    stem = "diffusion_model.blocks.1.self_attn.q_proj"
    conv, _ = convert_lora({f"{stem}.lora_A.weight": a, f"{stem}.lora_B.weight": b})
    path = tmp_path / "ext.safetensors"
    L.write_lora(path, conv, "anima")
    m = tiny_model()
    L.merge_lora(m, [(L.read_lora(path, "anima")[0], 1.0)])
    w0, w1 = tiny_model().blocks[1].qkv.weight, m.blocks[1].qkv.weight
    assert (w1[:128] - w0[:128] - b @ a).abs().max() < 1e-6
    assert torch.equal(w1[128:], w0[128:])


def test_legacy_adapter_files_still_read(saved, tmp_path):
    sd, _ = L.read_lora(saved)
    old = tmp_path / "old.safetensors"
    meta = {"format": "anima-flat-lora-1", "rank": "4", "alpha": "8.0"}
    save_file(sd, old, metadata=meta)
    assert len(L.read_lora(old, "anima")[0]) == len(sd)


def test_errors(adapted, saved, tmp_path):
    sd, _ = L.read_lora(saved)
    other, junk = tmp_path / "other.st", tmp_path / "junk.st"
    save_file(sd, other, metadata={"format": "pneumam-lora-1", "architecture": "wan"})
    save_file(sd, junk, metadata={"format": "nope"})
    stray = {
        "blocks.9.qkv.lora_A": sd["blocks.0.qkv.lora_A"],
        "blocks.9.qkv.lora_B": sd["blocks.0.qkv.lora_B"],
    }
    cases = (
        lambda: L.read_lora(other, "anima"),
        lambda: L.read_lora(junk),
        lambda: L.add_lora(tiny_model(), targets=("nope",)),
        lambda: L.add_lora(adapted, rank=2),
        lambda: L.merge_lora(tiny_model(), [(stray, 1.0)]),
        lambda: L.save_lora(tiny_model(), saved),
    )
    for case in cases:
        with pytest.raises(ValueError):
            case()


def test_dropout_takes_the_separate_path_and_stays_trainable():
    m = tiny_model()
    params = L.add_lora(m, rank=2, dropout=0.5)
    with torch.no_grad():
        for p in params[1::2]:
            p.normal_(std=0.05)
    m.eval()
    kept = run(m)
    m.train()
    dropped = run(m)
    assert not torch.equal(dropped, kept)  # half of the adapter's input is zeroed
    dropped.square().sum().backward()
    assert all(p.grad is not None for p in params)


def test_adapted_mlp_takes_the_fused_path_and_matches_the_separate_layers():
    m = tiny_model()
    params = L.add_lora(m, rank=2)
    with torch.no_grad():
        for p in params[1::2]:
            p.normal_(std=0.05)
    b = m.blocks[0]
    x = torch.randn(3, 5, b.up.base.in_features, requires_grad=True)
    fused_out = b.up.fused_mlp(b.down, x)
    separate = b.down(torch.nn.functional.gelu(b.up(x)))
    assert fused_out.shape == separate.shape and rel(fused_out, separate) < 1e-5
    leaves = (x, b.up.lora_A, b.up.lora_B, b.down.lora_A, b.down.lora_B)
    got = torch.autograd.grad(fused_out.square().sum(), leaves)
    want = torch.autograd.grad(separate.square().sum(), leaves)
    assert all(rel(a, c) < 1e-4 for a, c in zip(got, want, strict=True))
    assert b.up.fused_mlp(torch.nn.Linear(4, 4), x) is None
    L.enable_fp8_base(m)
    assert b.up.fused_mlp(b.down, x) is None
