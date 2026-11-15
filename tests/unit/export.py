import torch
from helpers import rel, tiny_model

from pneumam.adapters import lora as L
from pneumam.adapters.lllite import attach_lllite, load_lllite, new_lllite, save_lllite
from pneumam.models.anima.convert import convert_lora, export_lora


def test_exported_lora_converts_back_exactly():
    m = tiny_model()
    params = L.add_lora(m, rank=4)
    with torch.no_grad():
        for p in params:
            p.normal_()
    sd = L.lora_state_dict(m)
    exported = export_lora(sd, torch.float32)
    assert "diffusion_model.blocks.0.self_attn.k_proj.lora_A.weight" in exported
    assert "diffusion_model.blocks.2.mlp.layer2.alpha" in exported
    back, skipped = convert_lora(exported)
    assert not skipped and back.keys() == sd.keys()
    assert all(rel(back[k], sd[k]) < 1e-6 for k in sd)


def test_new_lllite_starts_as_a_no_op_and_saves_in_kohya_format(tmp_path):
    m = tiny_model()
    torch.manual_seed(1)
    x = torch.randn(1, 16, 128)
    before = m.blocks[0].qkv(x)
    lllite = new_lllite(128, 2, mlp_dim=16, emb_dim=8, cond_dim=16, resblocks=1)
    control = attach_lllite(m, lllite)
    control.set_strength(1.0)
    control.cond = lllite.encode(torch.rand(1, 3, 64, 64))
    assert rel(m.blocks[0].qkv(x), before) < 1e-6
    save_lllite(lllite, tmp_path / "l.safetensors", torch.float32)
    back = load_lllite(tmp_path / "l.safetensors", "cpu", torch.float32)
    assert back.metadata["lllite.mlp_dim"] == "16" and sorted(back.targets) == [0, 1]
    assert all(
        torch.equal(back.state_dict()[k], v) for k, v in lllite.state_dict().items()
    )


def test_lllite_training_reaches_the_encoder_and_the_adapters():
    m = tiny_model()
    lllite = new_lllite(128, 2, mlp_dim=16, emb_dim=8, cond_dim=16, resblocks=1)
    with torch.no_grad():
        for p in lllite.parameters():
            if p.dim() > 1:
                p.normal_(std=0.1)
    control = attach_lllite(m, lllite)
    control.set_strength(1.0)
    control.cond = lllite.encode(torch.rand(2, 3, 64, 64))
    m.blocks[0].qkv(torch.randn(2, 16, 128)).square().mean().backward()
    assert lllite.conditioning.conv1.weight.grad is not None
    up = lllite.adapters["lllite_dit_blocks_0_self_attn_q_proj"].up.weight
    assert up.grad is not None and up.grad.abs().sum() > 0
