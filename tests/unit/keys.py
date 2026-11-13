import pytest

from pneumam.models.anima.keys import DIFFUSERS, KOHYA, PARTS, PROJ, locate, split_key


@pytest.mark.parametrize(
    ("key", "stem", "kind"),
    [
        ("blocks.3.self_attn.q_proj.lora_A.weight", "blocks.3.self_attn.q_proj", "A"),
        ("a.b.lora_B.default.weight", "a.b", "B"),
        (
            "lora_unet_blocks_3_mlp_layer1.lora_down.weight",
            "lora_unet_blocks_3_mlp_layer1",
            "A",
        ),
        (
            "lora_unet_blocks_3_mlp_layer1.lora_up.weight",
            "lora_unet_blocks_3_mlp_layer1",
            "B",
        ),
        (
            "lora_unet_blocks_3_mlp_layer2.alpha",
            "lora_unet_blocks_3_mlp_layer2",
            "alpha",
        ),
        ("blocks.3.qkv.weight", None, None),
        ("junk", None, None),
    ],
)
def test_split_key(key, stem, kind):
    assert split_key(key) == (stem, kind)


@pytest.mark.parametrize(
    "stem",
    [
        "blocks.3.self_attn.q_proj",
        "diffusion_model.blocks.3.self_attn.q_proj",
        "model.diffusion_model.blocks.3.self_attn.q_proj",
        "transformer.transformer_blocks.3.attn1.to_q",
        "base_model.model.transformer.transformer_blocks.3.attn1.to_q",
        "base_model.model.blocks.3.self_attn.q_proj",
        "lora_unet_blocks_3_self_attn_q_proj",
    ],
)
def test_every_naming_scheme_locates_the_same_projection(stem):
    assert locate(stem) == (3, "self_attn.q_proj")


@pytest.mark.parametrize(
    "stem",
    ["blocks.3.adaln", "lora_unet_blocks_3_nope", "blocks.x.self_attn.q_proj", "junk"],
)
def test_unsupported_modules_are_not_located(stem):
    assert locate(stem) is None


def test_key_maps_are_consistent():
    assert set(DIFFUSERS.values()) == set(PROJ) == set(KOHYA.values())
    for target, parts in PARTS.items():
        slots = sorted(i for t, i in PROJ.values() if t == target)
        assert slots == list(range(parts))
    for name in DIFFUSERS:
        assert locate(f"transformer_blocks.0.{name}") == (0, DIFFUSERS[name])
    for name in KOHYA:
        assert locate(f"lora_unet_blocks_12_{name}") == (12, KOHYA[name])
