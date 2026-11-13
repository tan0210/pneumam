import pytest
import torch
from helpers import padded_text, rel, tiny_config, tiny_model, tiny_wan_vae

from pneumam.components.vae import Wan21Decoder
from pneumam.pipeline import Pipeline, to_uint8
from pneumam.runtime import Runtime
from pneumam.sampling.loop import sample
from pneumam.sampling.schedules import noise

pytest.importorskip("diffusers")


class FakeText:
    def __init__(self):
        self.calls = []

    def encode(self, prompts, dtype):
        self.calls.append(list(prompts))
        torch.manual_seed(len(prompts[0]))
        used = tuple(3 + len(p) % 5 for p in prompts)
        return padded_text(len(prompts), max(used), 32, used).to(dtype)


@pytest.fixture(scope="module")
def decoder():
    return Wan21Decoder.from_diffusers(tiny_wan_vae()).requires_grad_(False)


@pytest.fixture
def pipe(decoder):
    return Pipeline(tiny_model(tiny_config(in_ch=16)), FakeText(), decoder)


def test_generate_is_embed_sample_decode(pipe):
    out = pipe.generate("a cat", "blurry", 64, 48, seeds=[3, 4], steps=4)
    assert out.shape == (2, 3, 48, 64) and out.abs().max() <= 1
    cond, unc = pipe.text.encode(["a cat", "blurry"], torch.float32).split(1)
    x = torch.cat([noise(1, 6, 8, 16, s, "cpu") for s in (3, 4)])
    rep = lambda t: t.expand(2, -1, -1).contiguous()
    lat = sample(pipe.model, Runtime(), rep(cond), rep(unc), x, 4, seeds=[3, 4])
    assert rel(out, pipe.decoder.decode(lat)) < 1e-5


def test_an_image_depends_only_on_its_own_seed(pipe):
    both = pipe.generate("a cat", "", 32, 32, seeds=[5, 9], steps=3)
    one = pipe.generate("a cat", "", 32, 32, seeds=[9], steps=3)
    assert rel(one[0], both[1]) < 1e-4
    assert not torch.equal(both[0], both[1])


def test_no_negative_prompt_skips_guidance(pipe):
    cond, uncond = pipe.embed("a cat", None)
    assert uncond is None and cond.shape[0] == 1
    assert pipe.text.calls[-1] == ["a cat"]
    a = pipe.generate("a cat", None, 32, 32, steps=3)
    b = pipe.generate("a cat", "dusty", 32, 32, steps=3, scale=1.0)
    assert rel(a, b) < 1e-4


def test_context_length_makes_the_image_independent_of_the_negative_prompt(pipe):
    short = pipe.generate("a cat", None, 32, 32, steps=3)
    longer = pipe.generate("a cat", "blurry", 32, 32, steps=3, scale=1.0)
    assert rel(longer, short) > 1e-3  # the model attends to the extra padding
    pipe.context_length = 12
    cond, uncond = pipe.embed("a cat", "blurry")
    assert cond.shape == uncond.shape == (1, 12, 32)
    short = pipe.generate("a cat", None, 32, 32, steps=3)
    longer = pipe.generate("a cat", "blurry", 32, 32, steps=3, scale=1.0)
    assert rel(longer, short) < 1e-4
    pipe.context_length = 2
    with pytest.raises(ValueError, match="context length"):
        pipe.embed("a cat")


def test_options_reach_the_sampler(pipe):
    kw = {"width": 32, "height": 32, "steps": 4}
    base = pipe.generate("a cat", **kw)
    assert not torch.equal(pipe.generate("a cat", sampler="er_sde", **kw), base)
    assert not torch.equal(pipe.generate("a cat", shift=1.0, **kw), base)
    with pytest.raises(ValueError, match="unknown sampler"):
        pipe.generate("a cat", sampler="x", **kw)


@pytest.mark.parametrize("size", [(40, 32), (32, 0), (30, 32)])
def test_sizes_must_be_whole_tokens(pipe, size):
    with pytest.raises(ValueError, match="multiples of 16"):
        pipe.generate("a cat", width=size[0], height=size[1], steps=2)


def test_seeds_are_required(pipe):
    with pytest.raises(ValueError, match="seed"):
        pipe.generate("a cat", width=32, height=32, seeds=[], steps=2)


def test_to_uint8():
    img = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]).view(1, 1, 1, 5).expand(1, 3, 1, 5)
    out = to_uint8(img)
    assert out.shape == (1, 1, 5, 3) and out.dtype == torch.uint8
    assert out[0, 0, :, 0].tolist() == [0, 0, 128, 255, 255]
