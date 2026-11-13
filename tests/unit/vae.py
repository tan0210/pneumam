import pytest
import torch
from helpers import rel, tiny_wan_vae
from safetensors.torch import save_file

from pneumam.components.vae import (
    Wan21Decoder,
    Wan21Encoder,
    latents,
    load_decoder,
    load_encoder,
    save_vae,
)

pytest.importorskip("diffusers")


@pytest.fixture(scope="module")
def vae():
    return tiny_wan_vae()


@pytest.fixture(scope="module")
def dec(vae):
    return Wan21Decoder.from_diffusers(vae)


@pytest.fixture(scope="module")
def enc(vae):
    return Wan21Encoder.from_diffusers(vae)


@pytest.fixture
def z():
    return torch.randn(2, 16, 6, 5)


@pytest.fixture
def img():
    return torch.rand(2, 3, 48, 40) * 2 - 1


def diffusers_decode(vae, z):
    with torch.no_grad():
        return vae.decode(z[:, :, None]).sample[:, :, 0]


def diffusers_encode(vae, img):
    with torch.no_grad():
        return vae.encode(img[:, :, None]).latent_dist.mean[:, :, 0]


def test_decoder_matches_diffusers(vae, dec, z):
    with torch.no_grad():
        assert rel(dec(z), diffusers_decode(vae, z)) < 1e-4


def test_encoder_matches_diffusers(vae, enc, img):
    assert rel(enc(img), diffusers_encode(vae, img)) < 1e-4


def test_encoder_with_attention_in_the_down_path(img, monkeypatch):
    from diffusers.models.autoencoders import autoencoder_kl_wan as akw

    orig = akw.WanAttentionBlock.forward
    # diffusers passes this block a cache argument it does not accept
    monkeypatch.setattr(
        akw.WanAttentionBlock, "forward", lambda self, x, **kw: orig(self, x)
    )
    vae = tiny_wan_vae(attn_scales=(0.5,))
    enc = Wan21Encoder.from_diffusers(vae)
    assert sum(type(layer).__name__ == "_Attn" for layer in enc.layers) == 3
    assert rel(enc(img), diffusers_encode(vae, img)) < 1e-4


def test_decode_normalizes_clamps_and_accepts_every_layout(vae, dec, z):
    x = latents.normalize(z, dec.mean, dec.std)
    out = dec.decode(x.permute(0, 2, 3, 1))
    assert rel(out, diffusers_decode(vae, z).clamp(-1, 1)) < 1e-4
    assert out.abs().max() <= 1
    assert torch.equal(dec.decode(x, "BCHW"), out)
    assert torch.equal(dec.decode(x[:, :, None], "BCTHW"), out)


def test_encode_returns_normalized_float32_latents(vae, enc, img):
    want = latents.normalize(diffusers_encode(vae, img), enc.mean, enc.std)
    got = enc.encode(img)
    assert got.shape == (2, 6, 5, 16) and got.dtype == torch.float32
    assert rel(got, want.permute(0, 2, 3, 1)) < 1e-4
    assert enc.encode(img, "BCHW").shape == (2, 16, 6, 5)
    assert enc.encode(img, "BCTHW").shape == (2, 16, 1, 6, 5)


def test_decoder_is_differentiable_and_encoder_is_frozen(vae, enc, z):
    dec = Wan21Decoder.from_diffusers(vae)
    zg = z.clone().requires_grad_()
    dec(zg).square().sum().backward()
    assert zg.grad is not None
    assert all(p.grad is not None for p in dec.parameters())
    assert not any(p.requires_grad for p in enc.parameters())


def test_file_round_trip(dec, enc, z, img, tmp_path):
    path = tmp_path / "vae.safetensors"
    save_vae(path, dec, enc)
    d, e = (
        load_decoder(path, "cpu", torch.float32),
        load_encoder(path, "cpu", torch.float32),
    )
    with torch.no_grad():
        assert torch.equal(d(z), dec(z))
    assert torch.equal(e(img), enc(img))
    assert not d.training and not any(p.requires_grad for p in d.parameters())
    assert d.in_w.is_contiguous(memory_format=torch.channels_last)


def test_decoder_embedded_in_an_older_checkpoint_loads(dec, z, tmp_path):
    path = tmp_path / "legacy.safetensors"
    sd = {"vae." + k: v.contiguous() for k, v in dec.state_dict().items()}
    meta = {"format": "anima-flat-1", "config": "{}"}
    save_file(sd | {"blocks.0.x": torch.zeros(1)}, path, metadata=meta)
    with torch.no_grad():
        assert torch.equal(load_decoder(path, "cpu", torch.float32)(z), dec(z))
    with pytest.raises(ValueError, match="not a pneumam-vae-1 file"):
        load_encoder(path, "cpu")


def test_latent_statistics_stay_float32_under_a_cast(dec, z, tmp_path):
    path = tmp_path / "vae.safetensors"
    save_vae(path, decoder=dec)
    b = load_decoder(path, "cpu", torch.bfloat16)
    assert b.in_w.dtype == torch.bfloat16
    assert b.mean.dtype == torch.float32 and torch.equal(b.mean, dec.mean)
    x = latents.normalize(z, dec.mean, dec.std).permute(0, 2, 3, 1)
    out = b.decode(x)
    assert out.dtype == torch.bfloat16
    assert rel(out, dec.decode(x)) < 0.1


def test_weights_write_back_to_diffusers(vae, dec, z):
    other = tiny_wan_vae()
    with torch.no_grad():
        for p in other.decoder.parameters():
            p.mul_(1.1)
    want = diffusers_decode(vae, z)
    assert rel(diffusers_decode(other, z), want) > 1e-2
    dec.to_diffusers(other)
    assert rel(diffusers_decode(other, z), want) < 1e-6


def test_latent_helpers():
    t = torch.randn(2, 16, 3, 4)
    mean, std = torch.randn(16), torch.rand(16) + 0.5
    back = latents.denormalize(latents.normalize(t, mean, std), mean, std)
    assert (back - t).abs().max() < 1e-5
    for layout in latents.LAYOUTS:
        assert torch.equal(latents.to_bchw(latents.from_bchw(t, layout), layout), t)
    assert latents.from_bchw(t, "BHWC").shape == (2, 3, 4, 16)


def test_errors(vae, dec, tmp_path):
    junk, other, only_dec = (tmp_path / n for n in ("junk.st", "other.st", "dec.st"))
    save_file({"a": torch.zeros(1)}, junk, metadata={"format": "nope"})
    meta = {"format": "pneumam-vae-1", "architecture": "flux"}
    save_file({"decoder.a": torch.zeros(1)}, other, metadata=meta)
    save_vae(only_dec, decoder=dec)
    x = torch.randn(1, 4, 4, 16)
    cases = (
        lambda: load_decoder(junk, "cpu"),
        lambda: load_decoder(other, "cpu"),
        lambda: load_encoder(only_dec, "cpu"),
        lambda: save_vae(junk),
        lambda: dec.decode(x, "NHWC"),
        lambda: dec.decode(torch.randn(1, 16, 2, 4, 4), "BCTHW"),
    )
    for case in cases:
        with pytest.raises(ValueError):
            case()
    vae.config.is_residual = True
    try:
        with pytest.raises(ValueError, match="2.1"):
            Wan21Decoder.from_diffusers(vae)
    finally:
        vae.config.is_residual = False
