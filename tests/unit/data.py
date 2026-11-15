import random

import pytest
import torch

from pneumam.training.data.buckets import fit, make_buckets, pick_bucket
from pneumam.training.data.captions import CaptionRules
from pneumam.training.data.dataset import ShardedDataset
from pneumam.training.data.prepare import Sample, prepare, scan_folder

Image = pytest.importorskip("PIL.Image")

BUCKETS = [(64, 128), (96, 96), (128, 64)]


class FakeEncoder:
    mean = torch.zeros(4)

    def encode(self, images, layout):
        assert layout == "BHWC" and images.min() >= -1 and images.max() <= 1
        return torch.nn.functional.avg_pool2d(images, 8).permute(0, 2, 3, 1)


@pytest.fixture
def folder(tmp_path):
    root = tmp_path / "images"
    root.mkdir()
    sizes = [(100, 200)] * 5 + [(120, 110)] * 4 + [(300, 150)] * 3
    for i, size in enumerate(sizes):
        Image.new("RGB", size, (i * 20, 255 - i * 20, 128)).save(root / f"{i:02d}.png")
        if i != 3:
            (root / f"{i:02d}.txt").write_text(f"c{i}", encoding="utf-8")
    return root


@pytest.fixture
def dataset(folder, tmp_path):
    out = tmp_path / "data"
    seen = []
    index = prepare(
        scan_folder(folder),
        out,
        FakeEncoder(),
        buckets=BUCKETS,
        shard_size=3,
        batch_size=2,
        on_progress=lambda done, total: seen.append((done, total)),
    )
    assert seen[-1] == (12, 12)
    return ShardedDataset(out), index


def test_buckets_share_an_area_budget_and_cover_both_orientations():
    buckets = make_buckets(1024 * 1024, 64, 2.0)
    assert (1024, 1024) in buckets
    assert all(w % 64 == 0 and h % 64 == 0 and w * h <= 1024 * 1024 for w, h in buckets)
    assert all(max(w, h) / min(w, h) <= 2.0 for w, h in buckets)
    assert all((h, w) in buckets for w, h in buckets)
    assert len({round(w / h, 4) for w, h in buckets}) == len(buckets)
    with pytest.raises(ValueError):
        make_buckets(100, 64)


def test_images_go_to_the_closest_aspect_ratio():
    assert pick_bucket(1000, 2000, BUCKETS) == (64, 128)
    assert pick_bucket(500, 480, BUCKETS) == (96, 96)
    assert pick_bucket(1920, 1080, BUCKETS) == (128, 64)


def test_fit_covers_then_crops_the_center():
    image = Image.new("RGB", (300, 100), (0, 0, 0))
    image.paste((255, 255, 255), (100, 0, 200, 100))
    out = fit(image, (64, 64))
    assert out.size == (64, 64)
    assert out.getpixel((32, 32)) == (255, 255, 255)


def test_scan_pairs_images_with_captions_and_subset_settings(folder):
    samples = scan_folder(folder, subset=2, repeats=3, flip=True)
    assert len(samples) == 12
    assert samples[0].caption == "c0" and samples[3].caption == ""
    assert all(s.subset == 2 and s.repeats == 3 and s.flip for s in samples)


def test_scan_finds_control_images_or_fails(folder, tmp_path):
    control = tmp_path / "control"
    control.mkdir()
    for i in range(12):
        Image.new("RGB", (10, 10)).save(control / f"{i:02d}.jpg")
    samples = scan_folder(folder, control_dir=control)
    assert samples[0].control.endswith("00.jpg")  # type: ignore
    (control / "05.jpg").unlink()
    with pytest.raises(ValueError, match="no control image"):
        scan_folder(folder, control_dir=control)


def test_prepare_writes_one_bucket_per_shard(dataset):
    ds, _ = dataset
    assert ds.count == 12 and ds.layout == "BHWC" and ds.epoch_size == 12
    assert all(s["count"] <= 4 for s in ds.shards)
    by_size = {}
    for s in ds.shards:
        size = (s["width"], s["height"])
        by_size[size] = by_size.get(size, 0) + s["count"]
    assert by_size == {(64, 128): 5, (96, 96): 4, (128, 64): 3}


def test_batches_come_from_one_bucket_and_cover_every_sample_once(dataset):
    ds, _ = dataset
    seen = []
    for b in ds.batches(1, seed=1, epochs=1):
        assert b.latents.dtype == torch.float32 and b.control is None
        assert b.latents.shape[1:3] in {(16, 8), (12, 12), (8, 16)}
        seen += b.captions
    assert sorted(seen) == sorted([f"c{i}" for i in range(12) if i != 3] + [""])
    two = list(ds.batches(2, seed=1, epochs=1))
    assert all(len(b.captions) == 2 for b in two) and len(two) == 5


def test_order_depends_on_the_seed_and_changes_every_epoch(dataset):
    ds, _ = dataset

    def order(seed, epochs):
        return [b.captions[0] for b in ds.batches(1, seed=seed, epochs=epochs)]

    assert order(1, 1) == order(1, 1)
    assert order(1, 1) != order(2, 1)
    both = order(1, 2)
    assert both[:12] == order(1, 1) and both[12:] != both[:12]


def test_ranks_split_every_epoch_without_overlap(dataset):
    ds, _ = dataset
    parts = [
        [
            c
            for b in ds.batches(1, seed=4, rank=r, world=3, epochs=1)
            for c in b.captions
        ]
        for r in range(3)
    ]
    merged = [c for part in parts for c in part]
    assert sorted(merged) == sorted([f"c{i}" for i in range(12) if i != 3] + [""])
    assert all(parts)


def test_leftovers_carry_into_later_epochs(dataset):
    ds, _ = dataset
    stream = ds.batches(4, seed=0)
    assert sum(len(next(stream).captions) for _ in range(6)) == 24


def test_repeats_flips_and_control_images(tmp_path):
    root = tmp_path / "set"
    control = tmp_path / "control"
    root.mkdir()
    control.mkdir()
    for i in range(2):
        img = Image.new("RGB", (64, 64))
        img.paste((255, 255, 255), (0, 0, 32, 64))
        img.save(root / f"{i}.png")
        img.save(control / f"{i}.png")
    samples = scan_folder(root, repeats=5, flip=True, control_dir=control)
    prepare(samples, tmp_path / "d", FakeEncoder(), buckets=[(64, 64)])
    ds = ShardedDataset(tmp_path / "d")
    assert ds.epoch_size == 10
    batches = list(ds.batches(1, seed=3, epochs=1))
    assert len(batches) == 10
    for b in batches:
        mirrored = b.control[0, 0, 0, 0] == 0  # type: ignore # the white half moved to the right
        left = b.latents[0, :, 0, 0].mean() > b.latents[0, :, -1, 0].mean()
        assert bool(left) != bool(mirrored)
    assert 0 < sum(bool(b.control[0, 0, 0, 0] == 0) for b in batches) < 10  # type: ignore


def test_errors(dataset, tmp_path):
    with pytest.raises(ValueError, match="no prepared dataset"):
        ShardedDataset(tmp_path / "missing")
    with pytest.raises(ValueError, match="no images"):
        prepare([], tmp_path / "empty", FakeEncoder())
    with pytest.raises(ValueError, match="every sample"):
        prepare(
            [Sample("a.png", control="b.png"), Sample("c.png")],
            tmp_path / "x",
            FakeEncoder(),
        )


def test_caption_rules_follow_kohya():
    rng = random.Random(0)
    rules = CaptionRules(prefix="mychar", keep_tokens=1, shuffle=True)
    out = [rules.apply("1girl, solo, smile, outdoors", rng) for _ in range(20)]
    assert all(o.startswith("mychar, 1girl, ") for o in out)
    assert len(set(out)) > 1
    assert all(sorted(o.split(", ")[2:]) == ["outdoors", "smile", "solo"] for o in out)
    kept = CaptionRules(keep_tokens_separator="|||", tag_dropout=1.0)
    assert kept.apply("a, b ||| c, d", rng) == "a, b"
    assert CaptionRules(dropout=1.0).apply("a, b", rng) == ""
    assert CaptionRules(suffix="end").apply("a,  b ,", rng) == "a, b, end"


def test_only_flipping_samples_get_mirrored_latents(tmp_path):
    root = tmp_path / "set"
    root.mkdir()
    for i in range(2):
        Image.new("RGB", (64, 64), (200, 10 * i, 0)).save(root / f"{i}.png")
    samples = [scan_folder(root)[0], scan_folder(root, flip=True)[1]]
    prepare(samples, tmp_path / "d", FakeEncoder(), buckets=[(64, 64)])
    from safetensors.torch import load_file

    shard = ShardedDataset(tmp_path / "d").shards[0]
    data = load_file(tmp_path / "d" / shard["file"])
    flips = shard["flips"]
    assert data["latents_flip"][flips.index(False)].abs().max() == 0
    assert data["latents_flip"][flips.index(True)].abs().max() > 0


def test_exif_rotation_is_applied(tmp_path):
    root = tmp_path / "set"
    root.mkdir()
    img = Image.new("RGB", (64, 128))
    exif = img.getexif()
    exif[0x0112] = 6  # stored sideways, shown rotated 90 degrees
    img.save(root / "0.jpg", exif=exif)
    prepare(
        scan_folder(root), tmp_path / "d", FakeEncoder(), buckets=[(64, 128), (128, 64)]
    )
    shard = ShardedDataset(tmp_path / "d").shards[0]
    assert (shard["width"], shard["height"]) == (128, 64)
