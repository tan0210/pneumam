import pytest
import torch
from helpers import padded_text

from pneumam.training.data.buckets import fit, make_buckets, pick_bucket
from pneumam.training.data.dataset import ShardedDataset
from pneumam.training.data.prepare import prepare, scan_folder

Image = pytest.importorskip("PIL.Image")

BUCKETS = [(64, 128), (96, 96), (128, 64)]


class FakeEncoder:
    mean = torch.zeros(4)

    def encode(self, images, layout):
        assert layout == "BHWC" and images.min() >= -1 and images.max() <= 1
        return torch.nn.functional.avg_pool2d(images, 8).permute(0, 2, 3, 1)


class FakeText:
    def encode(self, prompts, dtype):
        used = tuple(1 + len(p) % 6 for p in prompts)
        out = padded_text(len(prompts), max(used), 16, used)
        for i, p in enumerate(prompts):
            out[i, 0, 0] = float(len(p))  # lets a test recognise the caption
        return out.to(dtype)


@pytest.fixture
def folder(tmp_path):
    root = tmp_path / "images"
    root.mkdir()
    sizes = [(100, 200)] * 5 + [(120, 110)] * 4 + [(300, 150)] * 3
    for i, size in enumerate(sizes):
        Image.new("RGB", size, (i * 20, 255 - i * 20, 128)).save(root / f"{i:02d}.png")
        if i != 3:
            (root / f"{i:02d}.txt").write_text("c" * (i + 1), encoding="utf-8")
    return root


@pytest.fixture
def dataset(folder, tmp_path):
    out = tmp_path / "data"
    seen = []
    index = prepare(
        scan_folder(folder),
        out,
        FakeEncoder(),
        FakeText(),
        torch.float32,
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


def test_scan_pairs_images_with_captions(folder):
    items = scan_folder(folder)
    assert len(items) == 12
    assert items[0][1] == "c" and items[3][1] == ""
    assert all(path.endswith(".png") for path, _ in items)


def test_prepare_writes_one_bucket_per_shard(dataset):
    ds, index = dataset
    assert ds.count == 12 and ds.layout == "BHWC"
    assert sum(s["count"] for s in ds.shards) == 12
    assert all(s["count"] <= 4 for s in ds.shards)
    by_size = {}
    for s in ds.shards:
        by_size[(s["width"], s["height"])] = (
            by_size.get((s["width"], s["height"]), 0) + s["count"]
        )
    assert by_size == {(64, 128): 5, (96, 96): 4, (128, 64): 3}
    assert index["text_dim"] == 16
    assert ds.uncond(10).shape == (1, 10, 16)


def test_batches_come_from_one_bucket_and_cover_every_sample_once(dataset):
    ds, _ = dataset
    seen = []
    for latents, context in ds.batches(1, 8, seed=1, epochs=1):
        assert latents.dtype == torch.float32 and context.shape == (1, 8, 16)
        assert latents.shape[1:3] in {(16, 8), (12, 12), (8, 16)}
        seen.append(int(context[0, 0, 0]))
    assert sorted(seen) == sorted([i + 1 for i in range(12) if i != 3] + [0])
    two = list(ds.batches(2, 8, seed=1, epochs=1))
    assert (
        all(lat.shape[0] == 2 for lat, _ in two) and len(two) == 5
    )  # one of each odd bucket waits


def test_order_depends_on_the_seed_and_changes_every_epoch(dataset):
    ds, _ = dataset
    order = lambda seed, epochs: [
        int(c[0, 0, 0]) for _, c in ds.batches(1, 8, seed=seed, epochs=epochs)
    ]
    assert order(1, 1) == order(1, 1)
    assert order(1, 1) != order(2, 1)
    both = order(1, 2)
    assert both[:12] == order(1, 1) and both[12:] != both[:12]


def test_ranks_split_every_epoch_without_overlap(dataset):
    ds, _ = dataset
    parts = [
        [
            int(c[0, 0, 0])
            for _, c in ds.batches(1, 8, seed=4, rank=r, world=3, epochs=1)
        ]
        for r in range(3)
    ]
    merged = [c for part in parts for c in part]
    assert sorted(merged) == sorted([i + 1 for i in range(12) if i != 3] + [0])
    assert all(parts)


def test_leftovers_carry_into_later_epochs(dataset):
    ds, _ = dataset
    stream = ds.batches(4, 8, seed=0)
    total = sum(next(stream)[0].shape[0] for _ in range(6))
    assert total == 24


def test_errors(dataset, tmp_path):
    ds, _ = dataset
    with pytest.raises(ValueError, match="context length"):
        next(ds.batches(1, 2))
    with pytest.raises(ValueError, match="no prepared dataset"):
        ShardedDataset(tmp_path / "missing")
    with pytest.raises(ValueError, match="no images"):
        prepare([], tmp_path / "empty", FakeEncoder(), FakeText())
