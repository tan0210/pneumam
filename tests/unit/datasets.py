import os

import pytest
import torch

from pneumam.training.data.config import build_datasets, load_datasets, mixed_batches
from pneumam.training.previews import Preview, load_previews

Image = pytest.importorskip("PIL.Image")
EXAMPLES = os.path.join(os.path.dirname(__file__), "..", "..", "examples")


class FakeEncoder:
    mean = torch.zeros(4)

    def __init__(self):
        self.calls = 0

    def encode(self, images, layout):
        self.calls += 1
        return torch.nn.functional.avg_pool2d(images, 8).permute(0, 2, 3, 1)


def test_example_config_merges_general_options_into_subsets():
    (ds,) = load_datasets(os.path.join(EXAMPLES, "dataset.toml"))
    first, second = ds.subsets
    assert ds.resolution == 1024 and os.path.isabs(first.image_dir)
    assert first.shuffle_caption and first.keep_tokens == 1 and first.num_repeats == 10
    assert first.rules().prefix == "mychar" and first.flip_aug
    assert not second.shuffle_caption and second.keep_tokens == 1


def test_kohya_only_options_load_and_unknown_ones_fail(tmp_path):
    f = tmp_path / "d.toml"
    f.write_text(
        '[[datasets]]\nbatch_size = 4\nenable_bucket = true\n[[datasets.subsets]]\nimage_dir = "x"\n'
    )
    assert load_datasets(f)[0].subsets[0].image_dir == os.path.normpath(
        str(tmp_path / "x")
    )
    f.write_text(
        '[[datasets]]\n[[datasets.subsets]]\nimage_dir = "x"\nnum_repeat = 2\n'
    )
    with pytest.raises(ValueError, match="num_repeat"):
        load_datasets(f)
    f.write_text("")
    with pytest.raises(ValueError, match="datasets"):
        load_datasets(f)


def test_build_reuses_the_cache_until_the_data_changes(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    for i in range(3):
        Image.new("RGB", (64, 64), (i * 50, 0, 0)).save(images / f"{i}.png")
        (images / f"{i}.txt").write_text(f"tag{i}, other")
    cfg = tmp_path / "d.toml"
    cfg.write_text(
        '[[datasets]]\nresolution = 64\n[[datasets.subsets]]\nimage_dir = "images"\n'
        "num_repeats = 2\nshuffle_caption = true\n"
    )
    enc = FakeEncoder()
    sets, rules = build_datasets(load_datasets(cfg), tmp_path / "cache", enc, "BHWC")
    calls = enc.calls
    assert sets[0].epoch_size == 6 and rules[0].shuffle
    build_datasets(load_datasets(cfg), tmp_path / "cache", enc, "BHWC")
    assert enc.calls == calls
    (images / "1.txt").write_text("changed")
    with pytest.raises(ValueError, match="not encoded"):
        build_datasets(
            load_datasets(cfg), tmp_path / "cache", None, "BHWC", encode=False
        )
    sets, _ = build_datasets(load_datasets(cfg), tmp_path / "cache", enc, "BHWC")
    assert enc.calls > calls
    stream = mixed_batches(sets, 2, seed=0, rank=1, world=2)
    assert all(len(next(stream).captions) == 2 for _ in range(4))


def test_prompt_files_skip_comments_and_resolve_control_images(tmp_path):
    f = tmp_path / "p.txt"
    f.write_text("# comment\n\na, b --d 3\nc --cn ctl/x.png --w 512\n")
    first, second = load_previews(f)
    assert first == Preview("a, b", seed=3)
    assert second.width == 512 and second.control == os.path.normpath(
        str(tmp_path / "ctl/x.png")
    )
