import hashlib
import json
import os
import random
import tomllib
from collections.abc import Iterator
from dataclasses import dataclass, fields
from typing import Any

from .buckets import make_buckets
from .captions import CaptionRules
from .dataset import Batch, ShardedDataset
from .prepare import INDEX, Sample, prepare, scan_folder


@dataclass
class Subset:
    """A folder of images with captions, named like a subset in kohya's dataset config.

    Attributes:
        image_dir: Folder of images. A caption file next to each image is optional.
        num_repeats: Times each image is drawn per epoch.
        caption_extension: Extension of the caption files.
        caption_prefix: Text put before every caption, such as a trigger word.
        caption_suffix: Text put after every caption.
        shuffle_caption: Shuffle the comma-separated tags every time.
        keep_tokens: Leading tags kept in place.
        keep_tokens_separator: Everything before it is kept in place, if present.
        caption_tag_dropout_rate: Probability of dropping each other tag.
        caption_dropout_rate: Probability of training on the empty caption.
        flip_aug: Also train on the images mirrored left to right.
        conditioning_data_dir: Folder of control images with the same names, for LLLite.
    """

    image_dir: str
    num_repeats: int = 1
    caption_extension: str = ".txt"
    caption_prefix: str = ""
    caption_suffix: str = ""
    shuffle_caption: bool = False
    keep_tokens: int = 0
    keep_tokens_separator: str = ""
    caption_tag_dropout_rate: float = 0.0
    caption_dropout_rate: float = 0.0
    flip_aug: bool = False
    conditioning_data_dir: str | None = None

    def rules(self) -> CaptionRules:
        """Return how this subset's captions change when drawn.

        Returns:
            The caption rules.
        """
        return CaptionRules(
            self.caption_prefix,
            self.caption_suffix,
            self.shuffle_caption,
            self.keep_tokens,
            self.keep_tokens_separator,
            self.caption_tag_dropout_rate,
            self.caption_dropout_rate,
        )


@dataclass
class Dataset:
    """Subsets trained at one resolution, named like a dataset in kohya's dataset config.

    Attributes:
        subsets: The folders.
        resolution: Side of the square whose area every bucket shares.
        bucket_reso_steps: Bucket sides are multiples of this. Use a multiple of 16.
        max_aspect: Largest ratio between a bucket's long and short side.
    """

    subsets: list[Subset]
    resolution: int = 1024
    bucket_reso_steps: int = 64
    max_aspect: float = 2.0


SUBSET_KEYS = {f.name for f in fields(Subset)}
DATASET_KEYS = {f.name for f in fields(Dataset)} - {"subsets"}
# kohya options that change nothing here, accepted so its files load unchanged
IGNORED = {
    "batch_size",
    "enable_bucket",
    "bucket_no_upscale",
    "min_bucket_reso",
    "max_bucket_reso",
}


def load_datasets(path: str | os.PathLike) -> list[Dataset]:
    """Read a dataset config in the TOML format of kohya's sd-scripts.

    Options under ``[general]`` apply to every dataset and subset, and dataset options to its
    subsets, unless they set their own. Relative folders are taken relative to the file.

    Args:
        path: The TOML file.

    Returns:
        The datasets.

    Raises:
        ValueError: If an option is unknown or the file has no datasets.
    """
    with open(path, "rb") as f:
        data = tomllib.load(f)
    root = os.path.dirname(os.path.abspath(path))
    general = data.get("general", {})
    out = []
    for d in data.get("datasets", []):
        merged = {**general, **{k: v for k, v in d.items() if k != "subsets"}}
        subsets = []
        for s in d.get("subsets", []):
            opts = {**merged, **s}
            unknown = set(opts) - SUBSET_KEYS - DATASET_KEYS - IGNORED
            if unknown:
                raise ValueError(f"unknown dataset options {sorted(unknown)}")
            for key in ("image_dir", "conditioning_data_dir"):
                if opts.get(key):
                    opts[key] = os.path.normpath(
                        os.path.join(root, os.path.expanduser(opts[key]))
                    )
            subsets.append(
                Subset(**{k: v for k, v in opts.items() if k in SUBSET_KEYS})
            )
        kept = {k: v for k, v in merged.items() if k in DATASET_KEYS}
        out.append(Dataset(subsets, **kept))
    if not out:
        raise ValueError(f"{path} has no [[datasets]]")
    return out


def _key(ds: Dataset, samples: list[Sample], vae: str, layout: str) -> str:
    # only what the encoded tensors depend on; captions and repeats are refreshed in place
    files = [
        (
            s.image,
            os.path.getmtime(s.image),
            s.flip,
            s.control,
            None if s.control is None else os.path.getmtime(s.control),
        )
        for s in samples
    ]
    data = [files, ds.resolution, ds.bucket_reso_steps, ds.max_aspect, vae, layout]
    return hashlib.sha1(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]


def _refresh(folder: str, samples: list[Sample]) -> None:
    path = os.path.join(folder, INDEX)
    with open(path, encoding="utf-8") as f:
        index = json.load(f)
    by_image = {s.image: s for s in samples}
    changed = False
    for shard in index["shards"]:
        now = [by_image[i] for i in shard["images"]]
        fields = {
            "captions": [s.caption for s in now],
            "subsets": [s.subset for s in now],
            "repeats": [s.repeats for s in now],
        }
        for name, value in fields.items():
            if shard[name] != value:
                shard[name], changed = value, True
    if changed:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=1)
        os.replace(tmp, path)


def build_datasets(
    datasets: list[Dataset],
    cache_dir: str | os.PathLike,
    encoder: Any,
    layout: str,
    vae_id: str = "",
    encode: bool = True,
) -> tuple[list[ShardedDataset], list[CaptionRules]]:
    """Encode datasets into shards, reusing earlier encodings of unchanged data.

    Each dataset is cached under a name derived from its images, control images and settings,
    so adding or changing an image encodes that dataset again and nothing else. Captions,
    repeats and subset settings are not encoded, so changing them only updates the cache's
    index.

    In a multi-GPU run, let one rank call this first with ``encode=True`` and the others after
    a barrier, so the data is encoded once.

    Args:
        datasets: The datasets. See ``load_datasets``.
        cache_dir: Folder for the encoded shards.
        encoder: VAE encoder for the model's latents.
        layout: Latent layout of the model.
        vae_id: Identifies the VAE, for example its path, so another VAE encodes again.
        encode: Encode missing datasets. If False, a missing dataset is an error.

    Returns:
        ``(sets, rules)``: one ``ShardedDataset`` per dataset, and the caption rules of every
        subset, indexed by the subset numbers the batches carry.

    Raises:
        ValueError: If a dataset has no images, or is missing and ``encode`` is False.
    """
    sets, rules = [], []
    for d, ds in enumerate(datasets):
        samples = []
        for s in ds.subsets:
            samples += scan_folder(
                s.image_dir,
                s.caption_extension,
                len(rules),
                s.num_repeats,
                s.flip_aug,
                s.conditioning_data_dir,
            )
            rules.append(s.rules())
        if not samples:
            raise ValueError(f"dataset {d} has no images")
        folder = os.path.join(
            os.fspath(cache_dir), f"dataset-{d}-{_key(ds, samples, vae_id, layout)}"
        )
        if os.path.exists(os.path.join(folder, INDEX)):
            _refresh(folder, samples)
        else:
            if not encode:
                raise ValueError(f"dataset {d} is not encoded yet")
            buckets = make_buckets(
                ds.resolution**2, ds.bucket_reso_steps, ds.max_aspect
            )
            prepare(samples, folder, encoder, layout, buckets)
        sets.append(ShardedDataset(folder))
    return sets, rules


def mixed_batches(
    sets: list[ShardedDataset],
    batch_size: int,
    seed: int = 0,
    rank: int = 0,
    world: int = 1,
) -> Iterator[Batch]:
    """Draw batches from several datasets forever, each in proportion to its epoch size.

    Args:
        sets: The datasets.
        batch_size: Samples per batch.
        seed: Seed of the order. Use the same seed on every rank.
        rank: Index of this process.
        world: Number of processes sharing the data.

    Yields:
        Batches, each from one dataset and one image size.
    """
    streams = [
        ds.batches(batch_size, seed + i, rank, world) for i, ds in enumerate(sets)
    ]
    weights = [ds.epoch_size for ds in sets]
    rng = random.Random(seed)
    while True:
        yield next(streams[rng.choices(range(len(sets)), weights)[0]])
