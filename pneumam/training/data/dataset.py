import json
import os
import random
from collections.abc import Iterator
from dataclasses import dataclass

import torch
from safetensors.torch import load_file

from .prepare import FORMAT, INDEX


@dataclass
class Batch:
    """A batch of training samples of one image size.

    Attributes:
        latents: Float32 latents in the dataset's layout.
        captions: Captions as written, one per sample.
        subsets: Subset index of each sample.
        control: Control images as uint8 ``[B, 3, H, W]``, or None without control images.
    """

    latents: torch.Tensor
    captions: list[str]
    subsets: list[int]
    control: torch.Tensor | None = None


class ShardedDataset:
    """Pre-encoded training data read one shard at a time.

    Args:
        root: Folder written by ``prepare``.

    Attributes:
        count: Number of distinct samples.
        layout: Latent layout the data was encoded for.
        shards: One entry per shard file with its ``file``, ``count``, ``width``, ``height``
            and per-sample ``captions``, ``subsets``, ``repeats`` and ``flips``.

    Raises:
        ValueError: If the folder holds no dataset in this format.
    """

    def __init__(self, root: str | os.PathLike) -> None:
        self.root = root
        try:
            with open(os.path.join(root, INDEX), encoding="utf-8") as f:
                index = json.load(f)
        except OSError as e:
            raise ValueError(f"{root} holds no prepared dataset") from e
        if index.get("format") != FORMAT:
            raise ValueError(f"{root} is not a {FORMAT} dataset")
        self.count: int = index["count"]
        self.layout: str = index["layout"]
        self.shards: list[dict] = index["shards"]

    @property
    def epoch_size(self) -> int:
        """Samples drawn per epoch, counting repeats.

        Returns:
            The sum of every sample's repeats.
        """
        return sum(sum(s["repeats"]) for s in self.shards)

    def batches(
        self,
        batch_size: int,
        seed: int = 0,
        rank: int = 0,
        world: int = 1,
        epochs: int | None = None,
    ) -> Iterator[Batch]:
        """Iterate over shuffled batches, each drawn from a single image size.

        Every epoch visits the shards in a new order, draws each sample as often as its repeats
        say, and shuffles within each shard. Samples that may be flipped are mirrored half of
        the time, together with their control image. With several ranks, each draw goes to
        exactly one rank. Samples that do not fill a batch wait for the next shard of the same
        image size, also across epochs.

        Args:
            batch_size: Samples per batch.
            seed: Seed of the order and the flips. Use the same seed on every rank.
            rank: Index of this process.
            world: Number of processes sharing the data.
            epochs: Passes over the data. None repeats forever.

        Yields:
            Batches with tensors on the CPU.
        """
        pending: dict[tuple[int, int], Batch] = {}
        epoch = 0
        while epochs is None or epoch < epochs:
            rng = random.Random(seed * 1_000_003 + epoch)
            shards = list(self.shards)
            rng.shuffle(shards)
            for shard in shards:
                picks = [
                    i for i in range(shard["count"]) for _ in range(shard["repeats"][i])
                ]
                rng.shuffle(picks)
                flips = [shard["flips"][i] and rng.random() < 0.5 for i in picks]
                mine = list(range(rank, len(picks), world))
                if not mine:
                    continue
                data = load_file(os.path.join(self.root, shard["file"]))
                index = torch.tensor([picks[j] for j in mine], dtype=torch.long)
                flip = torch.tensor([flips[j] for j in mine])
                latents = data["latents"][index]
                if flip.any():
                    mirrored = data["latents_flip"][index]
                    shape = (-1,) + (1,) * (latents.dim() - 1)
                    latents = torch.where(flip.view(shape), mirrored, latents)
                control = None
                if "control" in data:
                    control = data["control"][index]
                    control = torch.where(
                        flip.view(-1, 1, 1, 1), control.flip(-1), control
                    )
                got = Batch(
                    latents,
                    [shard["captions"][picks[j]] for j in mine],
                    [shard["subsets"][picks[j]] for j in mine],
                    control,
                )
                size = (shard["width"], shard["height"])
                if size in pending:
                    got = _join(pending.pop(size), got)
                whole = len(got.captions) // batch_size * batch_size
                for i in range(0, whole, batch_size):
                    yield _slice(got, i, i + batch_size)
                if whole < len(got.captions):
                    pending[size] = _slice(got, whole, len(got.captions))
            epoch += 1


def _slice(b: Batch, start: int, end: int) -> Batch:
    control = None if b.control is None else b.control[start:end]
    return Batch(
        b.latents[start:end], b.captions[start:end], b.subsets[start:end], control
    )


def _join(a: Batch, b: Batch) -> Batch:
    control = None
    if a.control is not None and b.control is not None:
        control = torch.cat([a.control, b.control])
    return Batch(
        torch.cat([a.latents, b.latents]),
        a.captions + b.captions,
        a.subsets + b.subsets,
        control,
    )
