import json
import os
import random
from collections.abc import Iterator

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from .prepare import FORMAT, INDEX, UNCOND


def _pad(context: torch.Tensor, length: int) -> torch.Tensor:
    if context.shape[1] > length:
        raise ValueError(
            f"a caption has {context.shape[1]} tokens, more than the context length {length}"
        )
    return F.pad(context, (0, 0, 0, length - context.shape[1]))


class ShardedDataset:
    """Pre-encoded training data read one shard at a time.

    Args:
        root: Folder written by ``prepare``.

    Attributes:
        count: Number of samples.
        layout: Latent layout the data was encoded for.
        shards: One entry per shard file with its ``file``, ``count``, ``width`` and ``height``.

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

    def uncond(self, context_length: int) -> torch.Tensor:
        """Load the empty-prompt embeddings, for caption dropout.

        Args:
            context_length: Tokens to zero-pad to.

        Returns:
            Tensor of shape ``[1, context_length, text_dim]``.

        Raises:
            ValueError: If the empty prompt is longer than ``context_length``.
        """
        context = load_file(os.path.join(self.root, UNCOND))["context"]
        return _pad(context, context_length)

    def batches(
        self,
        batch_size: int,
        context_length: int,
        seed: int = 0,
        rank: int = 0,
        world: int = 1,
        epochs: int | None = None,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Iterate over shuffled batches, each drawn from a single image size.

        Every epoch visits the shards in a new order and shuffles within each shard. With
        several ranks, each sample goes to exactly one rank per epoch. Samples that do not fill
        a batch are kept for the next shard of the same image size, also across epochs.

        Args:
            batch_size: Samples per batch.
            context_length: Tokens the text embeddings are zero-padded to.
            seed: Seed of the order. Use the same seed on every rank.
            rank: Index of this process.
            world: Number of processes sharing the data.
            epochs: Passes over the data. None repeats forever.

        Yields:
            ``(latents, context)`` on the CPU: float32 latents of batch size ``batch_size`` and
            text embeddings of shape ``[batch_size, context_length, text_dim]``.

        Raises:
            ValueError: If a caption is longer than ``context_length``.
        """
        pending: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
        epoch = 0
        while epochs is None or epoch < epochs:
            order = random.Random(seed * 1_000_003 + epoch)
            shards = list(self.shards)
            order.shuffle(shards)
            for shard in shards:
                picks = list(range(shard["count"]))
                order.shuffle(picks)
                mine = torch.tensor(picks[rank::world], dtype=torch.long)
                if mine.numel() == 0:
                    continue
                data = load_file(os.path.join(self.root, shard["file"]))
                latents = data["latents"][mine]
                context = _pad(data["context"][mine], context_length)
                size = (shard["width"], shard["height"])
                if size in pending:
                    latents = torch.cat([pending[size][0], latents])
                    context = torch.cat([pending[size][1], context])
                whole = latents.shape[0] // batch_size * batch_size
                for i in range(0, whole, batch_size):
                    yield latents[i : i + batch_size], context[i : i + batch_size]
                pending[size] = (latents[whole:], context[whole:])
            epoch += 1
