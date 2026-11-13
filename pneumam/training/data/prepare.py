import json
import os
from collections.abc import Callable, Iterable
from typing import Any

import torch
from safetensors.torch import save_file

from .buckets import fit, make_buckets, pick_bucket

FORMAT = "pneumam-data-1"
INDEX = "index.json"
UNCOND = "uncond.safetensors"
IMAGE_TYPES = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def scan_folder(root: str | os.PathLike) -> list[tuple[str, str]]:
    """Pair every image under a folder with the caption next to it.

    The caption of ``name.png`` is the text of ``name.txt`` in the same folder. Images without a
    caption file get an empty caption.

    Args:
        root: Folder to search, including its subfolders.

    Returns:
        ``(image path, caption)`` pairs, sorted by path.
    """
    items = []
    for folder, _, names in os.walk(root):
        for name in names:
            stem, ext = os.path.splitext(name)
            if ext.lower() not in IMAGE_TYPES:
                continue
            caption = ""
            text = os.path.join(folder, stem + ".txt")
            if os.path.exists(text):
                with open(text, encoding="utf-8") as f:
                    caption = f.read().strip()
            items.append((os.path.join(folder, name), caption))
    return sorted(items)


def _to_tensor(image: Any) -> torch.Tensor:
    data = torch.frombuffer(
        bytearray(image.convert("RGB").tobytes()), dtype=torch.uint8
    )
    pixels = data.view(image.height, image.width, 3).permute(2, 0, 1)
    return pixels.float() / 127.5 - 1.0


def _trim(context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    index = torch.arange(1, context.shape[1] + 1, device=context.device)
    lengths = ((context != 0).any(-1) * index).amax(1).clamp_min(1)
    return context[:, : int(lengths.max())], lengths


@torch.no_grad()
def prepare(
    items: Iterable[tuple[str, str]],
    out_dir: str | os.PathLike,
    encoder: Any,
    text: Any,
    dtype: torch.dtype = torch.bfloat16,
    layout: str = "BHWC",
    buckets: list[tuple[int, int]] | None = None,
    shard_size: int = 256,
    batch_size: int = 8,
    on_progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Encode images and captions into shards a ``ShardedDataset`` reads.

    Each image is resized and center-cropped to the bucket closest to its aspect ratio. A shard
    holds samples of one bucket only.

    Args:
        items: ``(image path, caption)`` pairs. See ``scan_folder``.
        out_dir: Output folder. It is created, and an existing dataset in it is overwritten.
        encoder: VAE encoder for the model's latents. See ``Wan21Encoder``.
        text: The model's text encoder. Its ``encode(prompts, dtype)`` returns text embeddings
            of shape ``[B, S, text_dim]``, zero-padded to the longest prompt.
        dtype: Dtype the text embeddings are computed and stored in.
        layout: Latent layout of the model. See ``ModelSpec``.
        buckets: Image sizes to sort into. None uses ``make_buckets()``.
        shard_size: Samples per shard file.
        batch_size: Images encoded together.
        on_progress: Called with ``(samples done, samples total)`` after every batch.

    Returns:
        The dataset index that was written.

    Raises:
        ValueError: If ``items`` is empty.
    """
    from PIL import Image

    buckets = buckets or make_buckets()
    items = list(items)
    if not items:
        raise ValueError("no images to prepare")
    groups: dict[tuple[int, int], list[tuple[str, str]]] = {}
    for path, caption in items:
        with Image.open(path) as im:
            groups.setdefault(pick_bucket(im.width, im.height, buckets), []).append(
                (path, caption)
            )
    os.makedirs(out_dir, exist_ok=True)
    device = encoder.mean.device
    shards, done = [], 0

    def flush(size, latents, contexts, lengths):
        width, longest = size[0], int(torch.cat(lengths).max())
        pad = [
            torch.nn.functional.pad(c, (0, 0, 0, longest - c.shape[1]))
            for c in contexts
        ]
        name = f"{size[0]}x{size[1]}-{len(shards):05d}.safetensors"
        tensors = {
            "latents": torch.cat(latents).contiguous(),
            "context": torch.cat(pad).contiguous(),
            "lengths": torch.cat(lengths).to(torch.int32),
        }
        save_file(tensors, os.path.join(out_dir, name))
        count = tensors["latents"].shape[0]
        shards.append({"file": name, "count": count, "width": width, "height": size[1]})

    for size in sorted(groups):
        latents, contexts, lengths, held = [], [], [], 0
        group = groups[size]
        for i in range(0, len(group), batch_size):
            chunk = group[i : i + batch_size]
            images = []
            for path, _ in chunk:
                with Image.open(path) as im:
                    images.append(_to_tensor(fit(im, size)))
            lat = encoder.encode(torch.stack(images).to(device), layout)
            ctx, n = _trim(text.encode([c for _, c in chunk], dtype))
            latents.append(lat.float().cpu())
            contexts.append(ctx.cpu())
            lengths.append(n.cpu())
            held += len(chunk)
            done += len(chunk)
            if held >= shard_size:
                flush(size, latents, contexts, lengths)
                latents, contexts, lengths, held = [], [], [], 0
            if on_progress is not None:
                on_progress(done, len(items))
        if held:
            flush(size, latents, contexts, lengths)
    empty, _ = _trim(text.encode([""], dtype))
    save_file({"context": empty.cpu().contiguous()}, os.path.join(out_dir, UNCOND))
    index = {
        "format": FORMAT,
        "layout": layout,
        "count": done,
        "text_dim": int(empty.shape[-1]),
        "shards": shards,
    }
    with open(os.path.join(out_dir, INDEX), "w", encoding="utf-8") as f:
        json.dump(index, f, indent=1)
    return index
