import json
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import torch
from safetensors.torch import save_file

from .buckets import fit, make_buckets, pick_bucket

FORMAT = "pneumam-data-2"
INDEX = "index.json"
IMAGE_TYPES = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


@dataclass(frozen=True)
class Sample:
    """One training image and what comes with it.

    Attributes:
        image: Path of the image.
        caption: Its caption as written.
        subset: Index of the subset it came from, which decides its caption rules.
        repeats: Times it is drawn per epoch.
        flip: Whether it may also be drawn mirrored left to right.
        control: Path of its control image, for training ControlNet-LLLite.
    """

    image: str
    caption: str = ""
    subset: int = 0
    repeats: int = 1
    flip: bool = False
    control: str | None = None


def _find_image(folder: str, stem: str) -> str | None:
    for ext in IMAGE_TYPES:
        for name in (stem + ext, stem + ext.upper()):
            path = os.path.join(folder, name)
            if os.path.exists(path):
                return path
    return None


def scan_folder(
    root: str | os.PathLike,
    caption_extension: str = ".txt",
    subset: int = 0,
    repeats: int = 1,
    flip: bool = False,
    control_dir: str | os.PathLike | None = None,
) -> list[Sample]:
    """List every image under a folder with the caption next to it.

    The caption of ``name.png`` is the text of ``name.txt`` in the same folder. Images without a
    caption file get an empty caption.

    Args:
        root: Folder to search, including its subfolders.
        caption_extension: Extension of the caption files.
        subset: Subset index given to every sample.
        repeats: Times each sample is drawn per epoch.
        flip: Whether the samples may also be drawn mirrored.
        control_dir: Folder holding a control image with the same name as each image, for
            ControlNet-LLLite. Subfolders of ``root`` are mirrored in it.

    Returns:
        The samples, sorted by path.

    Raises:
        ValueError: If ``control_dir`` is given and an image has no control image.
    """
    root = os.fspath(root)
    samples = []
    for folder, _, names in os.walk(root):
        for name in names:
            stem, ext = os.path.splitext(name)
            if ext.lower() not in IMAGE_TYPES:
                continue
            caption = ""
            text = os.path.join(folder, stem + caption_extension)
            if os.path.exists(text):
                with open(text, encoding="utf-8") as f:
                    caption = f.read().strip()
            control = None
            if control_dir is not None:
                where = os.path.join(
                    os.fspath(control_dir), os.path.relpath(folder, root)
                )
                control = _find_image(where, stem)
                if control is None:
                    raise ValueError(
                        f"no control image for {os.path.join(folder, name)}"
                    )
            path = os.path.join(folder, name)
            samples.append(Sample(path, caption, subset, repeats, flip, control))
    return sorted(samples, key=lambda s: s.image)


def _to_tensor(image: Any) -> torch.Tensor:
    data = torch.frombuffer(
        bytearray(image.convert("RGB").tobytes()), dtype=torch.uint8
    )
    return data.view(image.height, image.width, 3).permute(2, 0, 1)


@torch.no_grad()
def prepare(
    samples: Iterable[Sample],
    out_dir: str | os.PathLike,
    encoder: Any,
    layout: str = "BHWC",
    buckets: list[tuple[int, int]] | None = None,
    shard_size: int = 256,
    batch_size: int = 8,
    on_progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Encode images into shards a ``ShardedDataset`` reads.

    Each image is resized and center-cropped to the bucket closest to its aspect ratio, and its
    control image to the same size. A shard holds samples of one bucket only. Captions are
    stored as text, so caption rules can change them every time a sample is drawn.

    Args:
        samples: The samples. See ``scan_folder``.
        out_dir: Output folder. It is created, and an existing dataset in it is overwritten.
        encoder: VAE encoder for the model's latents. See ``Wan21Encoder``.
        layout: Latent layout of the model. See ``ModelSpec``.
        buckets: Image sizes to sort into. None uses ``make_buckets()``.
        shard_size: Samples per shard file.
        batch_size: Images encoded together.
        on_progress: Called with ``(samples done, samples total)`` after every batch.

    Returns:
        The dataset index that was written.

    Raises:
        ValueError: If ``samples`` is empty, or some samples have a control image and others
            do not.
    """
    from PIL import Image

    buckets = buckets or make_buckets()
    samples = list(samples)
    if not samples:
        raise ValueError("no images to prepare")
    controlled = {s.control is not None for s in samples}
    if len(controlled) > 1:
        raise ValueError("either every sample has a control image or none has")
    groups: dict[tuple[int, int], list[Sample]] = {}
    for s in samples:
        with Image.open(s.image) as im:
            groups.setdefault(pick_bucket(im.width, im.height, buckets), []).append(s)
    os.makedirs(out_dir, exist_ok=True)
    device = encoder.mean.device
    shards: list[dict[str, Any]] = []
    done = 0

    def flush(size, held: list[Sample], latents, flipped, controls):
        name = f"{size[0]}x{size[1]}-{len(shards):05d}.safetensors"
        tensors = {"latents": torch.cat(latents).contiguous()}
        if any(s.flip for s in held):
            tensors["latents_flip"] = torch.cat(flipped).contiguous()
        if controls:
            tensors["control"] = torch.cat(controls).contiguous()
        save_file(tensors, os.path.join(out_dir, name))
        shards.append(
            {
                "file": name,
                "count": len(held),
                "width": size[0],
                "height": size[1],
                "captions": [s.caption for s in held],
                "subsets": [s.subset for s in held],
                "repeats": [s.repeats for s in held],
                "flips": [s.flip for s in held],
            }
        )

    for size in sorted(groups):
        group = groups[size]
        held: list[Sample] = []
        latents, flipped, controls = [], [], []
        for i in range(0, len(group), batch_size):
            chunk = group[i : i + batch_size]
            pixels = []
            for s in chunk:
                with Image.open(s.image) as im:
                    pixels.append(_to_tensor(fit(im, size)))
            images = torch.stack(pixels).to(device).float() / 127.5 - 1.0
            latents.append(encoder.encode(images, layout).float().cpu())
            if any(s.flip for s in group):
                flipped.append(encoder.encode(images.flip(-1), layout).float().cpu())
            if chunk[0].control is not None:
                ctl = []
                for s in chunk:
                    with Image.open(s.control) as im:  # type: ignore[arg-type]
                        ctl.append(_to_tensor(fit(im, size)))
                controls.append(torch.stack(ctl))
            held += chunk
            done += len(chunk)
            if len(held) >= shard_size:
                flush(size, held, latents, flipped, controls)
                held, latents, flipped, controls = [], [], [], []
            if on_progress is not None:
                on_progress(done, len(samples))
        if held:
            flush(size, held, latents, flipped, controls)
    index = {"format": FORMAT, "layout": layout, "count": done, "shards": shards}
    with open(os.path.join(out_dir, INDEX), "w", encoding="utf-8") as f:
        json.dump(index, f, indent=1)
    return index
