import math
from typing import Any


def make_buckets(
    area: int = 1024 * 1024, step: int = 64, max_ratio: float = 2.0
) -> list[tuple[int, int]]:
    """List the image sizes training images are sorted into, one per aspect ratio.

    Args:
        area: Largest pixel count of a bucket.
        step: Both sides of every bucket are multiples of this. Use a multiple of the pixels one
            model token covers.
        max_ratio: Largest ratio between the long and the short side.

    Returns:
        ``(width, height)`` pairs, widest last, each as large as fits in ``area``.

    Raises:
        ValueError: If no bucket fits.
    """
    sizes = set()
    short = step
    while short * short <= area:
        long = min(area // short // step * step, int(short * max_ratio) // step * step)
        if long >= short:
            sizes.update({(short, long), (long, short)})
        short += step
    best: dict[float, tuple[int, int]] = {}
    for w, h in sizes:
        ratio = round(w / h, 4)
        if ratio not in best or w * h > best[ratio][0] * best[ratio][1]:
            best[ratio] = (w, h)
    if not best:
        raise ValueError(
            f"no bucket with sides that are multiples of {step} fits in {area} pixels"
        )
    return sorted(best.values(), key=lambda s: s[0] / s[1])


def pick_bucket(
    width: int, height: int, buckets: list[tuple[int, int]]
) -> tuple[int, int]:
    """Choose the bucket whose aspect ratio is closest to an image's.

    Args:
        width: Image width.
        height: Image height.
        buckets: Candidates from ``make_buckets``.

    Returns:
        The chosen ``(width, height)``.
    """
    target = math.log(width / height)
    return min(buckets, key=lambda s: abs(math.log(s[0] / s[1]) - target))


def fit(image: Any, size: tuple[int, int]) -> Any:
    """Resize a PIL image to cover a size and crop the overflow from the center.

    Args:
        image: The image.
        size: Target ``(width, height)``.

    Returns:
        A new image of exactly ``size``.
    """
    from PIL import Image

    w, h = size
    scale = max(w / image.width, h / image.height)
    rw, rh = max(w, round(image.width * scale)), max(h, round(image.height * scale))
    resized = image.resize((rw, rh), Image.Resampling.LANCZOS)
    left, top = (rw - w) // 2, (rh - h) // 2
    return resized.crop((left, top, left + w, top + h))
