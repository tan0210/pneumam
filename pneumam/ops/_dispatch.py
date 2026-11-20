import functools

import torch

_PATHS = ("reference", "triton")
_forced: str | None = None


@functools.cache
def _probe(device: torch.device) -> str:
    major, minor = torch.cuda.get_device_capability(device)
    if major >= 12:
        return "sm120"
    if major >= 10:
        return "sm100"
    if major == 9:
        return "sm90"
    if (major, minor) == (8, 9):
        return "sm89"
    if (major, minor) == (8, 6):
        return "sm86"
    if major == 8:
        return "sm80"
    return "other"


def arch(device: torch.device) -> str:
    """Name the kernel path ops should take on a device.

    Args:
        device: Device of the tensors an op was called with.

    Returns:
        The path pinned by ``force`` if there is one. Otherwise the GPU generation: ``"sm80"``,
        ``"sm86"``, ``"sm89"``, ``"sm90"``, ``"sm100"``, ``"sm120"``, or ``"other"`` for older GPUs.

    Raises:
        RuntimeError: If no path is pinned and ``device`` is not a CUDA device.
    """
    if _forced is not None:
        return _forced
    if device.type != "cuda":
        raise RuntimeError(f"pneumam runs on CUDA devices only, got {device}")
    return _probe(device)


def force(path: str | None) -> None:
    """Pin every op to one path, for tests and benchmarks.

    Args:
        path: ``"reference"`` for plain PyTorch, ``"triton"`` for the in-tree kernels without any
            external kernel library, or ``None`` to choose by GPU again.

    Raises:
        ValueError: If ``path`` is not one of the above.
    """
    global _forced
    if path is not None and path not in _PATHS:
        raise ValueError(f"unknown path {path!r}; expected one of {_PATHS} or None")
    _forced = path
