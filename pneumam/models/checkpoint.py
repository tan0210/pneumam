import os
from collections.abc import Callable
from dataclasses import dataclass

import torch
from safetensors import safe_open
from safetensors.torch import save_file

FORMAT = "pneumam-1"
_LEGACY = {"anima-flat-1": "anima"}  # ids from before files named an architecture


@dataclass
class Header:
    """What a checkpoint says about itself.

    Attributes:
        architecture: Name of the model family the weights belong to.
        format: Format id the file was written with.
        config: The architecture's config, serialized by that architecture.
    """

    architecture: str
    format: str
    config: str


def read_header(path: str | os.PathLike) -> Header:
    """Read a checkpoint's metadata without loading its tensors.

    Args:
        path: A safetensors checkpoint.

    Returns:
        The header.

    Raises:
        ValueError: If the file is not a checkpoint this library wrote, or names no architecture
            or config.
    """
    with safe_open(path, framework="pt", device="cpu") as f:
        meta = f.metadata() or {}
    fmt = meta.get("format")
    architecture = _LEGACY.get(fmt) if fmt in _LEGACY else meta.get("architecture")  # type: ignore
    if fmt != FORMAT and fmt not in _LEGACY:
        raise ValueError(f"{path} is not a {FORMAT} checkpoint (format: {fmt!r})")
    if not architecture or "config" not in meta:
        raise ValueError(f"{path} does not name its architecture and config")
    return Header(architecture, fmt, meta["config"])  # type: ignore


def read_tensors(
    path: str | os.PathLike,
    device: str | torch.device,
    dtype_of: Callable[[str], torch.dtype] | None = None,
    skip: tuple[str, ...] = (),
) -> dict[str, torch.Tensor]:
    """Load a checkpoint's tensors onto a device.

    Args:
        path: A safetensors checkpoint.
        device: Device to load onto.
        dtype_of: Maps a tensor name to the dtype it is cast to. None keeps the stored dtypes.
        skip: Name prefixes to leave out.

    Returns:
        The tensors by name.
    """
    out = {}
    with safe_open(path, framework="pt", device=str(device)) as f:
        for key in f.keys():  # noqa: SIM118
            if skip and key.startswith(skip):
                continue
            t = f.get_tensor(key)
            out[key] = t if dtype_of is None else t.to(dtype_of(key))
    return out


def save(
    path: str | os.PathLike,
    tensors: dict[str, torch.Tensor],
    architecture: str,
    config: str,
) -> None:
    """Write a checkpoint.

    Args:
        path: Output path.
        tensors: The tensors by name.
        architecture: Name of the model family the weights belong to.
        config: The architecture's serialized config.
    """
    meta = {"format": FORMAT, "architecture": architecture, "config": config}
    save_file(tensors, path, metadata=meta)
