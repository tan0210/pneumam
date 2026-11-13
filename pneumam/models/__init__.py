import os

import torch
from torch import nn

from ..ops import _dispatch
from .checkpoint import read_header


def load(
    path: str | os.PathLike,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> nn.Module:
    """Load a model from a checkpoint for inference.

    The checkpoint decides which architecture is built. State that is not a weight, such as FP8
    weights or context trimming, lives in a ``Runtime`` the caller creates.

    Args:
        path: A safetensors checkpoint written by this library.
        device: CUDA device to load the weights onto.
        dtype: Dtype of ordinary weights. Each architecture keeps its precision-sensitive
            weights in float32.

    Returns:
        The model in eval mode with gradients disabled.

    Raises:
        RuntimeError: If ``device`` is not a CUDA device.
        ValueError: If the file is not a checkpoint this library wrote, or its architecture is
            unknown.
    """
    device = torch.device(device)
    _dispatch.arch(device)
    header = read_header(path)
    if header.architecture == "anima":
        from .anima.dit import load_dit

        return load_dit(path, header.config, device, dtype)
    raise ValueError(f"{path} holds an unknown architecture {header.architecture!r}")
