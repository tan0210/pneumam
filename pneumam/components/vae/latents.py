import torch

LAYOUTS = ("BHWC", "BCHW", "BCTHW")


def to_bchw(x: torch.Tensor, layout: str = "BHWC") -> torch.Tensor:
    """Reorder single-frame latents or images to ``[B, C, H, W]``.

    Args:
        x: Tensor in ``layout``.
        layout: ``"BHWC"``, ``"BCHW"``, or ``"BCTHW"`` with one frame.

    Returns:
        A view of ``x`` with shape ``[B, C, H, W]``.

    Raises:
        ValueError: If ``layout`` is unknown or a ``"BCTHW"`` tensor has more than one frame.
    """
    if layout == "BHWC":
        return x.permute(0, 3, 1, 2)
    if layout == "BCHW":
        return x
    if layout == "BCTHW":
        if x.shape[2] != 1:
            raise ValueError(f"expected a single frame, got {x.shape[2]}")
        return x[:, :, 0]
    raise ValueError(f"unknown layout {layout!r}; expected one of {LAYOUTS}")


def from_bchw(x: torch.Tensor, layout: str = "BHWC") -> torch.Tensor:
    """Reorder a ``[B, C, H, W]`` tensor to another layout.

    Args:
        x: Tensor of shape ``[B, C, H, W]``.
        layout: ``"BHWC"``, ``"BCHW"``, or ``"BCTHW"``, which gets one frame.

    Returns:
        A view of ``x`` in ``layout``.

    Raises:
        ValueError: If ``layout`` is unknown.
    """
    if layout == "BHWC":
        return x.permute(0, 2, 3, 1)
    if layout == "BCHW":
        return x
    if layout == "BCTHW":
        return x[:, :, None]
    raise ValueError(f"unknown layout {layout!r}; expected one of {LAYOUTS}")


def normalize(z: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Map raw VAE latents to the normalized space a diffusion model works in.

    Args:
        z: Raw latents, shape ``[B, C, H, W]``.
        mean: Per-channel mean of the raw latents, shape ``[C]``.
        std: Per-channel standard deviation of the raw latents, shape ``[C]``.

    Returns:
        ``(z - mean) / std`` in float32.
    """
    return (z.float() - mean.float().view(1, -1, 1, 1)) / std.float().view(1, -1, 1, 1)


def denormalize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Map normalized latents back to raw VAE latents. The inverse of ``normalize``.

    Args:
        x: Normalized latents, shape ``[B, C, H, W]``.
        mean: Per-channel mean of the raw latents, shape ``[C]``.
        std: Per-channel standard deviation of the raw latents, shape ``[C]``.

    Returns:
        ``x * std + mean`` in float32.
    """
    return x.float() * std.float().view(1, -1, 1, 1) + mean.float().view(1, -1, 1, 1)
