import torch


def sigmas(n: int, shift: float) -> torch.Tensor:
    """Build the shifted flow-matching noise schedule.

    Args:
        n: Number of denoising steps.
        shift: Schedule shift. Larger values spend more steps at high noise.

    Returns:
        Float32 tensor of shape ``[n + 1]``, decreasing from 1 to a final 0.
    """
    s = torch.linspace(1.0, 1.0 / n, n, dtype=torch.float64)
    s = shift * s / (1 + (shift - 1) * s)
    return torch.cat([s, s.new_zeros(1)]).float()


def noise(
    batch: int, h: int, w: int, channels: int, seed: int, device: str | torch.device
) -> torch.Tensor:
    """Draw reproducible Gaussian latent noise.

    Args:
        batch: Batch size.
        h: Latent height.
        w: Latent width.
        channels: Latent channels.
        seed: RNG seed.
        device: Device for the generator and result.

    Returns:
        Float32 tensor of shape ``[batch, h, w, channels]``.
    """
    g = torch.Generator(device).manual_seed(seed)
    return torch.randn(batch, h, w, channels, generator=g, device=device)
