from collections.abc import Callable

import torch


def euler(
    velocity: Callable[[int, torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    sg: torch.Tensor,
) -> torch.Tensor:
    """Integrate the flow with Euler steps.

    Args:
        velocity: Maps ``(step index, latent)`` to the predicted velocity, shaped like the latent.
        x: Initial noise latent.
        sg: Noise schedule from ``sigmas``, on the device of ``x``.

    Returns:
        The denoised latent.
    """
    for i in range(len(sg) - 1):
        x = x + (sg[i + 1] - sg[i]) * velocity(i, x)
    return x
