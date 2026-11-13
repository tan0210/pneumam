from collections.abc import Callable

import torch


def guided(
    run: Callable[..., torch.Tensor], scale: float
) -> Callable[..., torch.Tensor]:
    """Wrap a velocity function with classifier-free guidance.

    Args:
        run: Maps ``(x, *state)`` to a velocity, for a batch whose first half is conditioned on
            the positive prompts and whose second half on the negative ones.
        scale: Guidance scale. 1 returns the positive prediction.

    Returns:
        A function of ``(z, *state)`` with ``z`` half the batch size of ``run``, returning the
        guided velocity for ``z``.
    """

    def fn(z: torch.Tensor, *state: torch.Tensor) -> torch.Tensor:
        vc, vu = run(torch.cat([z, z]), *state).chunk(2)
        return vu + scale * (vc - vu)

    return fn
