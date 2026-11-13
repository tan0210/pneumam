from typing import Any

import torch
import torch.nn.functional as F

from ...runtime import Runtime

DISTS = ("logit_normal", "uniform")


def sample_sigmas(
    n: int,
    shift: float,
    dist: str = "logit_normal",
    device: str | torch.device | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Draw training noise levels.

    Args:
        n: Number of samples.
        shift: Schedule shift, as in ``sigmas``. Larger values favor high noise.
        dist: Distribution of the unshifted time, ``"logit_normal"`` or ``"uniform"``.
        device: Device of the result.
        generator: Random generator on ``device``. None uses the global one.

    Returns:
        Float32 tensor of shape ``[n]`` with values in ``(0, 1)``.

    Raises:
        ValueError: If ``dist`` is unknown.
    """
    if dist == "logit_normal":
        t = torch.sigmoid(torch.randn(n, device=device, generator=generator))
    elif dist == "uniform":
        t = torch.rand(n, device=device, generator=generator)
    else:
        raise ValueError(f"unknown dist {dist!r}; expected one of {DISTS}")
    return shift * t / (1 + (shift - 1) * t)


def flow_batch(
    model: Any,
    latents: torch.Tensor,
    context: torch.Tensor,
    uncond: torch.Tensor | None = None,
    shift: float | None = None,
    dist: str = "logit_normal",
    caption_dropout: float = 0.1,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Draw the random parts of one rectified-flow training batch.

    Args:
        model: A model from ``load``.
        latents: Clean latents in the model's latent layout, batch size ``B``.
        context: Text embeddings, shape ``[B, S, text_dim]``, zero-padded as at inference.
        uncond: Empty-prompt embeddings, shape ``[1, S, text_dim]`` with the same ``S``. Needed
            for caption dropout.
        shift: Schedule shift. None uses the model's default.
        dist: Noise-level distribution. See ``sample_sigmas``.
        caption_dropout: Probability of replacing a sample's context with ``uncond``.
        generator: Random generator on the latents' device. None uses the global one.

    Returns:
        ``(noisy, sigma, context, target)``: the noisy latents and the text embeddings in the
        model's compute dtype, the noise level of each sample, and the float32 velocity
        ``noise - latents`` the model should predict. See ``velocity_loss``.

    Raises:
        ValueError: If ``caption_dropout`` is positive and ``uncond`` is missing, or ``dist`` is
            unknown.
    """
    if caption_dropout > 0 and uncond is None:
        raise ValueError("caption dropout needs the empty-prompt embeddings")
    B, device, dtype = latents.shape[0], latents.device, model.compute_dtype
    x0 = latents.float()
    eps = torch.randn(x0.shape, device=device, generator=generator)
    level_shift: float = model.spec.shift if shift is None else shift
    sigma = sample_sigmas(B, level_shift, dist, device, generator)
    s = sigma.view(B, *([1] * (x0.ndim - 1)))
    xt = (1 - s) * x0 + s * eps
    if caption_dropout > 0 and uncond is not None:
        drop = torch.rand(B, device=device, generator=generator) < caption_dropout
        empty = uncond.to(context).expand_as(context)
        context = torch.where(drop[:, None, None], empty, context)
    return xt.to(dtype), sigma, context.to(dtype), eps - x0


def velocity_loss(
    model: Any,
    rt: Runtime,
    noisy: torch.Tensor,
    sigma: torch.Tensor,
    context: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Mean squared error of the model's velocity prediction. Differentiable.

    It draws no random numbers, so it can be captured as a CUDA graph. See ``GraphedStep``.

    Args:
        model: A model from ``load``.
        rt: Its runtime.
        noisy: Noisy latents from ``flow_batch``.
        sigma: Noise level of each sample, shape ``[B]``.
        context: Text embeddings from ``flow_batch``.
        target: The velocity to predict, float32, shaped like ``noisy``.

    Returns:
        Scalar float32 loss.
    """
    return F.mse_loss(model.velocity(rt, noisy, sigma, context).float(), target)


def flow_loss(
    model: Any,
    rt: Runtime,
    latents: torch.Tensor,
    context: torch.Tensor,
    uncond: torch.Tensor | None = None,
    shift: float | None = None,
    dist: str = "logit_normal",
    caption_dropout: float = 0.1,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Rectified-flow loss for one batch.

    The model learns to predict the velocity ``noise - latents`` at a random noise level. This
    is ``velocity_loss`` of a batch from ``flow_batch``.

    Args:
        model: A model from ``load``.
        rt: Its runtime.
        latents: Clean latents in the model's latent layout, batch size ``B``.
        context: Text embeddings, shape ``[B, S, text_dim]``, zero-padded as at inference.
        uncond: Empty-prompt embeddings, shape ``[1, S, text_dim]`` with the same ``S``. Needed
            for caption dropout.
        shift: Schedule shift. None uses the model's default.
        dist: Noise-level distribution. See ``sample_sigmas``.
        caption_dropout: Probability of replacing a sample's context with ``uncond``.
        generator: Random generator on the latents' device. None uses the global one.

    Returns:
        Scalar float32 mean squared error.

    Raises:
        ValueError: If ``caption_dropout`` is positive and ``uncond`` is missing, or ``dist`` is
            unknown.
    """
    batch = flow_batch(
        model, latents, context, uncond, shift, dist, caption_dropout, generator
    )
    return velocity_loss(model, rt, *batch)
