from collections.abc import Sequence
from typing import Any

import torch

from ..graphs import Graphed
from ..runtime import Runtime
from .guidance import guided
from .samplers.er_sde import er_sde
from .samplers.euler import euler
from .schedules import sigmas

SAMPLERS = ("euler", "er_sde")


@torch.no_grad()
def sample(
    model: Any,
    rt: Runtime,
    cond: torch.Tensor,
    uncond: torch.Tensor | None,
    x: torch.Tensor,
    steps: int,
    scale: float = 4.0,
    shift: float | None = None,
    sampler: str = "euler",
    max_stage: int = 3,
    s_noise: float = 1.0,
    seeds: Sequence[int] | None = None,
    noise_scale: str = "er",
    graph: bool = False,
) -> torch.Tensor:
    """Denoise a latent, with classifier-free guidance if negative embeddings are given.

    Args:
        model: A model from ``load``.
        rt: Its runtime.
        cond: Positive text embeddings, shape ``[B, S, text_dim]``.
        uncond: Negative or empty text embeddings, shaped like ``cond``. None disables guidance
            and halves the cost of a step.
        x: Initial noise latent, float32, in the model's latent layout with batch size ``B``.
        steps: Number of sampler steps. Each costs one model evaluation.
        scale: Guidance scale. Ignored without ``uncond``.
        shift: Schedule shift. None uses the model's default.
        sampler: ``"euler"`` (deterministic) or ``"er_sde"`` (stochastic).
        max_stage: Highest solver order for ``"er_sde"``, 1 to 3.
        s_noise: Multiplier on the noise ``"er_sde"`` injects. Values below 1 shrink the result.
        seeds: One seed per batch element for the noise ``"er_sde"`` injects, so a sample does
            not depend on the rest of the batch. If None, the global torch RNG is used.
        noise_scale: For ``"er_sde"``: ``"er"`` (stochastic) or ``"ode"`` (its noise-free limit).
        graph: Replay each model evaluation as a CUDA graph. The first call for each combination
            of shapes, dtypes and guidance scale captures a graph, at the cost of about three
            extra evaluations, and later calls reuse it. Changes to the weights are not tracked,
            so clear ``rt.graphs`` after changing them.

    Returns:
        Denoised latent in float32, shaped like ``x``.

    Raises:
        ValueError: If ``sampler`` is unknown or ``max_stage``, ``seeds`` or ``noise_scale`` are
            invalid.
    """
    if sampler not in SAMPLERS:
        raise ValueError(f"unknown sampler {sampler!r}; expected one of {SAMPLERS}")
    sg = sigmas(steps, model.spec.shift if shift is None else shift)
    sg_dev = sg.to(x.device)
    text = cond if uncond is None else torch.cat([cond, uncond])
    prep = model.prepare(rt, sg_dev[:steps], text)
    fn = prep.run if uncond is None else guided(prep.run, scale)

    runner = None
    if graph:
        varying = (x, *prep.per_step(0))
        shapes = tuple((tuple(t.shape), t.dtype) for t in (*varying, *prep.fixed))
        key = (shapes, prep.key, None if uncond is None else scale)
        runner = rt.graphs.get(key)
        if runner is None:
            runner = rt.graphs[key] = Graphed(fn, varying, prep.fixed)
        runner.rebind(*prep.fixed)

    def velocity(i: int, z: torch.Tensor) -> torch.Tensor:
        if runner is not None:
            return runner(z, *prep.per_step(i))
        return fn(z, *prep.per_step(i), *prep.fixed)

    if sampler == "euler":
        return euler(velocity, x, sg_dev)
    return er_sde(velocity, x, sg, max_stage, s_noise, seeds, noise_scale)
