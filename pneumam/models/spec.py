from collections.abc import Callable, Hashable
from dataclasses import dataclass, field

import torch


@dataclass(frozen=True)
class ModelSpec:
    """What code outside a model family needs to know about a model.

    Attributes:
        architecture: Name of the model family, as written to checkpoints and adapter files.
        latent_channels: Channels of the latent the model denoises.
        patch_size: Side of the square latent patch that becomes one token.
        shift: Default shift of the sampling and training noise schedules.
        lora_targets: Names of the projections adapters can attach to, each with the number of
            equal output slices that are adapted independently.
        layout: Axis order of the latents the model takes and returns.
    """

    architecture: str
    latent_channels: int
    patch_size: int
    shift: float
    lora_targets: dict[str, int] = field(default_factory=dict)
    layout: str = "BHWC"


@dataclass
class Prepared:
    """A model readied for one sampling run: a schedule and a batch of text embeddings.

    Every tensor the run reads is listed here, so it can be replayed with different values.

    Attributes:
        run: Maps ``(x, *per_step(i), *fixed)`` to the float32 velocity at step ``i``. ``x`` is a
            latent of any floating dtype whose batch size matches the text embeddings.
        per_step: Maps a step index to the tensors that change with the step.
        fixed: The tensors that stay the same for the whole run.
        key: Everything else ``run`` depends on. Two runs may share a captured graph only if
            their keys are equal.
    """

    run: Callable[..., torch.Tensor]
    per_step: Callable[[int], tuple[torch.Tensor, ...]]
    fixed: tuple[torch.Tensor, ...] = ()
    key: Hashable = None
