from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from .models.vae.latents import from_bchw
from .runtime import Runtime
from .sampling.loop import sample
from .sampling.schedules import noise


def to_uint8(images: torch.Tensor) -> torch.Tensor:
    """Convert decoded images to 8-bit pixels.

    Args:
        images: Images in ``[-1, 1]``, shape ``[B, 3, H, W]``.

    Returns:
        A ``uint8`` tensor of shape ``[B, H, W, 3]`` on the CPU.
    """
    x = (images.float() / 2 + 0.5).clamp(0, 1)
    return (x * 255).round().byte().permute(0, 2, 3, 1).cpu()


@dataclass
class Pipeline:
    """Text to image: encode the prompts, denoise a latent, decode it.

    Attributes:
        model: A model from ``load``.
        text: The model's text encoder. Its ``encode(prompts, dtype)`` returns text embeddings
            of shape ``[B, S, text_dim]``, zero-padded to the longest prompt.
        decoder: The VAE decoder for the model's latents. See ``Wan21Decoder``.
        rt: The model's runtime. Set FP8 and context trimming on it.
        context_length: Zero-pad text embeddings to this many tokens. The model attends to the
            padding, so the padded length changes the image. None pads a prompt only to the
            length of its negative prompt.
    """

    model: Any
    text: Any
    decoder: Any
    rt: Runtime = field(default_factory=Runtime)
    context_length: int | None = None

    def embed(
        self, prompt: str, negative: str | None = ""
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Encode a prompt and its negative prompt.

        Args:
            prompt: The positive prompt.
            negative: The negative prompt. None encodes nothing, for sampling without guidance.

        Returns:
            ``(cond, uncond)``, each of shape ``[1, S, text_dim]`` with the same ``S``, in the
            model's compute dtype. ``uncond`` is None if ``negative`` is.

        Raises:
            ValueError: If a prompt is longer than ``context_length``.
        """
        prompts = [prompt] if negative is None else [prompt, negative]
        text = self.text.encode(prompts, self.model.compute_dtype)
        if self.context_length is not None:
            extra = self.context_length - text.shape[1]
            if extra < 0:
                raise ValueError(
                    f"a prompt has {text.shape[1]} tokens, more than the context length "
                    f"{self.context_length}"
                )
            text = F.pad(text, (0, 0, 0, extra))
        if negative is None:
            return text, None
        cond, uncond = text.split(1)
        return cond, uncond

    def denoise(
        self,
        cond: torch.Tensor,
        uncond: torch.Tensor | None,
        width: int,
        height: int,
        seeds: Sequence[int],
        steps: int = 30,
        scale: float = 4.0,
        **options: Any,
    ) -> torch.Tensor:
        """Denoise one latent per seed, all for the same prompt.

        Args:
            cond: Positive text embeddings from ``embed``, shape ``[1, S, text_dim]``.
            uncond: Negative text embeddings from ``embed``, or None to sample without guidance.
            width: Image width in pixels.
            height: Image height in pixels.
            seeds: One noise seed per image. The result for a seed does not depend on the others.
            steps: Number of sampler steps.
            scale: Guidance scale. Ignored without ``uncond``.
            **options: Further arguments of ``sample``: ``shift``, ``sampler``, ``max_stage``,
                ``s_noise``, ``noise_scale`` and ``graph``.

        Returns:
            Float32 latents in the model's latent layout, one per seed.

        Raises:
            ValueError: If ``width`` or ``height`` is not a positive multiple of the pixels one
                token covers, or ``seeds`` is empty. See ``sample`` for the rest.
        """
        spec, factor = self.model.spec, self.decoder.scale_factor
        cell = factor * spec.patch_size
        if width <= 0 or height <= 0 or width % cell or height % cell:
            raise ValueError(
                f"width and height must be positive multiples of {cell}, got {width}x{height}"
            )
        if not seeds:
            raise ValueError("need at least one seed")
        device = cond.device
        h, w, channels = height // factor, width // factor, spec.latent_channels
        x = torch.cat([noise(1, h, w, channels, int(s), device) for s in seeds])
        x = from_bchw(x.permute(0, 3, 1, 2), spec.layout).contiguous()
        batch = len(seeds)
        cond = cond.expand(batch, -1, -1).contiguous()
        if uncond is not None:
            uncond = uncond.expand(batch, -1, -1).contiguous()
        return sample(
            self.model, self.rt, cond, uncond, x, steps, scale, seeds=seeds, **options
        )

    def decode(self, latents: torch.Tensor, graph: bool = False) -> torch.Tensor:
        """Decode latents to images.

        Args:
            latents: Latents from ``denoise``.
            graph: Replay the decode as a CUDA graph. See ``Wan21Decoder.decode``.

        Returns:
            Images of shape ``[B, 3, height, width]`` in ``[-1, 1]``, on the latents' device.
        """
        return self.decoder.decode(latents, self.model.spec.layout, graph)

    def generate(
        self,
        prompt: str,
        negative: str | None = "",
        width: int = 1024,
        height: int = 1024,
        seeds: Sequence[int] = (0,),
        steps: int = 30,
        scale: float = 4.0,
        **options: Any,
    ) -> torch.Tensor:
        """Generate one image per seed for a prompt.

        Args:
            prompt: The positive prompt.
            negative: The negative prompt. None samples without guidance, at half the cost.
            width: Image width in pixels.
            height: Image height in pixels.
            seeds: One noise seed per image. The image for a seed does not depend on the others.
            steps: Number of sampler steps.
            scale: Guidance scale. Ignored without ``negative``.
            **options: Further arguments of ``sample``: ``shift``, ``sampler``, ``max_stage``,
                ``s_noise``, ``noise_scale`` and ``graph``. ``graph`` also applies to the decode.

        Returns:
            Images of shape ``[len(seeds), 3, height, width]`` in ``[-1, 1]``, on the model's
            device. See ``to_uint8``.

        Raises:
            ValueError: If an argument is invalid. See ``denoise``.
        """
        cond, uncond = self.embed(prompt, negative)
        latents = self.denoise(
            cond, uncond, width, height, seeds, steps, scale, **options
        )
        return self.decode(latents, bool(options.get("graph", False)))
