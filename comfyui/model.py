from typing import Any

import comfy.conds
import comfy.latent_formats
import comfy.model_base
import comfy.supported_models_base
import torch
from torch import nn

from safetensors.torch import load_file

from pneumam.adapters.lora import merge_lora, read_lora
from pneumam.graphs import Graphed
from pneumam.models.anima import fused
from pneumam.models.anima.convert import convert_lora
from pneumam.models.anima.dit import AnimaDiT
from pneumam.runtime import Runtime


LORAS = "pneumam_loras"


def read_adapter(path: str) -> dict[str, torch.Tensor]:
    """Read a LoRA file in Pneumam's format or a third-party one.

    Args:
        path: The file.

    Returns:
        The adapter's tensors in Pneumam's layout.

    Raises:
        ValueError: If the file holds no adapter tensors for Anima.
    """
    try:
        return read_lora(path, "anima")[0]
    except ValueError:
        sd, _ = convert_lora(load_file(path))
    if not sd:
        raise ValueError(f"{path} holds no adapter tensors for Anima")
    return sd


class PneumamConfig(comfy.supported_models_base.BASE):
    """ComfyUI model config for a Pneumam Anima model.

    It is its own model type, so ComfyUI never treats the model as its built-in Anima.
    """

    unet_config: dict[str, Any] = {"disable_unet_model_creation": True}
    sampling_settings = {"multiplier": 1.0, "shift": 3.0}
    latent_format = comfy.latent_formats.Wan21
    memory_usage_factor = 1.0
    supported_inference_dtypes = [torch.bfloat16]


class Denoiser(nn.Module):
    """Runs a Pneumam model behind ComfyUI's diffusion model interface.

    Each latent shape is captured as a CUDA graph on first use and replayed afterwards. A new
    prompt of the same padded length only refills the captured text context.

    Args:
        dit: The model.
        rt: Its runtime, with FP8 and context trimming already set.
        graph: Whether to replay steps as CUDA graphs.
    """

    def __init__(self, dit: AnimaDiT, rt: Runtime, graph: bool = True) -> None:
        super().__init__()
        self.dit, self.rt, self.graph = dit, rt, graph
        self.dtype = dit.compute_dtype
        self._ctx_in: torch.Tensor | None = None
        self._ctx = None
        self._where: int | None = None
        self._loras: tuple = ()
        self._backup: dict[str, torch.Tensor] | None = None
        self._adapters: dict[str, dict[str, torch.Tensor]] = {}

    def _context(self, context: torch.Tensor):
        cached = self._ctx_in
        if cached is None or cached.shape != context.shape or not torch.equal(cached, context):
            self._ctx_in = context.clone()
            self._ctx = self.dit.encode_context(self.rt, context.to(self.dtype))
        return self._ctx

    def _reset(self) -> None:
        self.rt.graphs.clear()
        self._ctx_in = self._ctx = None

    def sync(self, loras: tuple = ()) -> None:
        """Bring the model in line with the LoRAs a run asks for.

        The requested set replaces the merged one, starting from the original weights. FP8
        weights are quantized again and CUDA graphs captured again when anything changed,
        including ComfyUI moving the weights.

        Args:
            loras: ``(path, strength)`` pairs, as the Pneumam LoRA node records them.
        """
        where = self.dit.patch.weight.data_ptr()
        if where != self._where:
            self._where = where
            self._reset()
        loras = tuple(loras)
        if loras == self._loras:
            return
        for path, _ in loras:
            if path not in self._adapters:
                self._adapters[path] = read_adapter(path)
        if loras or self._backup:
            pairs = [(self._adapters[path], strength) for path, strength in loras]
            self._backup = merge_lora(self.dit, pairs, self._backup)
        if self.rt.fp8 is not None:
            fused.enable_fp8(self.dit, self.rt, self.rt.fp8_fast_accum)  # it copies the weights
        self._loras = loras
        self._reset()

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Predict the velocity.

        Args:
            x: Noisy latents, shape ``[B, C, 1, H, W]`` or ``[B, C, H, W]``.
            timesteps: Noise levels (sigmas), shape ``[B]``.
            context: Text context from the Pneumam text encode node, shape ``[B, S, D]``.
            **kwargs: Other ComfyUI conditioning. Only the LoRAs in ``transformer_options``
                are used.

        Returns:
            The velocity in the shape and dtype of ``x``.
        """
        if context is None:
            raise ValueError("Pneumam models need conditioning from Pneumam Text Encode")
        self.sync(kwargs.get("transformer_options", {}).get(LORAS, ()))
        frames = x.dim() == 5
        z = (x[:, :, 0] if frames else x).permute(0, 2, 3, 1).to(self.dtype).contiguous()
        t = timesteps.float().reshape(-1)
        ctx = self._context(context)
        with torch.inference_mode():
            if self.graph:
                v = self._replay(z, t, ctx)
            else:
                v = fused.forward(self.dit, self.rt, z, t, ctx)
        v = v.permute(0, 3, 1, 2)
        return (v[:, :, None] if frames else v).to(x.dtype)

    def _replay(self, z: torch.Tensor, t: torch.Tensor, ctx) -> torch.Tensor:
        key = ("comfy", tuple(z.shape), ctx.key())
        runner = self.rt.graphs.get(key)
        if runner is None:
            dit, rt = self.dit, self.rt

            def step(z, t, *tensors):
                return fused.forward(dit, rt, z, t, ctx.replace(tensors))

            runner = Graphed(step, (z, t), tuple(ctx.tensors()))
            self.rt.graphs[key] = runner
            runner.bound = ctx  # type: ignore[attr-defined]
        elif runner.bound is not ctx:
            runner.rebind(*ctx.tensors())
            runner.bound = ctx
        return runner(z, t)


class PneumamModel(comfy.model_base.BaseModel):
    """ComfyUI model wrapping a Pneumam Anima model.

    Args:
        config: Its ``PneumamConfig``.
        denoiser: The ``Denoiser`` that runs the model.
        device: Device the model runs on.
    """

    def __init__(self, config: PneumamConfig, denoiser: Denoiser, device=None) -> None:
        super().__init__(config, comfy.model_base.ModelType.FLOW, device=device)
        self.diffusion_model = denoiser

    def extra_conds(self, **kwargs: Any) -> dict[str, Any]:
        """Pass the text context through without ComfyUI's cross-attention padding.

        Args:
            **kwargs: The conditioning.

        Returns:
            The conds handed to the denoiser.
        """
        out = {}
        cross = kwargs.get("cross_attn")
        if cross is not None:
            out["c_crossattn"] = comfy.conds.CONDRegular(cross)
        return out
