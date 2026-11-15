import comfy.model_management as mm
import comfy.model_patcher
import comfy.sample
import folder_paths
import torch
import torch.nn.functional as F

from pneumam import models
from pneumam.models.anima import fused
from pneumam.models.anima.text import TextEncoder
from pneumam.models.vae import load_decoder
from pneumam.models.vae.latents import from_bchw, normalize, to_bchw
from pneumam.runtime import Runtime
from pneumam.sampling.loop import SAMPLERS, sample

from .model import LORAS, Denoiser, PneumamConfig, PneumamModel

CATEGORY = "Pneumam"
DTYPE = torch.bfloat16
TEXT_REPO = "circlestone-labs/Anima-Base-v1.0-Diffusers"
TEXT_COMPONENTS = ["tokenizer", "text_encoder", "t5_tokenizer", "text_conditioner"]


class LoadModel:
    """Load a Pneumam checkpoint as a ComfyUI model."""

    CATEGORY = CATEGORY
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "checkpoint": (folder_paths.get_filename_list("diffusion_models"),),
                "fp8": ("BOOLEAN", {"default": False}),
                "cuda_graphs": ("BOOLEAN", {"default": True}),
            }
        }

    def load(self, checkpoint: str, fp8: bool, cuda_graphs: bool):
        device = mm.get_torch_device()
        path = folder_paths.get_full_path_or_raise("diffusion_models", checkpoint)
        dit = models.load(path, device, DTYPE)
        rt = Runtime()
        if fp8:
            fused.enable_fp8(dit, rt)
        config = PneumamConfig(PneumamConfig.unet_config)
        config.set_inference_dtype(DTYPE, None)
        model = PneumamModel(config, Denoiser(dit, rt, cuda_graphs), device=device)
        patcher = comfy.model_patcher.ModelPatcher(
            model, load_device=device, offload_device=mm.unet_offload_device()
        )
        return (patcher,)


class LoadTextEncoder:
    """Load the Anima text encoder: Qwen3 and the conditioner that maps it onto T5 ids."""

    CATEGORY = CATEGORY
    RETURN_TYPES = ("PNEUMAM_TEXT",)
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"repo": ("STRING", {"default": TEXT_REPO})}}

    def load(self, repo: str):
        from diffusers.modular_pipelines.modular_pipeline import ModularPipeline

        parts = ModularPipeline.from_pretrained(repo)
        parts.load_components(TEXT_COMPONENTS, dtype=DTYPE)
        parts.to(mm.get_torch_device())
        return (TextEncoder.from_pipeline(parts),)


class TextEncode:
    """Encode a prompt into conditioning for a Pneumam model."""

    CATEGORY = CATEGORY
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "encoder": ("PNEUMAM_TEXT",),
                "length": ("INT", {"default": 512, "min": 1, "max": 4096}),
            }
        }

    def encode(self, text: str, encoder: TextEncoder, length: int):
        ctx = encoder.encode(text, DTYPE)
        if ctx.shape[1] > length:
            raise ValueError(f"the prompt has {ctx.shape[1]} tokens, more than {length}")
        # one padded length lets the positive and negative prompts share a batch
        ctx = F.pad(ctx, (0, 0, 0, length - ctx.shape[1]))
        return ([[ctx, {}]],)


class LoadVAE:
    """Load a VAE file written by Pneumam's converter."""

    CATEGORY = CATEGORY
    RETURN_TYPES = ("PNEUMAM_VAE",)
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"vae": (folder_paths.get_filename_list("vae"),)}}

    def load(self, vae: str):
        path = folder_paths.get_full_path_or_raise("vae", vae)
        return (load_decoder(path, mm.get_torch_device(), DTYPE),)


class VAEDecode:
    """Decode latents with Pneumam's decoder, replayed as a CUDA graph per shape."""

    CATEGORY = CATEGORY
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "vae": ("PNEUMAM_VAE",),
                "cuda_graph": ("BOOLEAN", {"default": True}),
            }
        }

    def decode(self, samples, vae, cuda_graph: bool):
        z = samples["samples"]
        z = to_bchw(z, "BCTHW") if z.dim() == 5 else z
        device = mm.get_torch_device()
        # ComfyUI hands over raw VAE latents; the decoder takes normalized ones
        x = normalize(z.to(device), vae.mean, vae.std).contiguous()
        with torch.inference_mode():
            img = vae.decode(x, "BCHW", cuda_graph)
        img = ((img.float() + 1) / 2).clamp(0, 1).permute(0, 2, 3, 1)
        return (img.cpu(),)


class Sampler:
    """Text to image with Pneumam's own sampling loop.

    The whole run happens inside Pneumam: the text context and modulation are prepared once and
    each step replays as a CUDA graph. The noise is ComfyUI's, so a seed starts from the same
    noise as in KSampler.
    """

    CATEGORY = CATEGORY
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 1000}),
                "cfg": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler": (list(SAMPLERS),),
                "shift": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "cuda_graphs": ("BOOLEAN", {"default": True}),
            }
        }

    def sample(self, model, positive, negative, latent_image, seed, steps, cfg, sampler, shift, cuda_graphs):
        denoiser = model.model.diffusion_model
        if not isinstance(denoiser, Denoiser):
            raise ValueError("Pneumam KSampler needs a model from Load Pneumam Model")
        mm.load_models_gpu([model])
        denoiser.sync(model.model_options.get("transformer_options", {}).get(LORAS, ()))
        z = comfy.sample.fix_empty_latent_channels(model, latent_image["samples"])
        noise = comfy.sample.prepare_noise(z, seed, latent_image.get("batch_index"))
        frames = noise.dim() == 5
        device = mm.get_torch_device()
        layout = denoiser.dit.spec.layout
        x = from_bchw(to_bchw(noise, "BCTHW") if frames else noise, layout)
        x = x.to(device, torch.float32).contiguous()
        batch = x.shape[0]

        def text(c):
            return c[0][0].to(device, DTYPE).expand(batch, -1, -1).contiguous()

        out = sample(
            denoiser.dit,
            denoiser.rt,
            text(positive),
            None if cfg == 1.0 else text(negative),
            x,
            steps,
            cfg,
            shift=shift,
            sampler=sampler,
            seeds=[seed + i for i in range(batch)],
            graph=cuda_graphs,
        )
        # Wan21's statistics are shaped for [B, C, T, H, W], so convert in that layout
        raw = model.model.process_latent_out(to_bchw(out, layout).float()[:, :, None])
        raw = raw if frames else raw[:, :, 0]
        return ({**latent_image, "samples": raw.cpu()},)


class LoRA:
    """Add a LoRA to a Pneumam model. Chain the node to stack several.

    The base model is not changed: the LoRAs are merged when a run uses this model, and
    replaced when a run uses another set. PEFT, kohya, ComfyUI and Pneumam files are read.
    """

    CATEGORY = CATEGORY
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "lora": (folder_paths.get_filename_list("loras"),),
                "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            }
        }

    def apply(self, model, lora: str, strength: float):
        if not isinstance(model.model.diffusion_model, Denoiser):
            raise ValueError("Pneumam LoRA needs a model from Load Pneumam Model")
        path = folder_paths.get_full_path_or_raise("loras", lora)
        out = model.clone()
        opts = dict(out.model_options.get("transformer_options", {}))
        opts[LORAS] = tuple(opts.get(LORAS, ())) + ((path, strength),)
        out.model_options["transformer_options"] = opts
        return (out,)


NODE_CLASS_MAPPINGS = {
    "PneumamLoRA": LoRA,
    "PneumamSampler": Sampler,
    "PneumamLoadModel": LoadModel,
    "PneumamLoadTextEncoder": LoadTextEncoder,
    "PneumamTextEncode": TextEncode,
    "PneumamLoadVAE": LoadVAE,
    "PneumamVAEDecode": VAEDecode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "PneumamSampler": "Pneumam KSampler",
    "PneumamLoRA": "Pneumam LoRA",
    "PneumamLoadModel": "Load Pneumam Model",
    "PneumamLoadTextEncoder": "Load Pneumam Text Encoder",
    "PneumamTextEncode": "Pneumam Text Encode",
    "PneumamLoadVAE": "Load Pneumam VAE",
    "PneumamVAEDecode": "Pneumam VAE Decode",
}
