# Pneumam nodes for ComfyUI

Runs the [Anima](https://huggingface.co/circlestone-labs/Anima) image model in ComfyUI on Pneumam's kernels. With FP8 enabled it generates about twice as fast as ComfyUI's own Anima support on an RTX 5090, with the same images.

The model is its own ComfyUI model type. Stock nodes that take a `MODEL`, such as KSampler, the samplers, schedulers and CFG, work with it. Nodes written for ComfyUI's built-in Anima, such as Apply Anima LLLite or regional prompting, do not attach to it. The Pneumam nodes below cover LoRA and LLLite instead.

## Install

1. Install Pneumam with the ComfyUI dependencies into the Python environment that runs ComfyUI:

   ```
   uv sync --extra comfyui --extra sm120   # RTX 50xx / RTX PRO 6000 Blackwell
   uv sync --extra comfyui --extra sm90    # H100 / H200
   ```

2. Link this folder into ComfyUI's `custom_nodes`:

   ```
   ln -s /path/to/Pneumam/comfyui /path/to/ComfyUI/custom_nodes/pneumam_nodes
   ```

   On Windows, use `mklink /D` or copy the folder.

3. Start ComfyUI with these flags:

   ```
   python main.py --highvram --disable-cuda-malloc
   ```

   `--highvram` keeps the model fully on the GPU. `--disable-cuda-malloc` is required: ComfyUI's default `cudaMallocAsync` allocator does not work with the CUDA graphs these nodes use, and generation fails with "Pointer argument cannot be accessed from Triton".

## Model files

Convert the Diffusers release once:

```
python scripts/convert.py --out checkpoints
```

This writes `checkpoints/anima.safetensors` and `checkpoints/wan21-vae.safetensors`. Put them in ComfyUI's `models/diffusion_models` and `models/vae` folders. The text encoder is downloaded from Hugging Face on first use.

LoRA files go in `models/loras`. LLLite files, such as [kohya-ss/Anima-LLLite](https://huggingface.co/kohya-ss/Anima-LLLite), go in `models/model_patches`.

## Nodes

| Node                      | What it does                                                                                                                                                                      |
| ------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Load Pneumam Model        | Loads a converted checkpoint as a `MODEL`. `fp8` runs the projections in FP8, which is about twice as fast. `cuda_graphs` replays each step as a CUDA graph.                      |
| Load Pneumam Text Encoder | Loads Qwen3 and the adapter that maps it onto the model's text space, from a Diffusers repository.                                                                                |
| Pneumam Text Encode       | Turns a prompt into `CONDITIONING`. Keep `length` the same for the positive and negative prompt, 512 by default, so they run as one batch.                                        |
| Pneumam KSampler          | Text to image with Pneumam's own sampling loop: `euler` or `er_sde`. Same inputs as KSampler, and a seed starts from the same noise. Stock KSampler works too and is as fast.     |
| Pneumam LoRA              | Adds a LoRA. Chain it to stack several. Reads PEFT, kohya, ComfyUI and Pneumam files.                                                                                             |
| Pneumam LLLite            | Adds ControlNet-LLLite control from an image, with strength and a start and end percent of the steps. Inpainting models also take a mask, where white marks the area to generate. |
| Load Pneumam VAE          | Loads a converted VAE file.                                                                                                                                                       |
| Pneumam VAE Decode        | Decodes latents to images, replayed as a CUDA graph. ComfyUI's own VAE Decode with the Qwen-Image VAE also works.                                                                 |

A minimal text-to-image graph:

1. Load Pneumam Model, optionally through Pneumam LoRA and Pneumam LLLite, into KSampler's `model`.
2. Load Pneumam Text Encoder into two Pneumam Text Encode nodes, one for `positive` and one for `negative`.
3. Empty Latent Image into `latent_image`.
4. KSampler's output and Load Pneumam VAE into Pneumam VAE Decode, then Save Image.

Use the scheduler `simple` and a guidance scale around 4.

## Speed

RTX 5090, 1024x1024, 30 Euler steps, guidance 4, after the first run:

| Setup                     | Seconds per image |
| ------------------------- | ----------------- |
| ComfyUI's Anima, bf16     | 6.2               |
| Pneumam, bf16             | 5.8               |
| Pneumam, FP8              | 3.3               |
| Pneumam, FP8, with LLLite | 3.4               |

The first run at a new size takes a few seconds longer while the CUDA graphs are captured.

## Limitations

- Pneumam KSampler always starts from pure noise. For image to image, use KSampler with `denoise` below 1.
- The loader reads checkpoints from `scripts/convert.py`, not ComfyUI's single-file Anima checkpoint.
- A LoRA keeps a copy of the original weights it changes on the GPU, up to the size of the model for a LoRA that covers every layer.
