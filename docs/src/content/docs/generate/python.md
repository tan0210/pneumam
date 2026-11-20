---
title: Python pipeline
description: Load the model, the text encoder and the decoder, and generate images from Python.
sidebar:
  order: 2
---

The pipeline loads three parts once and then generates as many images as you ask for. Import each part from its own module, because the top-level `pneumam` package doesn't re-export them.

```python
import torch
from pneumam.pipeline import Pipeline
from pneumam.runtime import Runtime
from pneumam.models import load
from pneumam.models.vae import load_decoder
from pneumam.models.anima.text import TextEncoder

model = load("checkpoints/anima.safetensors")
vae = load_decoder("checkpoints/wan21-vae.safetensors")
text = TextEncoder.from_repo(
    "circlestone-labs/Anima-Base-v1.0-Diffusers", "cuda", torch.bfloat16)

pipe = Pipeline(model, text, vae, Runtime(), context_length=512)
images = pipe.generate("a lighthouse at dusk", seeds=[0, 1, 2, 3], steps=30, scale=4.0)
pixels = pipe.to_uint8(images)
```

## Load the parts

| Function | What it loads |
|---|---|
| `load(path)` | Loads the image model from a converted checkpoint. |
| `load_decoder(path)` | Loads the decoder from the converted decoder file. |
| `TextEncoder.from_repo(repo, device, dtype)` | Downloads and loads the text encoder from Hugging Face. |

`load` and `load_decoder` put the model on `"cuda"` in `torch.bfloat16` unless you pass `device` and `dtype`.

## Generate images

```python
pipe.generate(prompt, negative="", width=1024, height=1024,
              seeds=(0,), steps=30, scale=4.0)
```

`generate` returns one image for each seed, as a tensor of shape `[len(seeds), 3, height, width]` with values from -1 to 1. Pass the result to `to_uint8` to get `[len(seeds), height, width, 3]` pixels on the CPU, ready to save.

Pass `negative=None` to skip guidance. Each step then takes about half as long, but the images follow the prompt less closely.

| Keyword | What it does |
|---|---|
| `sampler` | Chooses `"euler"` or `"er_sde"`. |
| `shift` | Shifts how the steps are spread between rough shapes and fine detail. |
| `max_stage`, `s_noise`, `noise_scale` | Tune the `er_sde` sampler. [Samplers and guidance](/generate/sampling/) describes them. |
| `graph` | Turns the speed-up for repeated images on or off. |

## Reuse an embedding

`generate` runs four steps in a row. You can call them yourself, for example to encode a prompt once and sample it with many seeds.

| Method | What it does |
|---|---|
| `embed(prompt, negative)` | Encodes the prompts. |
| `denoise(cond, uncond, width, height, seeds, steps, scale)` | Samples the images and returns them as latents. |
| `decode(latents)` | Turns latents into images. |
| `to_uint8(images)` | Turns images into 8-bit pixels on the CPU. |

## Make it faster

Pass a `Runtime` with `fp8=True` to make generation about twice as fast.

```python
pipe = Pipeline(model, text, vae, Runtime(fp8=True), context_length=512)
```
