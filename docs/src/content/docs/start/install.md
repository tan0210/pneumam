---
title: Install
description: Install Pneumam with uv, with the extras for your GPU and for ComfyUI.
sidebar:
  order: 1
---

Pneumam is a Python package. It needs Python 3.12 or newer and an NVIDIA GPU, and it runs on Linux and Windows.

```bash
uv add git+https://github.com/tan0210/pneumam
```

:::note
Pneumam works best on an RTX 50 series card. The [supported GPUs](/reference/gpus/) are the RTX 50 series, the RTX PRO 6000 Blackwell, the H100 and the H200.
:::

## Extras

Add the extra that matches your GPU, then the extras for what you plan to do.

| Extra | When to add it |
|---|---|
| `sm120` | Add it on an RTX 50 series card or an RTX PRO 6000 Blackwell. |
| `sm89` | Add it on an RTX 40 series card, an L4 or an L40. |
| `sm86` | Add it on an RTX 30 series card, an A10 or an A40. |
| `sm90` | Add it on an H100 or H200. It is available on Linux only. |
| `comfyui` | Add it to use the ComfyUI nodes. |
| `convert` | Add it to convert the Anima release into Pneumam's files. |
| `cli` | Add it to generate images from the command line. |
| `train` | Add it to train LoRA and LLLite adapters. |
| `text` | Add it to load the text encoder from Python. |
| `vae` | Add it for faster image decoding. |
| `bench` | Add it to plot profiling results. |

## Install from a clone

The ComfyUI nodes are linked from a clone of the repository. Clone it and sync with the extras you need:

```bash
git clone https://github.com/tan0210/pneumam
cd pneumam
uv sync --extra comfyui --extra sm120   # RTX 50 series
uv sync --extra comfyui --extra sm90    # H100 or H200
```

## Next steps

Convert the model once with [Convert checkpoints](/start/convert/). After that, you can generate from [ComfyUI](/comfyui/nodes/), [the command line](/generate/command-line/) or [Python](/generate/python/).
