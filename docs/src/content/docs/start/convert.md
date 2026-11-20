---
title: Convert checkpoints
description: Convert the Anima release into the files Pneumam loads.
sidebar:
  order: 2
---

Pneumam loads its own checkpoint files, so you convert the Anima release once before the first run. The conversion needs the `convert` extra.

```bash
python scripts/convert.py --out checkpoints
```

The script writes two files into `checkpoints`. `anima.safetensors` is the image model, and `wan21-vae.safetensors` is the decoder that turns its output into pixels. The text encoder isn't converted, because Pneumam downloads it from Hugging Face the first time it runs.

:::caution
Pneumam can't load ComfyUI's single-file Anima checkpoint. Use the files this script writes.
:::

## Options

| Flag | What it does |
|---|---|
| `--out` | Sets the folder the files are written to. The default is `checkpoints`. |
| `--repo` | Chooses which Hugging Face repository to convert. |
| `--dtype` | Sets the precision of the files: `keep`, `bfloat16` or `float32`. The default, `keep`, leaves it as released. |
| `--no-vae` | Skips the decoder file. |

If the Hugging Face repository asks for a login, set `HF_TOKEN` before you run the script. The [environment variables](/reference/environment/) page lists it with the others.
