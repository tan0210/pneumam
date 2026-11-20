---
title: LoRA and LLLite
description: Load, apply and remove LoRA and LLLite adapters from Python.
sidebar:
  order: 1
---

LoRA adapters change the model's style or teach it a character. LLLite adapters steer an image with a control image, such as a pose or a sketch. In ComfyUI, use the Pneumam LoRA and Pneumam LLLite nodes. This page covers Python.

## LoRA

Pneumam reads LoRA files from kohya, PEFT, ComfyUI and Pneumam itself. The LoRAs it trains use ComfyUI's naming, so stock ComfyUI loads them too.

```python
from pneumam.adapters.lora import read_lora, merge_lora, unmerge_lora

lora, meta = read_lora("loras/mychar.safetensors")
merge_lora(model, [(lora, 0.8)])   # apply at strength 0.8
# generate as usual
unmerge_lora(model)                # remove it again
```

`merge_lora` takes a list of `(lora, strength)` pairs, so you can apply several at once.

## LLLite

Pneumam reads LLLite files in kohya's sd-scripts format, such as kohya-ss/Anima-LLLite.

```python
from pneumam.adapters.lllite import load_lllite, attach_lllite, detach_lllite

lllite = load_lllite("model_patches/anima-lllite.safetensors")
control = attach_lllite(model, lllite)
# generate as usual
detach_lllite(model)
```

LLLite adds about a tenth of a second to each image on an RTX 5090.

To train your own adapters, see [Training](/adapters/training/).
