# Training examples

```
python examples/train-lora.py --model checkpoints/anima.safetensors --vae checkpoints/wan21-vae.safetensors \
    --datasets examples/dataset.toml --prompts examples/prompts.txt --out runs/mychar --fp8 --graph

torchrun --nproc-per-node 4 examples/train_lora.py ...same options...
```

Checkpoints come from `scripts/convert.py`. A run folder gets the encoded datasets under `cache`, the adapter every `--save-every` steps and at the end, previews under `previews`, and `state.pt`, which a rerun with the same `--out` resumes from. LoRAs are written in ComfyUI naming and LLLite models in kohya's format, so both load in stock ComfyUI and in Pneumam's nodes.
