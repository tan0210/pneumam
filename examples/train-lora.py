import argparse
import os
import random
from typing import Any, cast

import torch
import torch.distributed as dist
from common import learning_rate, load_state, render_previews, save_state
from safetensors.torch import save_file

from pneumam import models
from pneumam.adapters.lora import add_lora, enable_fp8_base, lora_state_dict
from pneumam.graphs import GraphedStep
from pneumam.models.anima.convert import export_lora
from pneumam.models.anima.dit import AnimaDiT
from pneumam.models.anima.text import TextEncoder, pad_context
from pneumam.models.vae import load_decoder, load_encoder
from pneumam.pipeline import Pipeline
from pneumam.runtime import Runtime
from pneumam.training import distributed
from pneumam.training.data.config import build_datasets, load_datasets, mixed_batches
from pneumam.training.objectives.flow import flow_batch, velocity_loss
from pneumam.training.previews import load_previews

DTYPE = torch.bfloat16


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Train an Anima LoRA. Launch with torchrun for several GPUs."
    )
    ap.add_argument(
        "--model", required=True, help="Checkpoint from scripts/convert.py."
    )
    ap.add_argument("--vae", required=True, help="VAE file from scripts/convert.py.")
    ap.add_argument("--text", default="circlestone-labs/Anima-Base-v1.0-Diffusers")
    ap.add_argument("--datasets", required=True, help="Dataset TOML in kohya's format.")
    ap.add_argument("--prompts", help="Preview prompts in kohya's format.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="lora")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=2, help="Per GPU.")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--caption-dropout", type=float, default=0.1)
    ap.add_argument("--context-length", type=int, default=512)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--preview-every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--fp8", action="store_true", help="Run the frozen base products in FP8."
    )
    ap.add_argument(
        "--graph",
        action="store_true",
        help="Replay forward and backward as CUDA graphs.",
    )
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()

    rank, world, device = distributed.setup()
    main_rank = rank == 0
    os.makedirs(args.out, exist_ok=True)

    model = cast(AnimaDiT, models.load(args.model, device, DTYPE))
    rt = Runtime(checkpointing=True, fp8_attention_grad=True)
    params = add_lora(model, args.rank)
    if world > 1:
        distributed.broadcast_params(params)  # every rank starts from rank 0's adapters
    if args.fp8:
        enable_fp8_base(model)
    text = TextEncoder.from_repo(args.text, device, DTYPE)
    uncond = pad_context(text.encode([""], DTYPE), args.context_length)

    # rank 0 encodes the images once, the other ranks then read its cache
    datasets = load_datasets(args.datasets)
    cache = os.path.join(args.out, "cache")
    if main_rank:
        build_datasets(
            datasets,
            cache,
            load_encoder(args.vae, device, DTYPE),
            model.spec.layout,
            args.vae,
        )
    if world > 1:
        dist.barrier()
    sets, rules = build_datasets(
        datasets, cache, None, model.spec.layout, args.vae, encode=False
    )

    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01, fused=True)
    state = os.path.join(args.out, "state.pt")
    start = load_state(state, params, optimizer)
    batches = mixed_batches(sets, args.batch, args.seed + start, rank, world)
    generator = torch.Generator(device).manual_seed(args.seed * 1000 + rank + start)
    rng = random.Random(args.seed * 1000 + rank + start)

    def loss_fn(noisy, sigma, context, target):
        return velocity_loss(model, rt, noisy, sigma, context, target)

    step_fn = GraphedStep(loss_fn, params) if args.graph else None
    previews = load_previews(args.prompts) if args.prompts else []
    pipe = (
        Pipeline(
            model, text, load_decoder(args.vae, device, DTYPE), rt, args.context_length
        )
        if previews and main_rank
        else None
    )
    wb: Any = None
    if args.wandb and main_rank:
        import wandb  # pyright: ignore[reportMissingImports]

        wb = wandb
        wb.init(project="pneumam", name=args.name, dir=args.out, resume="allow")

    def save(step: int, final: bool = False) -> None:
        name = (
            f"{args.name}.safetensors"
            if final
            else f"{args.name}-{step:06d}.safetensors"
        )
        save_file(export_lora(lora_state_dict(model)), os.path.join(args.out, name))
        save_state(state, step, params, optimizer)

    model.train()
    running = torch.zeros((), device=device)
    for step in range(start, args.steps):
        lr = learning_rate(step, args.lr, args.warmup, args.steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        b = next(batches)
        captions = [
            rules[s].apply(c, rng) for c, s in zip(b.captions, b.subsets, strict=True)
        ]
        with torch.no_grad():
            context = pad_context(text.encode(captions, DTYPE), args.context_length)
        latents = b.latents.to(device, non_blocking=True)
        inputs = flow_batch(
            model,
            latents,
            context,
            uncond,
            None,
            "logit_normal",
            args.caption_dropout,
            generator,
        )
        if step_fn is not None:
            loss = step_fn(*inputs)  # leaves this step's gradients in .grad
        else:
            loss = loss_fn(*inputs)
            loss.backward()
        if world > 1:
            distributed.average_grads(params)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        if step_fn is None:
            optimizer.zero_grad(set_to_none=True)
        running += loss.detach()
        done = step + 1
        if main_rank and done % 10 == 0:
            mean = running.item() / 10  # the only host sync of the loop
            running.zero_()
            print(f"step {done}/{args.steps}  loss {mean:.4f}  lr {lr:.2e}", flush=True)
            if wb is not None:
                wb.log({"loss": mean, "lr": lr}, step=done)
        if (
            main_rank
            and args.save_every
            and done % args.save_every == 0
            and done < args.steps
        ):
            save(done)
        if pipe is not None and args.preview_every and done % args.preview_every == 0:
            images = render_previews(
                pipe, previews, os.path.join(args.out, "previews"), done
            )
            if wb is not None:
                wb.log({"previews": [wb.Image(im) for im in images]}, step=done)
    if main_rank:
        save(args.steps, final=True)
    distributed.shutdown()


if __name__ == "__main__":
    main()
