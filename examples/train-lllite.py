import argparse
import os
import random
from typing import cast

import torch
import torch.distributed as dist
from common import learning_rate, load_control, load_state, render_previews, save_state

from pneumam import models
from pneumam.adapters.lllite import attach_lllite, new_lllite, save_lllite
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


def random_masks(n: int, h: int, w: int, rng: random.Random, device) -> torch.Tensor:
    m = torch.zeros(n, 1, h, w, device=device)
    for i in range(n):
        bh, bw = int(h * rng.uniform(0.2, 0.8)), int(w * rng.uniform(0.2, 0.8))
        y, x = rng.randrange(h - bh + 1), rng.randrange(w - bw + 1)
        m[i, :, y : y + bh, x : x + bw] = 1
    return m


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Train an Anima ControlNet-LLLite model. Launch with torchrun for several GPUs."
    )
    ap.add_argument("--model", required=True)
    ap.add_argument("--vae", required=True)
    ap.add_argument("--text", default="circlestone-labs/Anima-Base-v1.0-Diffusers")
    ap.add_argument(
        "--datasets",
        required=True,
        help="Dataset TOML with conditioning_data_dir in every subset.",
    )
    ap.add_argument(
        "--prompts", help="Preview prompts; --cn sets each one's control image."
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="anima-lllite")
    ap.add_argument(
        "--targets",
        default="self_attn_q_proj,self_attn_k_proj,self_attn_v_proj,mlp_layer1",
    )
    ap.add_argument("--inpainting", action="store_true")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=4, help="Per GPU.")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--caption-dropout", type=float, default=0.15)
    ap.add_argument("--context-length", type=int, default=512)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--preview-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rank, world, device = distributed.setup()
    main_rank = rank == 0
    os.makedirs(args.out, exist_ok=True)

    model = cast(AnimaDiT, models.load(args.model, device, DTYPE)).requires_grad_(False)
    rt = Runtime(checkpointing=True, fp8_attention_grad=True)
    lllite = new_lllite(
        model.cfg.dim,
        model.cfg.layers,
        tuple(args.targets.split(",")),
        in_ch=4 if args.inpainting else 3,
        masked_input=args.inpainting,
        seed=args.seed,
    ).to(device)
    control = attach_lllite(model, lllite)
    control.set_strength(1.0)
    params = list(lllite.parameters())
    if world > 1:
        distributed.broadcast_params(params)
    text = TextEncoder.from_repo(args.text, device, DTYPE)
    uncond = pad_context(text.encode([""], DTYPE), args.context_length)

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
    previews = load_previews(args.prompts) if args.prompts else []
    pipe = (
        Pipeline(
            model, text, load_decoder(args.vae, device, DTYPE), rt, args.context_length
        )
        if previews and main_rank
        else None
    )

    def set_preview_control(p) -> None:
        control.cond = None
        if p.control is not None:
            image = load_control(p.control, p.width, p.height, device)
            mask = torch.zeros_like(image[:, :1]) if args.inpainting else None
            control.cond = lllite.encode(image, mask)

    def save(step: int, final: bool = False) -> None:
        name = (
            f"{args.name}.safetensors"
            if final
            else f"{args.name}-{step:06d}.safetensors"
        )
        save_lllite(lllite, os.path.join(args.out, name))
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
        image = b.control.to(device, non_blocking=True).float() / 255  # type: ignore[union-attr]
        mask = (
            random_masks(image.shape[0], image.shape[2], image.shape[3], rng, device)
            if args.inpainting
            else None
        )
        control.cond = lllite.encode(
            image, mask
        )  # with gradients, so the image encoder trains too
        inputs = flow_batch(
            model,
            b.latents.to(device, non_blocking=True),
            context,
            uncond,
            None,
            "logit_normal",
            args.caption_dropout,
            generator,
        )
        loss = velocity_loss(model, rt, *inputs)
        loss.backward()
        if world > 1:
            distributed.average_grads(params)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        running += loss.detach()
        done = step + 1
        if main_rank and done % 10 == 0:
            print(
                f"step {done}/{args.steps}  loss {running.item() / 10:.4f}  lr {lr:.2e}",
                flush=True,
            )
            running.zero_()
        if (
            main_rank
            and args.save_every
            and done % args.save_every == 0
            and done < args.steps
        ):
            save(done)
        if pipe is not None and args.preview_every and done % args.preview_every == 0:
            render_previews(
                pipe,
                previews,
                os.path.join(args.out, "previews"),
                done,
                set_preview_control,
            )
    if main_rank:
        save(args.steps, final=True)
    distributed.shutdown()


if __name__ == "__main__":
    main()
