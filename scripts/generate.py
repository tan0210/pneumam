import argparse
import os
import time
from typing import cast

import torch
from PIL import Image
from safetensors.torch import load_file

from pneumam import models
from pneumam.adapters.lora import merge_lora, read_lora
from pneumam.models.anima import fused
from pneumam.models.anima.convert import convert_lora
from pneumam.models.anima.dit import AnimaDiT
from pneumam.models.anima.text import TextEncoder
from pneumam.models.vae import Wan21Decoder, load_decoder
from pneumam.pipeline import Pipeline, to_uint8
from pneumam.runtime import Runtime
from pneumam.sampling.loop import SAMPLERS
from pneumam.sampling.samplers.er_sde import NOISE_SCALES

REPO = "circlestone-labs/Anima-Base-v1.0-Diffusers"
PROMPT = (
    "masterpiece, best quality, score_7, safe. An anime girl wearing a black tank-top and denim "
    "shorts is standing outdoors. She's holding a rectangular sign out in front of her that reads "
    '"ANIMA". She\'s looking at the viewer with a smile. The background features some trees and '
    "blue sky with clouds."
)
NEGATIVE = "worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts, sepia"


def add_args(ap: argparse.ArgumentParser) -> None:
    """Add the model and sampling options.

    Args:
        ap: Parser to extend.
    """
    ap.add_argument(
        "--checkpoint", required=True, help="Model checkpoint from convert.py."
    )
    ap.add_argument(
        "--vae", help="VAE file from convert.py. Default: the VAE of --repo."
    )
    ap.add_argument(
        "--repo", default=REPO, help="Diffusers repo with the text encoder."
    )
    ap.add_argument(
        "--lora",
        action="append",
        default=[],
        metavar="PATH[:STRENGTH]",
        help="Adapter to merge, in this library's format or a third-party one. Repeatable.",
    )
    ap.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fp8", action="store_true", help="Run the projections in FP8.")
    ap.add_argument("--fp8-fast-accum", action="store_true")
    ap.add_argument(
        "--trim-context",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip padded text tokens in cross-attention. It does not change the image.",
    )
    ap.add_argument(
        "--graph",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Replay each step and the decode as CUDA graphs. The first batch captures them.",
    )
    ap.add_argument(
        "--context-length",
        type=int,
        default=None,
        help="Pad text embeddings to this many tokens. 512 matches Diffusers.",
    )
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--negative", default=NEGATIVE)
    ap.add_argument("--width", type=int, default=1024, help="Pixels, multiple of 16.")
    ap.add_argument("--height", type=int, default=1024, help="Pixels, multiple of 16.")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument(
        "--cfg", type=float, default=4.0, help="Guidance scale. 1 disables it."
    )
    ap.add_argument("--shift", type=float, default=None, help="Default: the model's.")
    ap.add_argument("--sampler", choices=SAMPLERS, default="euler")
    ap.add_argument("--max-stage", type=int, choices=[1, 2, 3], default=3)
    ap.add_argument("--s-noise", type=float, default=1.0)
    ap.add_argument("--noise-scale", choices=NOISE_SCALES, default="er")
    ap.add_argument("--seed", type=int, default=0, help="Seed of the first image.")


def read_adapter(spec: str, architecture: str) -> tuple[dict[str, torch.Tensor], float]:
    """Read one ``--lora`` argument.

    Args:
        spec: ``PATH`` or ``PATH:STRENGTH``.
        architecture: The model family the adapter must target.

    Returns:
        The adapter's tensors in this library's layout, and its strength.

    Raises:
        SystemExit: If the file holds no adapter tensors this model supports.
    """
    path, strength = spec, 1.0
    head, sep, tail = spec.rpartition(":")
    if sep and not os.path.exists(spec):
        path, strength = head, float(tail)
    try:
        return read_lora(path, architecture)[0], strength
    except ValueError:
        sd, skipped = convert_lora(load_file(path))  # a third-party file
    if not sd:
        raise SystemExit(f"{path} holds no adapter tensors for this model")
    if skipped:
        print(f"{path}: skipped {len(skipped)} unsupported tensors")
    return sd, strength


def load_pipeline(args: argparse.Namespace) -> Pipeline:
    """Load everything the options describe.

    Args:
        args: Options from ``add_args``.

    Returns:
        The pipeline, ready to generate.
    """
    from diffusers.modular_pipelines.modular_pipeline import ModularPipeline

    dtype = getattr(torch, args.dtype)
    model = cast(AnimaDiT, models.load(args.checkpoint, args.device, dtype))
    rt = Runtime(trim_context=args.trim_context)
    if args.lora:
        merge_lora(model, [read_adapter(s, model.spec.architecture) for s in args.lora])
    if args.fp8:
        fused.enable_fp8(
            model, rt, args.fp8_fast_accum
        )  # after merging: it copies the weights
    if args.vae:
        decoder = load_decoder(args.vae, args.device, dtype)
    else:
        parts = ModularPipeline.from_pretrained(args.repo)
        parts.load_components(["vae"], dtype=dtype)
        decoder = Wan21Decoder.from_diffusers(parts.vae).requires_grad_(False)
        decoder = decoder.to(args.device, dtype).eval()
    text = TextEncoder.from_repo(args.repo, args.device, dtype)
    if args.tune:
        rows = 1 if args.cfg == 1.0 else 2
        fused.warmup(
            model,
            rt,
            args.height // 8,
            args.width // 8,
            rows * args.batch,
            verbose=True,
        )
    return Pipeline(model, text, decoder, rt, args.context_length)


def main(argv: list[str] | None = None) -> int:
    """Generate images and save them as ``<seed>.png``.

    Image ``i`` of a run uses seed ``--seed + i``.

    Args:
        argv: Command-line arguments. None reads ``sys.argv``.

    Returns:
        0 on success.
    """
    ap = argparse.ArgumentParser(
        description="Generate images with a converted checkpoint."
    )
    add_args(ap)
    ap.add_argument("--count", type=int, default=1, help="Number of images.")
    ap.add_argument(
        "--tune",
        action="store_true",
        help="Tune the GPU kernels for this size and batch first. Slow the first time, then cached.",
    )
    ap.add_argument("--batch", type=int, default=1, help="Images denoised together.")
    ap.add_argument("--out", default="generate_out")
    args = ap.parse_args(argv)
    if args.count < 1 or args.batch < 1:
        raise SystemExit("--count and --batch must be at least 1")

    pipe = load_pipeline(args)
    negative = None if args.cfg == 1.0 else args.negative
    cond, uncond = pipe.embed(args.prompt, negative)
    os.makedirs(args.out, exist_ok=True)
    seeds = [args.seed + i for i in range(args.count)]
    t0 = time.perf_counter()
    for i in range(0, len(seeds), args.batch):
        chunk = seeds[i : i + args.batch]
        latents = pipe.denoise(
            cond,
            uncond,
            args.width,
            args.height,
            chunk,
            args.steps,
            args.cfg,
            shift=args.shift,
            sampler=args.sampler,
            max_stage=args.max_stage,
            s_noise=args.s_noise,
            noise_scale=args.noise_scale,
            graph=args.graph,
        )
        images = to_uint8(pipe.decode(latents, args.graph))
        for seed, image in zip(chunk, images, strict=True):
            path = os.path.join(args.out, f"{seed}.png")
            Image.fromarray(image.numpy()).save(path)
            print(path)
    dt = time.perf_counter() - t0
    print(
        f"{args.count} image(s) in {dt:.1f} s, {dt / args.count:.2f} s each with warmup"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
