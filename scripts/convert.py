import argparse
import os

import torch

from pneumam.models.anima.convert import convert_checkpoint
from pneumam.models.vae import Wan21Decoder, Wan21Encoder, save_vae

REPO = "circlestone-labs/Anima-Base-v1.0-Diffusers"


def resolve(repo: str) -> str:
    """Find a Diffusers repository on disk, downloading it if it is a Hub id.

    Args:
        repo: A local directory, or a Hugging Face Hub repository id.

    Returns:
        A directory holding the ``transformer`` and ``vae`` subfolders.
    """
    if os.path.isdir(repo):
        return repo
    from huggingface_hub import snapshot_download

    return snapshot_download(repo, allow_patterns=["transformer/*", "vae/*"])


def main(argv: list[str] | None = None) -> int:
    """Convert a Diffusers Anima repository into a model checkpoint and a VAE file.

    Args:
        argv: Command-line arguments. None reads ``sys.argv``.

    Returns:
        0 on success.
    """
    ap = argparse.ArgumentParser(description="Convert a Diffusers Anima repository.")
    ap.add_argument(
        "--repo", default=REPO, help="Local directory or Hub repository id."
    )
    ap.add_argument("--out", default="checkpoints", help="Output directory.")
    ap.add_argument(
        "--dtype",
        choices=["keep", "bfloat16", "float32"],
        default="keep",
        help="Dtype the model weights are stored in.",
    )
    ap.add_argument("--qk-eps", type=float, default=1e-5)
    ap.add_argument("--no-vae", action="store_true", help="Convert the model only.")
    args = ap.parse_args(argv)

    root = resolve(args.repo)
    os.makedirs(args.out, exist_ok=True)
    dtype = None if args.dtype == "keep" else getattr(torch, args.dtype)
    model_path = os.path.join(args.out, "anima.safetensors")
    cfg = convert_checkpoint(
        os.path.join(root, "transformer"), model_path, args.qk_eps, dtype
    )
    print(f"{model_path}: {cfg}")
    if not args.no_vae:
        from diffusers.models.auto_model import AutoModel

        vae = AutoModel.from_pretrained(root, subfolder="vae")
        vae_path = os.path.join(args.out, "wan21-vae.safetensors")
        save_vae(
            vae_path, Wan21Decoder.from_diffusers(vae), Wan21Encoder.from_diffusers(vae)
        )
        print(vae_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
