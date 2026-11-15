import math
import os

import torch
from PIL import Image

from pneumam.pipeline import Pipeline, to_uint8
from pneumam.training.data.buckets import fit
from pneumam.training.previews import Preview


def learning_rate(
    step: int, peak: float, warmup: int, total: int, floor: float = 0.1
) -> float:
    """Linear warmup, then cosine decay to ``floor * peak``."""
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = min((step - warmup) / max(1, total - warmup), 1.0)
    return peak * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress)))


def save_state(
    path: str, step: int, params: list, optimizer: torch.optim.Optimizer
) -> None:
    """Write what a run needs to continue: the trained values, the optimizer and the step."""
    state = {
        "step": step,
        "params": [p.detach().cpu() for p in params],
        "optimizer": optimizer.state_dict(),
    }
    torch.save(state, path + ".tmp")
    os.replace(path + ".tmp", path)  # a crash mid-write keeps the previous state


def load_state(path: str, params: list, optimizer: torch.optim.Optimizer) -> int:
    """Restore a state written by ``save_state`` and return its step, or 0 without one."""
    if not os.path.exists(path):
        return 0
    state = torch.load(path, map_location="cpu", weights_only=True)
    with torch.no_grad():
        for p, v in zip(params, state["params"], strict=True):
            p.copy_(v)
    optimizer.load_state_dict(state["optimizer"])
    return state["step"]


def load_control(
    path: str, width: int, height: int, device: torch.device
) -> torch.Tensor:
    """Load a control image as ``[1, 3, height, width]`` in ``[0, 1]``."""
    with Image.open(path) as im:
        pixels = fit(im.convert("RGB"), (width, height))
    data = torch.frombuffer(bytearray(pixels.tobytes()), dtype=torch.uint8)
    return data.view(height, width, 3).permute(2, 0, 1)[None].to(device).float() / 255


@torch.no_grad()
def render_previews(
    pipe: Pipeline, previews: list[Preview], folder: str, step: int, before=None
) -> list[Image.Image]:
    """Generate and save the preview images, calling ``before(preview)`` ahead of each one."""
    os.makedirs(folder, exist_ok=True)
    pipe.model.eval()
    out = []
    for i, p in enumerate(previews):
        if before is not None:
            before(p)
        images = pipe.generate(
            p.prompt, p.negative, p.width, p.height, (p.seed,), p.steps, p.cfg
        )
        im = Image.fromarray(to_uint8(images)[0].numpy())
        im.save(os.path.join(folder, f"{step:06d}-{i}.png"))
        out.append(im)
    pipe.model.train()
    return out
