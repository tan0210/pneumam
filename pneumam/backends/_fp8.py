import math
from typing import NamedTuple

import torch

FP8 = torch.float8_e4m3fn
FP8_MAX = 448.0


class FP8Weight(NamedTuple):
    """A weight quantized to FP8 with one scale per output channel.

    Attributes:
        q: FP8 weight, shape ``[N, K]``.
        s: Float32 scales, shape ``[N]``.
    """

    q: torch.Tensor
    s: torch.Tensor


def quantize_weight(w: torch.Tensor) -> FP8Weight:
    """Quantize a ``[N, K]`` weight to FP8 with a scale per output channel.

    Args:
        w: Weight of any floating dtype.

    Returns:
        The quantized weight.
    """
    wf = w.float()
    s = (wf.abs().amax(dim=1) / FP8_MAX).clamp_min(1e-12)
    q = (wf / s[:, None]).clamp(-FP8_MAX, FP8_MAX).to(FP8)
    return FP8Weight(q.contiguous(), s.contiguous())


def norm_scale(weight: torch.Tensor) -> float:
    """Pick the FP8 scale for RMS-normalized, rotated heads.

    Such a head never exceeds ``sqrt(hd) * max(abs(weight))`` in magnitude, so the scale needs no
    look at the data and can be computed once per weight.

    Args:
        weight: The norm scale of the head, shape ``[hd]``.

    Returns:
        The largest power of two that keeps every scaled value within the FP8 range.
    """
    bound = math.sqrt(weight.numel()) * max(float(weight.abs().max()), 1e-12)
    return 2.0 ** math.floor(math.log2(FP8_MAX / bound))


def to_fp8(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Quantize a tensor to FP8 after multiplying it by a fixed scale.

    Values beyond 448 in magnitude after scaling are clipped.

    Args:
        x: Tensor of any floating dtype.
        scale: Factor applied before quantizing.

    Returns:
        FP8 tensor with the shape of ``x``. Divide by ``scale`` to recover the values.
    """
    return (x.float() * scale).clamp(-FP8_MAX, FP8_MAX).to(FP8)
