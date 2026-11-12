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
