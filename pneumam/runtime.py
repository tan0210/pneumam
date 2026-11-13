from dataclasses import dataclass, field
from typing import Any

import torch

from .ops.gemm import FP8Weight


@dataclass
class Runtime:
    """Per-model state that is not a parameter.

    It lives outside the module so that a model can be cloned, offloaded or patched without
    carrying caches along. Whoever runs the model owns one.

    Attributes:
        trim_context: Drop trailing zero padding from the text context where that is faster.
            Ignored while autograd is enabled.
        checkpointing: Recompute each block during backward instead of storing its activations.
            Only matters while autograd is enabled.
        fp8: Quantized projection weights per block, by layer name, or None while FP8 is off.
        fp8_fast_accum: Accumulate FP8 products with reduced precision where the GPU offers it.
        fp8_scales: Per block, the fixed FP8 scales of the attention operands, by operand name.
            Only used by the experimental FP8 attention path.
        tables: Cached rotary tables, keyed by token grid and device.
        graphs: Captured CUDA graphs, keyed by whatever captured them.
    """

    trim_context: bool = False
    checkpointing: bool = True
    fp8: list[dict[str, FP8Weight]] | None = None
    fp8_fast_accum: bool = False
    fp8_scales: list[dict[str, float]] | None = None
    tables: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)
    graphs: dict[tuple, Any] = field(default_factory=dict)
