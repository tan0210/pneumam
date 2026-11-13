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
        trim_context: Drop trailing zero padding from the text context, which is faster and
            does not change the result. Ignored while autograd is enabled.
        checkpointing: Recompute each block during backward instead of storing its activations.
            Only matters while autograd is enabled.
        fp8: Quantized projection weights per block, by layer name, or None while FP8 is off.
        fp8_fast_accum: Accumulate FP8 products with reduced precision where the GPU offers it.
        fp8_scales: Per block, the fixed FP8 scales of the attention operands, by operand name.
            Used by the experimental FP8 attention path and by ``fp8_attention_grad``.
        fp8_attention_grad: Run self- and cross-attention in FP8 in both directions while autograd is
            enabled, where the GPU has kernels for it. Faster training, noisier gradients.
        tables: Cached rotary tables, keyed by token grid and device.
        graphs: Captured CUDA graphs, keyed by whatever captured them.
    """

    trim_context: bool = True
    checkpointing: bool = True
    fp8: list[dict[str, FP8Weight]] | None = None
    fp8_fast_accum: bool = False
    fp8_scales: list[dict[str, float]] | None = None
    fp8_attention_grad: bool = False
    tables: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)
    graphs: dict[tuple, Any] = field(default_factory=dict)
