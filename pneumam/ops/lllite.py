import torch

from ..backends import reference, triton
from . import _dispatch


def lllite_hidden(
    z: torch.Tensor,
    down_bias: torch.Tensor,
    mid: torch.Tensor,
    cond_mid: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    strength: torch.Tensor,
    tokens: int,
    width: int,
) -> torch.Tensor:
    """Compute the hidden states of a projection's LLLite adapters, ready for the up product.

    Each adapter ``j`` computes ``silu((silu(z_j + b_j) @ mid_j.T + cond_mid_j) * (1 + gamma_j)
    + beta_j)``. The result holds them side by side, times the strength, then a column holding
    the strength, then zeros. Multiplied by the folded up weights with the folded up bias in the
    strength column, it is the adapters' contribution to the projection. Forward only.

    Args:
        z: Down products without bias, shape ``[M, >= J * m]`` with unit column stride.
        down_bias: Down biases, shape ``[J * m]``.
        mid: Hidden-to-hidden part of each ``mid`` weight, shape ``[J, m, m]``.
        cond_mid: Control part of ``mid`` with its bias, float32 ``[BC * tokens, J * m]``.
        gamma: FiLM scales, float32, shaped like ``cond_mid``.
        beta: FiLM shifts, float32, shaped like ``cond_mid``.
        strength: One-element float32 tensor on the device of ``z``.
        tokens: Tokens per image. Row ``r`` of ``z`` belongs to control image
            ``(r // tokens) % BC``.
        width: Columns of the result, a multiple of 64 above ``J * m``.

    Returns:
        Tensor of shape ``[M, width]`` in the dtype of ``z``.

    Raises:
        RuntimeError: If ``z`` is not on a CUDA device.
    """
    if triton.available and _dispatch.arch(z.device) != "reference":
        return triton.lllite_hidden(
            z, down_bias, mid, cond_mid, gamma, beta, strength, tokens, width
        )
    return reference.lllite_hidden(
        z, down_bias, mid, cond_mid, gamma, beta, strength, tokens, width
    )
