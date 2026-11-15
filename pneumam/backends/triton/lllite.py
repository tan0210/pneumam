from typing import Any

import torch
import triton
import triton.language as tl


@triton.jit
def _hidden_kernel(
    z_ptr,
    z_stride,
    db_ptr,
    wh_ptr,
    cm_ptr,
    g_ptr,
    b_ptr,
    s_ptr,
    out_ptr,
    out_stride,
    M,
    N,
    BC,
    J: tl.constexpr,
    MD: tl.constexpr,
    PAD: tl.constexpr,
    BM: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    live = rows < M
    rows64 = rows.to(tl.int64)
    crow = (((rows // N) % BC) * N + rows % N).to(tl.int64)
    cols = tl.arange(0, MD)
    s = tl.load(s_ptr)
    for j in tl.static_range(J):  # type: ignore
        z = tl.load(
            z_ptr + rows64[:, None] * z_stride + j * MD + cols[None, :],
            mask=live[:, None],
        )
        z = z.to(tl.float32) + tl.load(db_ptr + j * MD + cols).to(tl.float32)[None, :]
        h = z * tl.sigmoid(z)
        # element [k, n] is Wh[n, k], so the product is h @ Wh.T
        wt = tl.load(wh_ptr + j * MD * MD + cols[None, :] * MD + cols[:, None])
        u = tl.dot(h.to(wt.dtype), wt)
        at = crow[:, None] * (J * MD) + j * MD + cols[None, :]
        u += tl.load(cm_ptr + at, mask=live[:, None])
        v = u * (1 + tl.load(g_ptr + at, mask=live[:, None])) + tl.load(
            b_ptr + at, mask=live[:, None]
        )
        v = v * tl.sigmoid(v) * s
        tl.store(
            out_ptr + rows64[:, None] * out_stride + j * MD + cols[None, :],
            v.to(out_ptr.dtype.element_ty),
            mask=live[:, None],
        )
    pc = tl.arange(0, PAD)
    tail = tl.where(pc == 0, s, 0.0)
    tail = tl.broadcast_to(tail[None, :], (BM, PAD))
    tl.store(
        out_ptr + rows64[:, None] * out_stride + J * MD + pc[None, :],
        tail.to(out_ptr.dtype.element_ty),
        mask=live[:, None],
    )


@torch.library.custom_op("pneumam::lllite_hidden", mutates_args=())
def _hidden(
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
    M = z.shape[0]
    J, MD = mid.shape[0], mid.shape[1]
    out = torch.empty(M, width, dtype=z.dtype, device=z.device)
    BM = 64
    meta: dict[str, Any] = {"J": J, "MD": MD, "PAD": width - J * MD, "BM": BM}
    _hidden_kernel[(triton.cdiv(M, BM),)](
        z,
        z.stride(0),
        down_bias,
        mid,
        cond_mid,
        gamma,
        beta,
        strength,
        out,
        out.stride(0),
        M,
        tokens,
        cond_mid.shape[0] // tokens,
        **meta,
    )
    return out


@_hidden.register_fake
def _(z, down_bias, mid, cond_mid, gamma, beta, strength, tokens, width):
    return z.new_empty(z.shape[0], width)


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

    Args:
        z: Down products without bias, shape ``[M, >= J * m]`` with unit column stride.
        down_bias: Down biases, shape ``[J * m]``.
        mid: Hidden-to-hidden part of each ``mid`` weight, shape ``[J, m, m]``.
        cond_mid: Control part of ``mid`` with its bias, float32 ``[BC * tokens, J * m]``.
        gamma: FiLM scales, float32, shaped like ``cond_mid``.
        beta: FiLM shifts, float32, shaped like ``cond_mid``.
        strength: One-element float32 tensor.
        tokens: Tokens per image. Row ``r`` of ``z`` belongs to control image
            ``(r // tokens) % BC``.
        width: Columns of the result, a multiple of 64 above ``J * m``.

    Returns:
        ``[M, width]`` in the dtype of ``z``: each adapter's hidden state times the strength,
        then the strength itself, then zeros.
    """
    return _hidden(z, down_bias, mid, cond_mid, gamma, beta, strength, tokens, width)
