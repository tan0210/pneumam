import os

import torch
import torch.distributed as dist
from torch import nn


def setup() -> tuple[int, int, torch.device]:
    """Join multi-GPU training when launched by ``torchrun``, and use one GPU otherwise.

    Returns:
        ``(rank, world_size, device)`` for this process.
    """
    local = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    if "RANK" not in os.environ:
        return 0, 1, device
    dist.init_process_group("nccl")
    return dist.get_rank(), dist.get_world_size(), device


def shutdown() -> None:
    """Leave multi-GPU training. Does nothing in a single-GPU run."""
    if dist.is_initialized():
        dist.destroy_process_group()


def broadcast_params(params: list[nn.Parameter]) -> None:
    """Copy rank 0's values to every rank.

    Adapters start from different random values in each process, so call this once after
    attaching them.

    Args:
        params: Parameters, in the same order on every rank.
    """
    with torch.no_grad():
        for p in params:
            dist.broadcast(p, 0)


def average_grads(params: list[nn.Parameter]) -> None:
    """Average gradients across ranks, in place.

    Call it after the last micro-batch's backward pass and before clipping. A parameter without
    a gradient on this rank counts as zero.

    Args:
        params: Parameters, in the same order on every rank.
    """
    for p in params:
        if p.grad is None:
            p.grad = torch.zeros_like(p)
    grads = [p.grad for p in params if p.grad is not None]
    flat = torch.cat([g.reshape(-1) for g in grads])
    dist.all_reduce(flat, op=dist.ReduceOp.AVG)
    offset = 0
    for g in grads:
        g.copy_(flat[offset : offset + g.numel()].view_as(g))
        offset += g.numel()


def average(value: torch.Tensor) -> torch.Tensor:
    """Average a tensor across ranks, in place.

    Args:
        value: The tensor, with the same shape on every rank.

    Returns:
        ``value``, now holding the average.
    """
    dist.all_reduce(value, op=dist.ReduceOp.AVG)
    return value
