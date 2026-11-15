import math
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import torch
from torch import nn

from ..adapters.lora import enable_fp8_base
from ..graphs import GraphedStep
from ..runtime import Runtime
from . import distributed
from .config import TrainConfig
from .objectives.flow import flow_batch, velocity_loss

Batch = tuple[torch.Tensor, torch.Tensor]


def learning_rate(step: int, cfg: TrainConfig) -> float:
    """Learning rate with linear warmup and cosine decay.

    Args:
        step: Zero-based optimizer step.
        cfg: The run's settings.

    Returns:
        The learning rate for ``step``.
    """
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    progress = min((step - cfg.warmup) / max(1, cfg.steps - cfg.warmup), 1.0)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return cfg.lr * (cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * cosine)


class Trainer:
    """Runs optimizer steps on a model's trainable parameters.

    Each rank of a multi-GPU run holds one trainer. Gradients are averaged across ranks once per
    optimizer step, so the copies of the trained weights stay identical.

    Args:
        model: A model from ``load``, with its trainable parameters already attached.
        rt: Its runtime.
        params: The parameters to train, in the same order on every rank.
        cfg: The run's settings.
        uncond: Empty-prompt embeddings on the model's device, shape ``[1, S, text_dim]`` with
            ``S`` equal to ``cfg.context_length``. Needed if ``cfg.caption_dropout`` is positive.
        loss_fn: Maps ``(latents, context)`` on the model's device to a scalar loss. None uses
            the rectified-flow loss with the settings in ``cfg``. With ``cfg.graph``, a custom
            loss must do the same work for any input values, and its random numbers come from
            the global CUDA generator.
        world: Number of ranks training together.

    Attributes:
        step: Optimizer steps taken so far.
        optimizer: The AdamW optimizer.
        ema: The averaged copies of ``params``, or None if ``cfg.ema`` is 0.
    """

    def __init__(
        self,
        model: Any,
        rt: Runtime,
        params: list[nn.Parameter],
        cfg: TrainConfig,
        uncond: torch.Tensor | None = None,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        world: int = 1,
    ) -> None:
        self.model, self.rt, self.params, self.cfg = model, rt, params, cfg
        self.world = world
        self.step = 0
        self.device = params[0].device
        self.generator = torch.Generator(self.device).manual_seed(cfg.seed)
        self.optimizer = torch.optim.AdamW(
            params,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            fused=self.device.type == "cuda",
        )
        self.ema = [p.detach().clone() for p in params] if cfg.ema > 0 else None
        if loss_fn is None:
            self._draw = lambda latents, context: flow_batch(
                model,
                latents,
                context,
                uncond,
                cfg.shift,
                cfg.dist,
                cfg.caption_dropout,
                self.generator,
            )
            self._loss = lambda *inputs: velocity_loss(model, rt, *inputs)
        else:
            self._draw = lambda latents, context: (latents, context)
            self._loss = loss_fn
        if cfg.fp8_base:
            enable_fp8_base(model, cfg.fp8_backward)
        rt.fp8_attention_grad = cfg.fp8_attention
        self._graphed = None
        if cfg.graph:
            self._graphed = GraphedStep(self._loss, params)

    def seed_rank(self, rank: int) -> None:
        """Give this rank its own noise, so ranks do not train on identical noise.

        Args:
            rank: Index of this process.
        """
        self.generator.manual_seed(self.cfg.seed + rank)

    def _backward(self, batches: Iterator[Batch]) -> torch.Tensor:
        accum = self.cfg.accum
        total = torch.zeros((), device=self.device)
        summed = None
        for _ in range(accum):
            latents, context = next(batches)
            inputs = self._draw(latents.to(self.device), context.to(self.device))
            if self._graphed is None:
                loss = self._loss(*inputs)
                (loss / accum).backward()
            else:
                loss = self._graphed(*inputs)
                if accum > 1:
                    grads = [p.grad for p in self.params if p.grad is not None]
                    if summed is None:
                        summed = [g / accum for g in grads]
                    else:
                        for acc, g in zip(summed, grads, strict=True):
                            acc.add_(
                                g, alpha=1 / accum
                            )  # a replay overwrites its gradients
            total += loss.detach() / accum
        if summed is not None:
            for p, acc in zip(self.params, summed, strict=True):
                p.grad = acc
        return total

    def train_step(self, batches: Iterator[Batch]) -> float:
        """Take one optimizer step.

        Args:
            batches: Yields ``(latents, context)`` micro-batches. ``cfg.accum`` of them are used.

        Returns:
            The loss, averaged over the micro-batches and the ranks.
        """
        cfg = self.cfg
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate(self.step, cfg)
        total = self._backward(batches)
        if self.world > 1:
            distributed.average_grads(self.params)
            distributed.average(total)
        if cfg.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.params, cfg.max_grad_norm)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.step += 1
        if self.ema is not None:
            decay = min(
                cfg.ema, self.step / (self.step + 10)
            )  # warm up from the first weights
            with torch.no_grad():
                for avg, p in zip(self.ema, self.params, strict=True):
                    avg.lerp_(p, 1 - decay)
        return total.item()

    def fit(
        self,
        batches: Iterator[Batch],
        on_step: Callable[[int, float], None] | None = None,
    ) -> None:
        """Train until ``cfg.steps`` optimizer steps have been taken.

        A trainer restored by ``load`` continues from where it stopped.

        Args:
            batches: Yields ``(latents, context)`` micro-batches. See ``ShardedDataset.batches``.
            on_step: Called with ``(step, loss)`` after every optimizer step.
        """
        self.model.train()
        while self.step < self.cfg.steps:
            loss = self.train_step(batches)
            if on_step is not None:
                on_step(self.step, loss)

    @contextmanager
    def ema_weights(self) -> Iterator[None]:
        """Temporarily put the averaged weights into the model, for saving or evaluating.

        Without an average this leaves the weights alone.

        Yields:
            Nothing. The trained weights are restored on exit.
        """
        if self.ema is None:
            yield
            return
        saved = [p.detach().clone() for p in self.params]
        with torch.no_grad():
            for p, avg in zip(self.params, self.ema, strict=True):
                p.copy_(avg)
        try:
            yield
        finally:
            with torch.no_grad():
                for p, old in zip(self.params, saved, strict=True):
                    p.copy_(old)

    def save(self, path: str | os.PathLike) -> None:
        """Write everything needed to continue this run later.

        Args:
            path: Output file. It holds the trained parameters, the optimizer state, the
                average, the step count, the noise generator and the settings. It is replaced
                in one step, so a crash while saving leaves the previous file intact.
        """
        state = {
            "step": self.step,
            "config": self.cfg.dumps(),
            "params": [p.detach().cpu() for p in self.params],
            "optimizer": self.optimizer.state_dict(),
            "ema": None if self.ema is None else [t.cpu() for t in self.ema],
            "generator": self.generator.get_state(),
        }
        tmp = os.fspath(path) + ".tmp"
        torch.save(state, tmp)
        os.replace(tmp, path)  # a crash mid-write keeps the previous state

    def load(self, path: str | os.PathLike) -> None:
        """Restore a run saved by ``save`` into this trainer.

        The settings of this trainer are kept. The file's settings are not applied.

        Args:
            path: File written by ``save``.

        Raises:
            ValueError: If the file's parameters do not match this trainer's.
        """
        state = torch.load(path, map_location="cpu", weights_only=True)
        shapes = [tuple(p.shape) for p in self.params]
        if [tuple(t.shape) for t in state["params"]] != shapes:
            raise ValueError(f"{path} was saved for different trainable parameters")
        with torch.no_grad():
            for p, value in zip(self.params, state["params"], strict=True):
                p.copy_(value)
        self.optimizer.load_state_dict(state["optimizer"])
        if self.ema is not None and state["ema"] is not None:
            self.ema = [t.to(self.device) for t in state["ema"]]
        self.generator.set_state(state["generator"])
        self.step = state["step"]
