from collections.abc import Callable, Sequence

import torch


class Graphed:
    """A function of tensors captured as a CUDA graph and replayed with new inputs.

    The function must do the same work for any values of its inputs: same shapes, same dtypes,
    and no Python branching on tensor contents.

    Args:
        fn: The function to capture. It is called a few times before the capture.
        varying: Example values of the leading arguments, which change on every call.
        fixed: Example values of the remaining arguments, which change rarely. See ``rebind``.
        warmup: Calls made before capturing, so that lazy initialization is not recorded.
    """

    def __init__(
        self,
        fn: Callable[..., torch.Tensor],
        varying: Sequence[torch.Tensor],
        fixed: Sequence[torch.Tensor] = (),
        warmup: int = 3,
    ) -> None:
        self.varying = [t.clone() for t in varying]
        self.fixed = [t.clone() for t in fixed]
        args = (*self.varying, *self.fixed)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(warmup):
                fn(*args)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.out = fn(*args)

    def rebind(self, *fixed: torch.Tensor) -> None:
        """Replace the values of the rarely changing arguments.

        Args:
            *fixed: New values, with the shapes and dtypes of the captured ones.
        """
        for dst, src in zip(self.fixed, fixed, strict=True):
            dst.copy_(src)

    def __call__(self, *varying: torch.Tensor) -> torch.Tensor:
        """Replay the function.

        Args:
            *varying: New values of the leading arguments, with the captured shapes and dtypes.

        Returns:
            A copy of the result.
        """
        for dst, src in zip(self.varying, varying, strict=True):
            dst.copy_(src)
        self.graph.replay()
        return self.out.clone()


class GraphedStep:
    """A training step captured as a CUDA graph and replayed with new inputs.

    The first call for each combination of input shapes and dtypes runs a few eager steps and
    then captures one. Parameters and optimizer state end up as they were before those steps.
    After a call, each parameter's ``grad`` holds the gradient of that step. It is overwritten,
    not accumulated, by the next call with inputs of the same shapes. Every combination of
    shapes keeps its own graph and the memory that goes with it.

    Args:
        loss_fn: Maps the inputs to a scalar loss. It must do the same work for any input
            values: no ``.item()`` and no Python branching on tensor contents. The random
            numbers it draws come from the CUDA generator.
        params: The trainable parameters.
        optimizer: An optimizer created with ``capturable=True``, or None to
            capture only the forward and backward passes. Use a tensor learning rate to change
            it between steps.
        max_grad_norm: Clip the gradient norm to this before the optimizer step.
        warmup: Eager steps before the capture, so that lazy initialization is not recorded.

    Raises:
        ValueError: If ``optimizer`` is not capturable.
    """

    def __init__(
        self,
        loss_fn: Callable[..., torch.Tensor],
        params: list[torch.nn.Parameter],
        optimizer: torch.optim.Optimizer | None = None,
        max_grad_norm: float | None = None,
        warmup: int = 3,
    ) -> None:
        if optimizer is not None and not optimizer.defaults.get("capturable"):
            raise ValueError("the optimizer must be created with capturable=True")
        self.loss_fn, self.params, self.optimizer = loss_fn, params, optimizer
        self.max_grad_norm, self.warmup = max_grad_norm, warmup
        self.graphs: dict[tuple, tuple] = {}

    def _step(self, inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        for p in self.params:
            p.grad = None
        loss = self.loss_fn(*inputs)
        loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.params, self.max_grad_norm, foreach=True
            )
        if self.optimizer is not None:
            self.optimizer.step()
        return loss.detach()

    def _capture(self, inputs: Sequence[torch.Tensor]) -> tuple:
        static = [x.clone() for x in inputs]
        saved = [p.detach().clone() for p in self.params]
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(self.warmup):
                self._step(static)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        with torch.no_grad():
            for p, v in zip(self.params, saved, strict=True):
                p.copy_(v)
            if self.optimizer is not None:
                for state in self.optimizer.state.values():
                    for v in state.values():
                        if isinstance(v, torch.Tensor):
                            v.zero_()  # undo what the warmup steps accumulated
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            loss = self._step(static)
        return graph, static, loss, [p.grad for p in self.params]

    def __call__(self, *inputs: torch.Tensor) -> torch.Tensor:
        """Run one training step.

        Args:
            *inputs: CUDA tensors passed to ``loss_fn``.

        Returns:
            The loss of this step, detached.
        """
        key = tuple((tuple(x.shape), x.dtype) for x in inputs)
        if key not in self.graphs:
            self.graphs[key] = self._capture(inputs)
        graph, static, loss, grads = self.graphs[key]
        for dst, src in zip(static, inputs, strict=True):
            dst.copy_(src)
        graph.replay()
        for p, g in zip(self.params, grads, strict=True):
            p.grad = g  # each captured shape writes its own gradient buffers
        return loss.clone()
