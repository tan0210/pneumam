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
