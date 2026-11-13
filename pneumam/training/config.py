import json
from dataclasses import asdict, dataclass


@dataclass
class TrainConfig:
    """Settings of a training run.

    Attributes:
        steps: Number of optimizer steps.
        batch: Samples per micro-batch, per GPU.
        accum: Micro-batches whose gradients are summed into one optimizer step.
        lr: Peak learning rate.
        warmup: Steps of linear warmup.
        min_lr_ratio: Learning rate at the end of the cosine decay, as a fraction of ``lr``.
        weight_decay: AdamW weight decay.
        max_grad_norm: Gradient-norm clip. 0 turns clipping off.
        ema: Decay of the exponential moving average of the trained weights. 0 turns it off.
        shift: Shift of the training noise levels. None uses the model's default.
        dist: Noise-level distribution, ``"logit_normal"`` or ``"uniform"``.
        caption_dropout: Probability of training a sample on the empty prompt.
        context_length: Text embeddings are zero-padded to this many tokens. Use the value
            images will be generated with.
        seed: Seed of the noise, the noise levels and the data order.
        graph: Replay the forward and backward passes as a CUDA graph, which is faster at small
            batch sizes. Each latent size captures its own graph on first use and keeps its
            memory.
    """

    steps: int = 1000
    batch: int = 2
    accum: int = 1
    lr: float = 1e-4
    warmup: int = 50
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    ema: float = 0.0
    shift: float | None = None
    dist: str = "logit_normal"
    caption_dropout: float = 0.1
    context_length: int = 512
    seed: int = 1234
    graph: bool = False

    def dumps(self) -> str:
        """Serialize the config to JSON.

        Returns:
            The JSON string.
        """
        return json.dumps(asdict(self))

    @classmethod
    def loads(cls, text: str) -> "TrainConfig":
        """Deserialize a config produced by ``dumps``.

        Args:
            text: JSON string.

        Returns:
            The config.
        """
        return cls(**json.loads(text))
