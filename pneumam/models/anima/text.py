from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


def pad_context(ctx: torch.Tensor, length: int) -> torch.Tensor:
    """Zero-pad text embeddings along the sequence.

    Args:
        ctx: Text embeddings, shape ``[B, S, text_dim]``.
        length: Target sequence length, at least ``S``.

    Returns:
        Tensor of shape ``[B, length, text_dim]``.
    """
    return F.pad(ctx, (0, 0, 0, length - ctx.shape[1]))


@dataclass
class TextEncoder:
    """The parts that turn a prompt into the model's text context.

    A prompt is encoded by the language model, and its hidden states are then mapped onto the
    prompt's T5 token ids by the conditioner.

    Attributes:
        tokenizer: Tokenizer of the language model.
        encoder: The language model. Its last hidden state is used.
        t5_tokenizer: T5 tokenizer. Only its token ids are used.
        conditioner: Maps the hidden states and the T5 ids to text embeddings.
        max_length: Prompts are truncated to this many tokens by both tokenizers.
    """

    tokenizer: Any
    encoder: Any
    t5_tokenizer: Any
    conditioner: Any
    max_length: int = 512

    @classmethod
    def from_pipeline(cls, pipe: Any) -> "TextEncoder":
        """Take the text components of a loaded Diffusers Anima pipeline.

        Args:
            pipe: The pipeline, with its components loaded.

        Returns:
            An encoder sharing the pipeline's components.
        """
        return cls(
            pipe.tokenizer, pipe.text_encoder, pipe.t5_tokenizer, pipe.text_conditioner
        )

    @torch.no_grad()
    def encode(self, prompts: str | Sequence[str], dtype: torch.dtype) -> torch.Tensor:
        """Encode prompts into text embeddings.

        Args:
            prompts: One prompt or several.
            dtype: Dtype of the result. Use the model's compute dtype.

        Returns:
            Text embeddings, shape ``[B, S, text_dim]``, on the encoder's device. Shorter prompts
            are zero-padded to the longest one.
        """
        if isinstance(prompts, str):
            prompts = [prompts]
        outs = [self._encode_one(p).to(dtype) for p in prompts]
        length = max(o.shape[1] for o in outs)
        return torch.cat([pad_context(o, length) for o in outs])

    def _encode_one(self, prompt: str) -> torch.Tensor:
        device = self.encoder.device
        tok = {
            "padding": "longest",
            "max_length": self.max_length,
            "truncation": True,
            "return_tensors": "pt",
        }
        llm = self.tokenizer([prompt], **tok)
        ids, mask = llm.input_ids.to(device), llm.attention_mask.to(device)
        h = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        h = h * mask.to(h).unsqueeze(-1)
        t5 = self.t5_tokenizer([prompt], **tok)
        return self.conditioner(
            source_hidden_states=h.to(self.conditioner.dtype),
            target_input_ids=t5.input_ids.to(device),
            target_attention_mask=t5.attention_mask.to(device),
            source_attention_mask=mask,
        )
