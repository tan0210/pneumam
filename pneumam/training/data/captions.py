import random
from dataclasses import dataclass


@dataclass(frozen=True)
class CaptionRules:
    """How a subset's captions change every time a sample is drawn, as in kohya's sd-scripts.

    Attributes:
        prefix: Text put before the caption, such as a trigger word. It is never shuffled or
            dropped.
        suffix: Text put after the caption, kept the same way.
        shuffle: Shuffle the comma-separated tags.
        keep_tokens: Leading tags that are never shuffled or dropped.
        keep_tokens_separator: If set and present in a caption, everything before it is kept
            in place instead of the first ``keep_tokens`` tags.
        tag_dropout: Probability of dropping each tag that is not kept.
        dropout: Probability of training on the empty caption instead.
    """

    prefix: str = ""
    suffix: str = ""
    shuffle: bool = False
    keep_tokens: int = 0
    keep_tokens_separator: str = ""
    tag_dropout: float = 0.0
    dropout: float = 0.0

    def apply(self, caption: str, rng: random.Random) -> str:
        """Produce the caption to train on this time.

        Args:
            caption: The caption as written.
            rng: Random source.

        Returns:
            The processed caption.
        """
        if self.dropout > 0 and rng.random() < self.dropout:
            return ""
        sep = self.keep_tokens_separator
        if sep and sep in caption:
            head, tail = caption.split(sep, 1)
            kept, rest = _tags(head), _tags(tail)
        else:
            tags = _tags(caption)
            kept, rest = tags[: self.keep_tokens], tags[self.keep_tokens :]
        if self.tag_dropout > 0:
            rest = [t for t in rest if rng.random() >= self.tag_dropout]
        if self.shuffle:
            rng.shuffle(rest)
        parts = [
            p for p in (self.prefix.strip(), *kept, *rest, self.suffix.strip()) if p
        ]
        return ", ".join(parts)


def _tags(text: str) -> list[str]:
    return [t.strip() for t in text.split(",") if t.strip()]
