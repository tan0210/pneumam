import os
from dataclasses import dataclass


@dataclass
class Preview:
    """An image to generate during training to see how it goes.

    Attributes:
        prompt: The prompt.
        negative: The negative prompt.
        width: Image width in pixels.
        height: Image height in pixels.
        seed: Noise seed.
        steps: Sampler steps.
        cfg: Guidance scale.
        control: Control image, for LLLite.
    """

    prompt: str
    negative: str = ""
    width: int = 1024
    height: int = 1024
    seed: int = 0
    steps: int = 30
    cfg: float = 4.0
    control: str | None = None

    @classmethod
    def parse(cls, line: str) -> "Preview":
        """Read a sample prompt line in the format of kohya's sd-scripts.

        ``--n`` sets the negative prompt, ``--w`` and ``--h`` the size, ``--d`` the seed,
        ``--s`` the steps, ``--l`` the guidance scale and ``--cn`` the control image.

        Args:
            line: For example ``1girl, smile --n blurry --w 832 --h 1216 --d 7``.

        Returns:
            The preview.
        """
        prompt, *opts = line.split(" --")
        out = cls(prompt.strip())
        names = {"w": "width", "h": "height", "d": "seed", "s": "steps"}
        for opt in opts:
            key, _, value = opt.partition(" ")
            value = value.strip()
            if key == "n":
                out.negative = value
            elif key in names:
                setattr(out, names[key], int(value))
            elif key == "l":
                out.cfg = float(value)
            elif key == "cn":
                out.control = value or None
        return out


def load_previews(path: str | os.PathLike) -> list[Preview]:
    """Read a sample prompt file in the format of kohya's sd-scripts.

    Each line is one prompt. Empty lines and lines starting with ``#`` are skipped. Relative
    control images are taken relative to the file.

    Args:
        path: The text file.

    Returns:
        The previews.
    """
    root = os.path.dirname(os.path.abspath(path))
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = Preview.parse(line)
            if p.control is not None:
                p.control = os.path.normpath(os.path.join(root, p.control))
            out.append(p)
    return out
