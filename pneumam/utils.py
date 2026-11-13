import os


def _flag(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() not in (
        "",
        "0",
        "false",
        "no",
        "off",
    )


def is_experimental_fp8_attention() -> bool:
    """Tell whether the experimental FP8 attention path is switched on.

    With it, FP8 inference quantizes the attention operands with fixed scales and runs attention
    itself in FP8. It is less accurate than the default path and only faster on GPUs with an FP8
    attention kernel.

    Returns:
        True if the ``PNEUMAM_EXPERIMENTAL_FP8_ATTENTION`` environment variable is set to anything
        but an empty string, ``0``, ``false``, ``no`` or ``off``.
    """
    return _flag("PNEUMAM_EXPERIMENTAL_FP8_ATTENTION")
