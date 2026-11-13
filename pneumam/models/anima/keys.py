import re

PARTS = {"qkv": 3, "xkv": 2, "xq": 1, "o1": 1, "o2": 1, "up": 1, "down": 1}
TARGETS = tuple(PARTS)
PROJ = {
    "self_attn.q_proj": ("qkv", 0),
    "self_attn.k_proj": ("qkv", 1),
    "self_attn.v_proj": ("qkv", 2),
    "self_attn.output_proj": ("o1", 0),
    "cross_attn.q_proj": ("xq", 0),
    "cross_attn.k_proj": ("xkv", 0),
    "cross_attn.v_proj": ("xkv", 1),
    "cross_attn.output_proj": ("o2", 0),
    "mlp.layer1": ("up", 0),
    "mlp.layer2": ("down", 0),
}
DIFFUSERS = {
    "attn1.to_q": "self_attn.q_proj",
    "attn1.to_k": "self_attn.k_proj",
    "attn1.to_v": "self_attn.v_proj",
    "attn1.to_out.0": "self_attn.output_proj",
    "attn2.to_q": "cross_attn.q_proj",
    "attn2.to_k": "cross_attn.k_proj",
    "attn2.to_v": "cross_attn.v_proj",
    "attn2.to_out.0": "cross_attn.output_proj",
    "ff.net.0.proj": "mlp.layer1",
    "ff.net.2": "mlp.layer2",
}
KOHYA = {k.replace(".", "_"): k for k in PROJ}
PREFIX = r"^(?:model\.diffusion_model\.|diffusion_model\.|base_model\.model\.transformer\.|base_model\.model\.|transformer\.)?"


def split_key(k: str) -> tuple[str | None, str | None]:
    """Split a LoRA tensor name into module stem and role.

    Args:
        k: Tensor name in PEFT (``lora_A`` / ``lora_B``) or kohya (``lora_down`` / ``lora_up``)
            naming, or an ``.alpha`` entry.

    Returns:
        ``(stem, kind)`` with ``kind`` one of ``"A"``, ``"B"`` and ``"alpha"``, or
        ``(None, None)`` if ``k`` is not a LoRA tensor.
    """
    m = re.search(r"\.lora_(A|B)(?:\.[^.]+)?\.weight$", k)
    if m:
        return k[: m.start()], m.group(1)
    m = re.search(r"\.lora_(down|up)\.weight$", k)
    if m:
        return k[: m.start()], "A" if m.group(1) == "down" else "B"
    if k.endswith(".alpha"):
        return k[:-6], "alpha"
    return None, None


def locate(stem: str) -> tuple[int, str] | None:
    """Map a LoRA module stem to a block index and canonical projection name.

    Dotted stems in native, Diffusers, PEFT and ComfyUI naming and underscore stems in kohya
    naming are accepted.

    Args:
        stem: Module stem from ``split_key``.

    Returns:
        ``(layer, name)`` with ``name`` a key of ``PROJ``, or None if the module is not
        supported.
    """
    m = re.match(PREFIX + r"(?:transformer_)?blocks\.(\d+)\.(.+)$", stem)
    if m:
        layer, rest = int(m.group(1)), m.group(2)
        rest = DIFFUSERS.get(rest, rest)
        return (layer, rest) if rest in PROJ else None
    m = re.match(r"^lora_unet_(?:transformer_)?blocks_(\d+)_(.+)$", stem)
    if m and m.group(2) in KOHYA:
        return int(m.group(1)), KOHYA[m.group(2)]
    return None
