import math
import os
import re
from collections.abc import Sequence

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn

from ..ops.lora import lora_add

LORA_FORMAT = "pneumam-lora-1"
_LEGACY = re.compile(
    r"^(\w+)-flat-lora-1$"
)  # older ids carry the architecture in their name


class LoRALinear(nn.Module):
    """A frozen linear layer plus a trainable low-rank update.

    For fused projections (``parts`` > 1) each part gets its own independent rank-``rank``
    update, exactly as if the projections were adapted separately.

    Args:
        base: The layer to adapt. Its weight is not modified.
        parts: Number of equal-sized output slices adapted independently.
        rank: Rank of each update.
        alpha: Scaling numerator. The update is scaled by ``alpha / rank``.
        dropout: Dropout probability on the input of the low-rank path.

    Raises:
        ValueError: If ``rank < 1`` or the output width is not divisible by ``parts``.
    """

    def __init__(
        self,
        base: nn.Linear,
        parts: int = 1,
        rank: int = 16,
        alpha: float | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("rank must be >= 1")
        if base.out_features % parts:
            raise ValueError(
                f"{base.out_features} outputs not divisible into {parts} parts"
            )
        self.base, self.parts, self.rank = base, parts, rank
        self.alpha = float(rank if alpha is None else alpha)
        self.scale = self.alpha / rank
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        kw = {"device": base.weight.device, "dtype": torch.float32}
        out = base.out_features // parts
        self.lora_A = nn.Parameter(torch.empty(parts * rank, base.in_features, **kw))
        self.lora_B = nn.Parameter(torch.zeros(parts, out, rank, **kw))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.cache_casts = True
        self._key = None
        self._cast: tuple[torch.Tensor, torch.Tensor] | None = None

    @property
    def weight(self) -> torch.Tensor:
        """Weight of the wrapped layer, without the update.

        Returns:
            The base weight, shape ``[out_features, in_features]``.
        """
        return self.base.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the adapted layer.

        Args:
            x: Input, shape ``[..., in_features]``.

        Returns:
            Output in the dtype of ``x``, shape ``[..., out_features]``.
        """
        lead = x.shape[:-1]
        flat = x.reshape(-1, x.shape[-1])
        a, bs = self._cached(flat.dtype)
        base, low = self.base(flat), self.drop(flat)
        out = lora_add(base, low, self.lora_A, self.lora_B, a, bs, self.scale)
        return out.view(*lead, out.shape[-1])

    def _cached(self, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        A, B = self.lora_A, self.lora_B
        if not self.cache_casts or torch.compiler.is_compiling():
            with torch.no_grad():
                return A.to(dtype), (B * self.scale).to(dtype)
        key = (A._version, B._version, A.data_ptr(), B.data_ptr(), dtype)
        if self._key != key or self._cast is None:
            with torch.no_grad():
                self._cast = (A.to(dtype), (B * self.scale).to(dtype))
            self._key = key
        return self._cast

    def delta_weight(self) -> torch.Tensor:
        """Weight update this adapter adds to the base weight.

        Returns:
            Float32 tensor of shape ``[out_features, in_features]``, including the
            ``alpha / rank`` scale.
        """
        b = torch.block_diag(*self.lora_B.unbind(0))
        return (b.float() @ self.lora_A.float()) * self.scale


def _targets(model: nn.Module) -> dict[str, int]:
    return model.spec.lora_targets  # type: ignore


def _layers(model: nn.Module):
    for i, block in enumerate(model.blocks):  # type: ignore
        for name in _targets(model):
            yield i, block, name, getattr(block, name)


def has_lora(model: nn.Module) -> bool:
    """Report whether any layer of ``model`` carries an adapter.

    Args:
        model: The model to inspect.

    Returns:
        True if at least one ``LoRALinear`` is present.
    """
    return any(isinstance(m, LoRALinear) for m in model.modules())


def lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Collect the trainable adapter parameters.

    Args:
        model: The model.

    Returns:
        All adapter parameters, in module order.
    """
    mods = [m for m in model.modules() if isinstance(m, LoRALinear)]
    return [p for m in mods for p in (m.lora_A, m.lora_B)]


def _wrap(
    model: nn.Module,
    block: nn.Module,
    name: str,
    rank: int,
    alpha: float | None,
    dropout: float,
) -> LoRALinear:
    base = getattr(block, name)
    if isinstance(base, LoRALinear):
        raise ValueError(f"{name} already has an adapter")  # noqa: TRY004
    mod = LoRALinear(base, _targets(model)[name], rank, alpha, dropout)
    setattr(block, name, mod)
    return mod


def add_lora(
    model: nn.Module,
    rank: int = 16,
    alpha: float | None = None,
    targets: Sequence[str] | None = None,
    dropout: float = 0.0,
    layers: Sequence[int] | None = None,
) -> list[nn.Parameter]:
    """Attach trainable adapters to the model and freeze everything else.

    Args:
        model: A model from ``load``.
        rank: Rank of each update.
        alpha: Scaling numerator. Defaults to ``rank``.
        targets: Projections to adapt, out of ``model.spec.lora_targets``. Defaults to all.
        dropout: Dropout probability on the adapter input.
        layers: Block indices to adapt. Defaults to all blocks.

    Returns:
        The trainable adapter parameters, in a stable order.

    Raises:
        ValueError: If a target is unknown or a selected layer already has an adapter.
    """
    known = _targets(model)
    targets = tuple(known) if targets is None else targets
    unknown = [t for t in targets if t not in known]
    if unknown:
        raise ValueError(
            f"unknown targets {unknown}; expected a subset of {tuple(known)}"
        )
    model.requires_grad_(False)
    blocks = model.blocks
    for i in range(len(blocks)) if layers is None else layers:  # type: ignore
        for name in targets:
            _wrap(model, blocks[i], name, rank, alpha, dropout)  # type: ignore
    return lora_parameters(model)


@torch.no_grad()
def remove_lora(model: nn.Module, merge: bool = True) -> None:
    """Detach all adapters from the model.

    Args:
        model: The model.
        merge: If True, add each adapter's update into the base weight first, cast to its dtype.
    """
    for _, block, name, m in _layers(model):
        if not isinstance(m, LoRALinear):
            continue
        if merge:
            w = m.base.weight
            w.copy_((w.float() + m.delta_weight().to(w.device)).to(w.dtype))
        setattr(block, name, m.base)


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Export the attached adapters as tensors.

    Args:
        model: The model.

    Returns:
        Float32 CPU tensors ``blocks.{layer}.{target}.lora_A`` and ``.lora_B``, with
        ``alpha / rank`` already applied to ``lora_B``. See ``merge_lora``.
    """
    out = {}
    for i, _, name, m in _layers(model):
        if isinstance(m, LoRALinear):
            b = torch.block_diag(*m.lora_B.detach().unbind(0)).float() * m.scale
            out[f"blocks.{i}.{name}.lora_A"] = (
                m.lora_A.detach().float().cpu().contiguous()
            )
            out[f"blocks.{i}.{name}.lora_B"] = b.cpu().contiguous()
    return out


def write_lora(
    path: str | os.PathLike,
    sd: dict[str, torch.Tensor],
    architecture: str,
    rank: int | None = None,
    alpha: float | None = None,
) -> None:
    """Write adapter tensors to a safetensors file.

    Args:
        path: Output path.
        sd: Tensors in the layout of ``lora_state_dict``.
        architecture: Name of the model family the adapter targets.
        rank: Rank of the adapter, if it is the same for every layer.
        alpha: Its scaling numerator, if ``rank`` is given.
    """
    meta = {"format": LORA_FORMAT, "architecture": architecture}
    if rank is not None:
        meta["rank"] = str(rank)
        meta["alpha"] = str(float(rank if alpha is None else alpha))
    save_file(sd, path, metadata=meta)


def read_lora(
    path: str | os.PathLike, architecture: str | None = None
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Read an adapter file.

    Args:
        path: Adapter file from ``write_lora`` or ``save_lora``.
        architecture: If given, the model family the adapter must target.

    Returns:
        ``(sd, meta)``: the CPU tensors in the layout of ``lora_state_dict`` and the file's
        metadata.

    Raises:
        ValueError: If the file is not an adapter file, or targets another architecture.
    """
    with safe_open(path, framework="pt", device="cpu") as f:
        meta = f.metadata() or {}
        fmt = meta.get("format", "")
        legacy = _LEGACY.match(fmt)
        if fmt != LORA_FORMAT and legacy is None:
            raise ValueError(f"{path} is not a {LORA_FORMAT} file (format: {fmt!r})")
        found = legacy.group(1) if legacy else meta.get("architecture")
        if architecture is not None and found != architecture:
            raise ValueError(f"{path} targets {found!r}, not {architecture!r}")
        sd = {k: f.get_tensor(k) for k in f.keys()}  # noqa: SIM118
    return sd, meta


def save_lora(model: nn.Module, path: str | os.PathLike) -> None:
    """Save the attached adapters to a safetensors file.

    The file can be loaded back with ``load_lora`` or merged with ``merge_lora``.

    Args:
        model: The model.
        path: Output path.

    Raises:
        ValueError: If the model has no adapters.
    """
    sd = lora_state_dict(model)
    if not sd:
        raise ValueError("model has no adapters")
    first = next(m for m in model.modules() if isinstance(m, LoRALinear))
    write_lora(path, sd, model.spec.architecture, first.rank, first.alpha)  # type: ignore


def _locate(model: nn.Module, key: str) -> tuple[nn.Module, str]:
    _, layer, name, _ = key.split(".")
    if name not in _targets(model) or int(layer) >= len(model.blocks):  # type: ignore
        raise ValueError(f"{key} does not match the model")
    return model.blocks[int(layer)], name  # type: ignore


def load_lora(
    model: nn.Module, path: str | os.PathLike, dropout: float = 0.0
) -> list[nn.Parameter]:
    """Attach adapters from a file written by ``save_lora`` and freeze everything else.

    Args:
        model: A model without adapters on the layers in the file.
        path: Adapter file.
        dropout: Dropout probability on the adapter input.

    Returns:
        The trainable adapter parameters.

    Raises:
        ValueError: If the file is not an adapter file for this architecture, its layer names do
            not match the model, or its per-projection ranks are unequal.
    """
    sd, meta = read_lora(path, model.spec.architecture)  # type: ignore
    model.requires_grad_(False)
    for key in sorted(k for k in sd if k.endswith(".lora_A")):
        block, name = _locate(model, key)
        parts = _targets(model)[name]
        a, b = sd[key], sd[key[:-1] + "B"]
        if a.shape[0] % parts:
            raise ValueError(f"{key}: unequal per-projection ranks are not supported")
        rank = a.shape[0] // parts
        same = meta.get("rank") == str(rank)
        alpha = float(meta.get("alpha", rank)) if same else float(rank)
        mod = _wrap(model, block, name, rank, alpha, dropout)
        out = b.shape[0] // parts
        blocks = [
            b[j * out : (j + 1) * out, j * rank : (j + 1) * rank] for j in range(parts)
        ]
        with torch.no_grad():
            mod.lora_A.copy_(a)
            mod.lora_B.copy_(torch.stack(blocks) / mod.scale)
    return lora_parameters(model)


@torch.no_grad()
def merge_lora(
    model: nn.Module,
    loras: Sequence[tuple[dict[str, torch.Tensor], float]],
    backup: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Merge adapters into the model's weights, keeping what is needed to undo it.

    All adapters are summed in float32 and each weight is rounded once. Merging with the backup
    of an earlier merge starts again from the original weights, so the new set of adapters
    replaces the old one instead of stacking on top of it.

    Args:
        model: The model.
        loras: ``(sd, strength)`` pairs, with ``sd`` in the layout of ``lora_state_dict``.
            See ``read_lora``.
        backup: The return value of an earlier merge into this model, or None for a model with
            its original weights.

    Returns:
        The original weights of every layer merged into so far, by layer key. Pass it to
        ``unmerge_lora``, or to a later ``merge_lora``.

    Raises:
        ValueError: If a layer name does not match the model.
    """
    backup = {} if backup is None else backup
    deltas: dict[str, torch.Tensor] = {}
    for sd, strength in loras:
        for key in sd:
            if not key.endswith(".lora_A"):
                continue
            base = key[: -len(".lora_A")]
            w = _weight(model, base)
            a = sd[key].to(w.device, torch.float32)
            b = sd[base + ".lora_B"].to(w.device, torch.float32)
            d = strength * (b @ a)
            deltas[base] = deltas[base] + d if base in deltas else d
    for base in set(backup) | set(deltas):
        w = _weight(model, base)
        if base not in backup:
            backup[base] = w.clone()
        orig = backup[base]
        w.copy_((orig.float() + deltas[base]).to(w.dtype) if base in deltas else orig)
    return backup


def _weight(model: nn.Module, base: str) -> torch.Tensor:
    block, name = _locate(model, base + ".lora_A")
    return getattr(block, name).weight


@torch.no_grad()
def unmerge_lora(model: nn.Module, backup: dict[str, torch.Tensor]) -> None:
    """Restore the weights a model had before ``merge_lora``, exactly.

    Args:
        model: The model.
        backup: The return value of ``merge_lora``. It is emptied.
    """
    for base, orig in backup.items():
        _weight(model, base).copy_(orig)
    backup.clear()
