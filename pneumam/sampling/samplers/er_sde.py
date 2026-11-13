import math
from collections.abc import Callable, Sequence

import numpy as np
import torch

NOISE_SCALES = ("er", "ode")


def _er_phi(lam: np.ndarray) -> np.ndarray:
    return lam * (np.exp(lam**0.3) + 10.0)


def _ode_phi(lam: np.ndarray) -> np.ndarray:
    return lam


def _er_integrals(
    lam_t: float, lam_s: float, phi: Callable[[np.ndarray], np.ndarray]
) -> tuple[float, float]:
    nodes, w = np.polynomial.legendre.leggauss(64)
    a, b = math.log(lam_t), math.log(lam_s)
    u = 0.5 * (b - a) * nodes + 0.5 * (
        b + a
    )  # integrate in log(lam), where it is smooth
    lam = np.exp(u)
    scale = 0.5 * (b - a) * w * lam / phi(lam)
    i1 = float(scale.sum())  # integral of 1 / phi from lam_t to lam_s
    i2 = float((scale * (lam - lam_s)).sum())  # integral of (lam - lam_s) / phi
    return i1, i2


def er_sde_plan(
    sg: Sequence[float] | torch.Tensor,
    max_stage: int = 3,
    s_noise: float = 1.0,
    phi: Callable[[np.ndarray], np.ndarray] = _er_phi,
    first_offset: float = 1e-4,
) -> list[tuple[float, tuple[float, float, float], float]]:
    """Precompute the per-step coefficients of the ER-SDE solver.

    Args:
        sg: Noise schedule from ``sigmas``, length ``steps + 1``, ending in 0.
        max_stage: Highest solver order used, 1 to 3. Step ``i`` uses
            ``min(max(i, 1), max_stage)``.
        s_noise: Multiplier on the injected noise.
        phi: Noise-scale function of ``lambda = sigma / (1 - sigma)``. The identity gives the
            noise-free ODE solver.
        first_offset: The first sigma is clamped to ``1 - first_offset``, since ``lambda`` is
            infinite at sigma 1.

    Returns:
        One ``(c_x, (w_0, w_1, w_2), noise_std)`` per step. The new latent is
        ``c_x * x + w_0 * x0 + w_1 * x0_prev + w_2 * x0_prev2 + noise_std * z``, where ``x0`` is
        the current clean-latent prediction and ``z`` is standard normal.

    Raises:
        ValueError: If ``max_stage`` is not 1, 2 or 3.
    """
    if max_stage not in (1, 2, 3):
        raise ValueError("max_stage must be 1, 2 or 3")
    values = torch.as_tensor(sg).double().tolist()
    sig = [min(float(v), 1.0 - first_offset) for v in values]
    n = len(sig) - 1
    alpha = [1.0 - v for v in sig]
    lam = [v / a for v, a in zip(sig, alpha, strict=True)]
    plan = []
    for i in range(n):
        if sig[i + 1] <= 0.0:
            plan.append((0.0, (1.0, 0.0, 0.0), 0.0))
            continue
        ls, lt, at, as_ = lam[i], lam[i + 1], alpha[i + 1], alpha[i]
        pt, ps = float(phi(np.array(lt))), float(phi(np.array(ls)))
        r = pt / ps
        c_x = at / as_ * r
        std = 0.0
        if phi is not _ode_phi:
            std = s_noise * at * math.sqrt(max(lt * lt - r * r * ls * ls, 0.0))
        w = [at * (1.0 - r), 0.0, 0.0]
        stage = min(max(i, 1), max_stage)  # the prediction at pure noise is not history
        if stage >= 2:
            i1, i2 = _er_integrals(lt, ls, phi)
            a1 = at * ((lt - ls) + pt * i1)
            d1 = ls - lam[i - 1]
            if stage == 2:
                w[0] += a1 / d1
                w[1] -= a1 / d1
            else:
                a2 = at * ((lt - ls) ** 2 / 2 + pt * i2)
                d1b = lam[i - 1] - lam[i - 2]
                big = d1 + d1b
                f1 = (1 / d1, -1 / d1, 0.0)  # Newton form: x0' = f1 + d1 * f2
                f2 = (1 / (d1 * big), -(1 / d1 + 1 / d1b) / big, 1 / (d1b * big))
                for k in range(3):
                    w[k] += a1 * (f1[k] + d1 * f2[k]) + a2 * 2 * f2[k]  # x0'' = 2 * f2
        plan.append((c_x, (w[0], w[1], w[2]), std))
    return plan


def er_sde(
    velocity: Callable[[int, torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    sg: torch.Tensor,
    max_stage: int = 3,
    s_noise: float = 1.0,
    seeds: Sequence[int] | None = None,
    noise_scale: str = "er",
) -> torch.Tensor:
    """Sample with the ER-SDE solver for flow models, one model evaluation per step.

    The solver is from Cui et al., arXiv:2309.06169.

    Args:
        velocity: Maps ``(step index, latent)`` to the predicted velocity, shaped like the latent.
        x: Initial noise latent, float32, shape ``[B, ...]``.
        sg: Noise schedule from ``sigmas``.
        max_stage: Highest solver order, 1 to 3.
        s_noise: Multiplier on the injected noise. Values below 1 shrink the result, and 0
            collapses it to the average prediction. For a noise-free sampler use
            ``noise_scale="ode"``.
        seeds: One seed per batch element for the injected noise, so a sample does not depend on
            the rest of the batch. If None, the global torch RNG is used.
        noise_scale: ``"er"`` for the stochastic solver, or ``"ode"`` for its noise-free limit,
            a multistep ODE solver of the same order.

    Returns:
        The denoised latent.

    Raises:
        ValueError: If ``max_stage`` is not 1, 2 or 3, ``noise_scale`` is unknown, or ``seeds``
            does not match the batch size.
    """
    if noise_scale not in NOISE_SCALES:
        raise ValueError(
            f"unknown noise_scale {noise_scale!r}; expected one of {NOISE_SCALES}"
        )
    phi = _ode_phi if noise_scale == "ode" else _er_phi
    plan = er_sde_plan(sg, max_stage, s_noise, phi)
    sig = sg.double().cpu().tolist()
    if seeds is not None and len(seeds) != x.shape[0]:
        raise ValueError("need one seed per batch element")
    gens = None
    if seeds is not None:
        offset = (
            1 << 31
        )  # keeps this stream apart from the one that drew the initial latent
        gens = [torch.Generator(x.device).manual_seed(int(s) + offset) for s in seeds]
    hist: list[torch.Tensor] = []
    for i, (c_x, w, std) in enumerate(plan):
        x0 = x - sig[i] * velocity(i, x)
        out = c_x * x + w[0] * x0
        if w[1]:
            out = out + w[1] * hist[-1]
        if w[2]:
            out = out + w[2] * hist[-2]
        if std:
            if gens is None:
                z = torch.randn_like(x)
            else:
                shape, kw = (1, *x.shape[1:]), {"device": x.device, "dtype": x.dtype}
                z = torch.cat([torch.randn(shape, generator=g, **kw) for g in gens])
            out = out + std * z
        hist = [*hist, x0][-2:]
        x = out
    return x
