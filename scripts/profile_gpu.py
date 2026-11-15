import argparse
import json
import os
import platform
import statistics
import time
import traceback
from collections.abc import Callable
from typing import Any, cast

import torch

from pneumam import models
from pneumam.adapters import lora as L
from pneumam.backends import ffpa, flash_attn, quack, triton, warpy
from pneumam.graphs import Graphed, GraphedStep
from pneumam.models.anima import fused
from pneumam.models.anima.config import AnimaConfig
from pneumam.models.anima.dit import AnimaDiT
from pneumam.models.vae import load_decoder
from pneumam.ops import _dispatch, adaln, attention, gemm, norm, rope
from pneumam.runtime import Runtime
from pneumam.sampling.guidance import guided
from pneumam.sampling.schedules import noise, sigmas

FLAG = "PNEUMAM_EXPERIMENTAL_FP8_ATTENTION"
MODES = ("bf16", "fp8", "fp8+attn")
FAILURES: list[dict[str, str]] = []


def heading(title: str) -> None:
    """Print a section title underlined with dashes.

    Args:
        title: The title.
    """
    print(f"\n{title}\n{'-' * len(title)}")


def attempt(label: str, fn: Callable[[], Any]) -> Any:
    """Run one measurement, recording a failure instead of raising.

    Args:
        label: What is being measured, for the failure report.
        fn: The measurement.

    Returns:
        The result of ``fn``, or None if it raised.
    """
    try:
        return fn()
    except Exception as e:  # noqa: BLE001  # every kernel path is allowed to fail here
        tb = traceback.extract_tb(e.__traceback__)[-1]
        where = f"{os.path.basename(tb.filename)}:{tb.lineno}"
        FAILURES.append(
            {"what": label, "error": f"{type(e).__name__}: {e}", "at": where}
        )
        print(f"  FAILED {label}: {type(e).__name__}: {str(e)[:200]} ({where})")
        torch.cuda.synchronize()
        return None


def bench(fn: Callable[[], Any], warmup: int, iters: int) -> dict[str, float]:
    """Time a GPU function.

    Args:
        fn: The function.
        warmup: Untimed calls made first.
        iters: Timed calls.

    Returns:
        ``cold_ms`` for the very first call, then ``ms`` (median) and ``min_ms`` of the timed
        calls.
    """
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    cold = time.perf_counter() - t0
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return {
        "cold_ms": cold * 1e3,
        "ms": statistics.median(times) * 1e3,
        "min_ms": min(times) * 1e3,
    }


def environment(device: torch.device) -> dict[str, Any]:
    """Describe the machine and which kernel libraries loaded.

    Args:
        device: The CUDA device under test.

    Returns:
        A JSON-friendly description.
    """

    def version(name: str) -> str | None:
        try:
            from importlib.metadata import version as v

            return v(name)
        except Exception:  # noqa: BLE001
            return None

    props = torch.cuda.get_device_properties(device)
    names = (
        "triton",
        "triton-windows",
        "quack-kernels",
        "flash-attn-4",
        "warpy-kernels",
    )
    return {
        "gpu": props.name,
        "capability": list(torch.cuda.get_device_capability(device)),
        "memory_gb": round(props.total_memory / 2**30, 1),
        "arch": _dispatch.arch(device),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "available": {
            "triton": triton.available,
            "quack": quack.available,
            "flash_attn": flash_attn.available,
            "warpy": warpy.available,
            "ffpa": ffpa.available,
        },
        "versions": {n: version(n) for n in (*names, "ffpa-attn") if version(n)},
    }


def build_model(args: argparse.Namespace, device: torch.device) -> AnimaDiT:
    """Load the checkpoint, or build a full-size model with random weights.

    Args:
        args: Parsed options.
        device: Device to put the model on.

    Returns:
        The model in eval mode with frozen weights.
    """
    if args.checkpoint:
        return cast(AnimaDiT, models.load(args.checkpoint, device))
    print("no --checkpoint: random weights (timings are valid, FP8 errors are not)")
    with torch.device(device):
        model = AnimaDiT(AnimaConfig())
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("_norm"):
                p.uniform_(0.7, 1.3)
            else:
                p.normal_(std=0.5 / p.shape[-1] ** 0.5)
    return model.to(torch.bfloat16).requires_grad_(False).eval()


def runtime(model: AnimaDiT, mode: str, trim: bool = True) -> Runtime:
    """Make a runtime for one mode and set the experimental flag to match.

    Args:
        model: The model.
        mode: One of ``MODES``.
        trim: Whether to trim the text context.

    Returns:
        The runtime.
    """
    os.environ[FLAG] = "1" if mode == "fp8+attn" else "0"
    rt = Runtime(trim_context=trim)
    if mode != "bf16":
        fused.enable_fp8(model, rt)
    return rt


def make_text(args: argparse.Namespace, model: AnimaDiT, batch: int) -> torch.Tensor:
    """Make random text embeddings with realistic zero padding.

    Args:
        args: Parsed options with ``ctx_len`` and ``prompt_tokens``.
        model: The model.
        batch: Batch size.

    Returns:
        Embeddings of shape ``[batch, ctx_len, text_dim]`` in the model's compute dtype.
    """
    device = model.patch.weight.device
    g = torch.Generator(device).manual_seed(1)
    text = torch.zeros(batch, args.ctx_len, model.cfg.text_dim, device=device)
    used = min(args.prompt_tokens, args.ctx_len)
    text[:, :used] = torch.randn(
        batch, used, model.cfg.text_dim, generator=g, device=device
    )
    return text.to(model.compute_dtype)


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    """Relative error of ``a`` against ``b``.

    Args:
        a: Result.
        b: Reference.

    Returns:
        ``|a - b| / |b|`` in float32.
    """
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm()).item()


@torch.no_grad()
def check(args: argparse.Namespace, model: AnimaDiT) -> list[dict[str, Any]]:
    """Compare every kernel path against the plain PyTorch reference on a small latent.

    Args:
        args: Parsed options.
        model: The model.

    Returns:
        One row per path and mode with its relative error.
    """
    device = model.patch.weight.device
    side = args.check_size // 8
    x = noise(2, side, side, model.cfg.in_ch, 0, device).to(model.compute_dtype)
    t = torch.tensor([0.3, 0.8], device=device)
    text = make_text(args, model, 2)

    def velocity(mode: str, path: str | None, trim: bool) -> torch.Tensor:
        _dispatch.force(path)
        try:
            rt = runtime(model, mode, trim)
            return fused.forward(
                model, rt, x, t, model.encode_context(rt, text)
            ).float()
        finally:
            _dispatch.force(None)

    base = attempt("reference bf16", lambda: velocity("bf16", "reference", False))
    rows = []
    if base is None:
        return rows
    cases = [
        (m, p, tr)
        for m in MODES
        for p in ("reference", "triton", None)
        for tr in (False, True)
    ]
    for mode, path, trim in cases:
        if mode != "bf16" and path == "triton":
            continue  # the in-tree kernels alone have no FP8 product on this GPU
        label = f"check {mode} path={path or 'auto'} trim={trim}"
        out = attempt(label, lambda m=mode, p=path, tr=trim: velocity(m, p, tr))
        if out is not None:
            row = {
                "mode": mode,
                "path": path or "auto",
                "trim": trim,
                "rel_err": rel(out, base),
            }
            rows.append(row)
            print(
                f"  {mode:9s} {row['path']:9s} trim={trim!s:5s} rel err {row['rel_err']:.2e}"
            )
    return rows


@torch.no_grad()
def graph_check(args: argparse.Namespace, model: AnimaDiT) -> list[dict[str, Any]]:
    """Compare a replayed CUDA graph against the same step run eagerly, on a new input.

    Args:
        args: Parsed options.
        model: The model.

    Returns:
        One row per mode with the relative error of the replay.
    """
    device = model.patch.weight.device
    side = args.check_size // 8
    x0 = noise(1, side, side, model.cfg.in_ch, 0, device)
    x1 = noise(1, side, side, model.cfg.in_ch, 1, device)
    text = make_text(args, model, 2)
    sg = sigmas(args.steps, model.spec.shift).to(device)
    rows = []
    for mode in MODES:

        def run(mode=mode):
            rt = runtime(model, mode)
            prep = model.prepare(rt, sg[: args.steps], text)
            fn = guided(prep.run, 4.0)
            runner = Graphed(fn, (x0, *prep.per_step(0)), prep.fixed)
            want = fn(x1, *prep.per_step(3), *prep.fixed)  # not what was captured
            return rel(runner(x1, *prep.per_step(3)), want)

        err = attempt(f"graph {mode}", run)
        if err is not None:
            rows.append({"mode": mode, "rel_err": err})
            print(f"  {mode:9s} replay vs eager rel err {err:.2e}")
    return rows


@torch.no_grad()
def steps(
    args: argparse.Namespace, model: AnimaDiT, width: int, height: int
) -> list[dict[str, Any]]:
    """Time one guided denoising step in every mode, with and without a CUDA graph.

    Args:
        args: Parsed options.
        model: The model.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        One row per mode, graph setting and trim setting.
    """
    device = model.patch.weight.device
    B = args.batch
    x = noise(B, height // 8, width // 8, model.cfg.in_ch, 0, device)
    text = make_text(args, model, 2 * B)
    sg = sigmas(args.steps, model.spec.shift).to(device)
    rows = []
    for mode in MODES:
        for trim in (True, False):
            for graph in (False, True):
                label = f"step {width}x{height} {mode} trim={trim} graph={graph}"

                def run(mode=mode, trim=trim, graph=graph):
                    rt = runtime(model, mode, trim)
                    prep = model.prepare(rt, sg[: args.steps], text)
                    fn = guided(prep.run, 4.0)
                    varying = (x, *prep.per_step(0))
                    torch.cuda.reset_peak_memory_stats(device)
                    if not graph:
                        r = bench(
                            lambda: fn(*varying, *prep.fixed), args.warmup, args.iters
                        )
                    else:
                        t0 = time.perf_counter()
                        runner = Graphed(fn, varying, prep.fixed)
                        torch.cuda.synchronize()
                        capture = (time.perf_counter() - t0) * 1e3
                        r = bench(lambda: runner(*varying), args.warmup, args.iters)
                        r["cold_ms"] = capture
                    r["peak_gb"] = torch.cuda.max_memory_allocated(device) / 2**30
                    return r

                r = attempt(label, run)
                if r is not None:
                    row = {
                        "size": f"{width}x{height}",
                        "mode": mode,
                        "trim": trim,
                        "graph": graph,
                        **r,
                    }
                    rows.append(row)
                    print(
                        f"  {mode:9s} trim={trim!s:5s} graph={graph!s:5s} {r['ms']:8.2f} ms/step "
                        f"{1e3 / r['ms']:6.2f} it/s  cold {r['cold_ms'] / 1e3:6.2f} s  peak {r['peak_gb']:.1f} GB"
                    )
    return rows


@torch.no_grad()
def ops(
    args: argparse.Namespace, model: AnimaDiT, width: int, height: int
) -> list[dict[str, Any]]:
    """Time each op of a block on its own, at the shapes of one guided step.

    Args:
        args: Parsed options.
        model: The model.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        One row per op with its time and how often a block runs it.
    """
    c, b = model.cfg, cast(Any, model.blocks[0])
    device, dt = model.patch.weight.device, model.compute_dtype
    R, hw = 2 * args.batch, (height // 16, width // 16)
    T, D = hw[0] * hw[1], c.dim
    g = torch.Generator(device).manual_seed(0)
    rand = lambda *shape: torch.randn(*shape, generator=g, device=device)
    hid, z = rand(R, T, D).to(dt), rand(R, T, D).to(dt)
    gate, shift, scale = (
        rand(R, D).to(dt),
        rand(R, D).to(dt),
        (rand(R, D) * 0.1 + 1).to(dt),
    )
    rt = Runtime()
    cos, sin = model.tables(rt, *hw, device)
    text = make_text(args, model, R)
    ctx = model.encode_context(Runtime(trim_context=True), text)
    ctx_full = model.encode_context(Runtime(trim_context=False), text)
    qkv = b.qkv(hid)
    rp = (b.q_norm, b.k_norm, cos, sin, c.heads)
    q, k, v = rope.qkv_norm_rope(qkv, *rp, c.qk_eps)
    xq = norm.head_norm(b.xq(hid), b.xq_norm, c.heads, c.qk_eps)
    att = attention.attention(q, k, v)
    cases: list[tuple[str, str, int, Callable[[], Any]]] = [
        ("bf16", "linear qkv", 1, lambda: gemm.linear(hid, b.qkv.weight)),
        ("bf16", "qkv_norm_rope", 1, lambda: rope.qkv_norm_rope(qkv, *rp, c.qk_eps)),
        ("bf16", "attention", 1, lambda: attention.attention(q, k, v)),
        ("bf16", "linear o1/o2/xq", 3, lambda: gemm.linear(hid, b.o1.weight)),
        (
            "bf16",
            "residual_adaln",
            3,
            lambda: adaln.residual_adaln(hid, z, gate, shift, scale),
        ),
        (
            "bf16",
            "head_norm",
            1,
            lambda: norm.head_norm(hid, b.xq_norm, c.heads, c.qk_eps),
        ),
        ("bf16", "cross-attention trimmed", 1, lambda: ctx.attend(0, xq)),
        ("bf16", "cross-attention untrimmed", 0, lambda: ctx_full.attend(0, xq)),
        ("bf16", "mlp", 1, lambda: gemm.mlp(hid, b.up.weight, b.down.weight)),
    ]
    fp8 = attempt("ops: quantize block 0", lambda: runtime(model, "fp8"))
    if fp8 is not None and fp8.fp8 is not None and fp8.fp8_scales is not None:
        w, s = fp8.fp8[0], fp8.fp8_scales[0]
        hq, hs = adaln.adaln_fp8(z, shift, scale)
        aq, as_ = gemm.rowquant(att)
        up, up_sc = gemm.fp8_linear_unscaled(hq, hs, w["up"])
        up_true = gemm.fp8_linear(hq, hs, w["up"])
        gq, gs = gemm.rowquant(up_true, gelu=True)
        o, o_sc = gemm.fp8_linear_unscaled(aq, as_, w["o1"])
        lin, unscaled = gemm.fp8_linear, gemm.fp8_linear_unscaled
        radaln = adaln.residual_adaln_fp8
        cases += [
            ("fp8", "fp8_linear qkv, scaled", 1, lambda: lin(hq, hs, w["qkv"])),
            ("fp8", "fp8_linear xq, scaled", 1, lambda: lin(hq, hs, w["xq"])),
            (
                "fp8",
                "fp8_linear o1/o2, unscaled",
                2,
                lambda: unscaled(aq, as_, w["o1"]),
            ),
            ("fp8", "fp8_linear up, unscaled", 1, lambda: unscaled(hq, hs, w["up"])),
            (
                "fp8",
                "fp8_linear down, unscaled",
                1,
                lambda: unscaled(gq, gs, w["down"]),
            ),
            ("fp8", "fp8_linear up, scaled (old)", 0, lambda: lin(hq, hs, w["up"])),
            ("fp8", "rowquant", 2, lambda: gemm.rowquant(att)),
            (
                "fp8",
                "rowquant gelu, folded scales",
                1,
                lambda: gemm.rowquant(up, True, up_sc),
            ),
            (
                "fp8",
                "rowquant gelu, plain (old)",
                0,
                lambda: gemm.rowquant(up_true, True),
            ),
            (
                "fp8",
                "residual_adaln_fp8, folded scales",
                3,
                lambda: radaln(o, z, gate, shift, scale, x_scales=o_sc),
            ),
            (
                "fp8",
                "residual_adaln_fp8, plain (old)",
                0,
                lambda: radaln(hid, z, gate, shift, scale),
            ),
            ("fp8", "qkv_norm_rope", 1, lambda: rope.qkv_norm_rope(qkv, *rp, c.qk_eps)),
            ("fp8", "attention", 1, lambda: attention.attention(q, k, v)),
            (
                "fp8",
                "head_norm",
                1,
                lambda: norm.head_norm(hid, b.xq_norm, c.heads, c.qk_eps),
            ),
            ("fp8", "cross-attention trimmed", 1, lambda: ctx.attend(0, xq)),
            (
                "fp8+attn",
                "qkv_norm_rope_fp8",
                0,
                lambda: rope.qkv_norm_rope_fp8(qkv, *rp, s["q"], s["k"], 1.0, c.qk_eps),
            ),
            (
                "fp8+attn",
                "head_norm_fp8",
                0,
                lambda: norm.head_norm_fp8(hid, b.xq_norm, c.heads, s["xq"], c.qk_eps),
            ),
            ("fp8+attn", "attention_fp8", 0, lambda: attention.attention_fp8(q, k, v)),
        ]
    rows = []
    for mode, name, count, fn in cases:
        r = attempt(
            f"op {mode} {name}", lambda fn=fn: bench(fn, args.warmup, args.iters)
        )
        if r is not None:
            per_eval = r["ms"] * count * c.layers
            rows.append(
                {
                    "mode": mode,
                    "op": name,
                    "per_block": count,
                    **r,
                    "ms_per_eval": per_eval,
                }
            )
            print(
                f"  {mode:9s} {name:36s} {r['ms']:8.3f} ms  x{count} x{c.layers} = {per_eval:8.2f} ms"
            )
    for mode in ("bf16", "fp8"):
        total = sum(r["ms_per_eval"] for r in rows if r["mode"] == mode)
        print(f"  {mode}: ops sum to {total:.1f} ms per evaluation")
    return rows


def tune(
    args: argparse.Namespace, model: AnimaDiT, sizes: list[tuple[int, ...]]
) -> list[dict[str, Any]]:
    """Tune the GPU kernels for every shape the other sections measure.

    Args:
        args: Parsed options.
        model: The model.
        sizes: Image sizes as ``(width, height)``.

    Returns:
        One row per size and batch with the time tuning took.
    """
    heading("tune")
    rows = []
    batches = sorted({2 * args.batch, 1, *args.train_batches})
    for width, height in sizes:
        for batch in batches:
            for mode in ("bf16", "fp8"):
                rt = runtime(model, mode)
                t0 = time.perf_counter()
                n = fused.warmup(
                    model, rt, height // 8, width // 8, batch, args.ctx_len
                )
                seconds = time.perf_counter() - t0
                rows.append(
                    {
                        "size": f"{width}x{height}",
                        "batch": batch,
                        "mode": mode,
                        "s": seconds,
                    }
                )
                print(
                    f"  {width}x{height} batch {batch} {mode}: {n} kernels in {seconds:.1f} s"
                )
    return rows


TRAIN_VARIANTS = (
    ("bf16", False, False, False),
    ("fp8 attention", False, False, True),
    ("fp8 base forward+backward", True, True, False),
    ("fp8 base forward+backward + fp8 attention", True, True, True),
)


def train(
    args: argparse.Namespace, model: AnimaDiT, width: int, height: int, batch: int
) -> dict[str, Any]:
    """Time a LoRA training step in every precision variant, eager and as a CUDA graph.

    Each variant's gradient is compared with the bf16 eager gradient.

    Args:
        args: Parsed options with ``lora_rank``.
        model: The model, without adapters.
        width: Image width in pixels.
        height: Image height in pixels.
        batch: Samples per step.

    Returns:
        The timing rows and a profiler table per traced variant.
    """
    from torch.profiler import ProfilerActivity, profile

    device, dt = model.patch.weight.device, model.compute_dtype
    size = f"{width}x{height} b{batch}"
    x0 = noise(batch, height // 8, width // 8, model.cfg.in_ch, 0, device)
    eps = noise(batch, height // 8, width // 8, model.cfg.in_ch, 1, device)
    sigma = torch.linspace(0.2, 0.9, batch, device=device)
    s = sigma.view(batch, 1, 1, 1)
    inputs = (
        ((1 - s) * x0 + s * eps).to(dt),
        sigma,
        make_text(args, model, batch),
        eps - x0,
    )
    rows: list[dict[str, Any]] = []
    tables: dict[str, str] = {}
    heading(f"train {size}")

    def add(label: str, fn: Callable[[], Any], note: str = "") -> float | None:
        def run():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            r = bench(fn, args.warmup, args.iters)
            r["peak_gb"] = torch.cuda.max_memory_allocated(device) / 2**30
            return r

        r = attempt(f"train {size} {label}", run)
        if r is None:
            return None
        rows.append({"size": size, "what": label, "note": note, **r})
        print(f"  {label:58s} {r['ms']:8.2f} ms  peak {r['peak_gb']:5.1f} GB  {note}")
        return r["ms"]

    def infer():
        with torch.no_grad():
            rt = Runtime()
            xt, sg, text, _ = inputs
            return fused.forward(model, rt, xt, sg, model.encode_context(rt, text))

    add("inference forward, no adapter", infer)
    params = L.add_lora(model, rank=args.lora_rank)
    flat = lambda grads: torch.cat(
        [g.flatten().float() for g in grads if g is not None]
    )
    cosine = torch.nn.functional.cosine_similarity
    try:
        with torch.no_grad():
            for p in params[1::2]:
                p.normal_(std=0.02)  # a zero B would hide the adapter's gradient path
        add("inference forward, adapter attached", infer)
        reference = None
        for name, base, backward, attn in TRAIN_VARIANTS:
            if base:
                L.enable_fp8_base(model, backward)
            else:
                L.disable_fp8_base(model)
            for ckpt in (False, True):
                rt = Runtime(checkpointing=ckpt, fp8_attention_grad=attn)
                tag = f"{name}, checkpointing {'on' if ckpt else 'off'}"

                def loss_fn(xt, sg, text, target, rt=rt):
                    v = model.velocity(rt, xt, sg, text)
                    return torch.nn.functional.mse_loss(v.float(), target)

                def forward_backward(loss_fn=loss_fn, with_input=False):
                    for p in params:
                        p.grad = None  # a graphed step leaves its gradients attached
                    xt = inputs[0].detach().requires_grad_(with_input)
                    loss_fn(xt, *inputs[1:]).backward()
                    grads = [p.grad for p in params]
                    for p in params:
                        p.grad = None
                    return (grads, xt.grad) if with_input else grads

                def measure(
                    tag=tag, loss_fn=loss_fn, forward_backward=forward_backward
                ):
                    nonlocal reference
                    grads, dx = forward_backward(with_input=True)
                    eager = flat(grads)
                    if reference is None:
                        reference = (eager, dx)
                    ref, ref_dx = reference
                    note = (
                        f"vs bf16: LoRA grad rel err {rel(eager, ref):.1e} "
                        f"cos {cosine(eager, ref, dim=0).item():.4f}, "
                        f"input grad rel err {rel(dx, ref_dx):.1e}"
                    )
                    add(f"{tag}: eager", forward_backward, note)
                    runner = GraphedStep(loss_fn, params)
                    runner(*inputs)
                    got = flat([p.grad for p in params])
                    note = f"graph vs eager gradient rel err {rel(got, eager):.1e}"
                    add(f"{tag}: CUDA graph", lambda: runner(*inputs), note)

                attempt(f"train {size} {tag}", measure)
                traced = (
                    name in (TRAIN_VARIANTS[0][0], TRAIN_VARIANTS[-1][0]) and not ckpt
                )
                if traced and "trace" not in args.skip:

                    def table(forward_backward=forward_backward):
                        for _ in range(2):
                            forward_backward()
                        kinds = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
                        with profile(activities=kinds) as prof:
                            for _ in range(3):
                                forward_backward()
                            torch.cuda.synchronize()
                        return prof.key_averages().table(
                            sort_by="self_cuda_time_total", row_limit=args.top
                        )

                    out = attempt(f"train {size} {tag} trace", table)
                    if out is not None:
                        tables[tag] = out
                        heading(f"{size}, {tag}, three forward + backward passes")
                        print(out)
    finally:
        L.remove_lora(model, merge=False)
        for p in model.parameters():
            p.grad = None
        torch.cuda.empty_cache()
    return {"rows": rows, "tables": tables}


@torch.no_grad()
def trace(
    args: argparse.Namespace, model: AnimaDiT, width: int, height: int
) -> dict[str, str]:
    """Collect the costliest GPU kernels of one ungraphed step per mode.

    Args:
        args: Parsed options.
        model: The model.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        A profiler table per mode.
    """
    from torch.profiler import ProfilerActivity, profile

    device = model.patch.weight.device
    x = noise(args.batch, height // 8, width // 8, model.cfg.in_ch, 0, device)
    text = make_text(args, model, 2 * args.batch)
    sg = sigmas(args.steps, model.spec.shift).to(device)
    tables = {}
    for mode in MODES:

        def run(mode=mode):
            rt = runtime(model, mode)
            prep = model.prepare(rt, sg[: args.steps], text)
            fn = guided(prep.run, 4.0)
            call = lambda: fn(x, *prep.per_step(0), *prep.fixed)
            for _ in range(3):
                call()
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
            ) as prof:
                for _ in range(3):
                    call()
                torch.cuda.synchronize()
            return prof.key_averages().table(
                sort_by="self_cuda_time_total", row_limit=args.top
            )

        table = attempt(f"trace {mode}", run)
        if table is not None:
            tables[mode] = table
            heading(f"{mode}, three steps")
            print(table)
    return tables


@torch.no_grad()
def decode(
    args: argparse.Namespace, device: torch.device, width: int, height: int
) -> list[dict[str, Any]]:
    """Time the VAE decode with and without a CUDA graph.

    Args:
        args: Parsed options with ``vae``.
        device: The CUDA device.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        One row per graph setting.
    """
    decoder = load_decoder(args.vae, device)
    x = noise(args.batch, height // 8, width // 8, decoder.mean.shape[0], 0, device)
    x1 = noise(args.batch, height // 8, width // 8, decoder.mean.shape[0], 1, device)

    def replay():
        decoder.decode(x, graph=True)
        return rel(decoder.decode(x1, graph=True), decoder.decode(x1))

    err = attempt(f"decode replay {width}x{height}", replay)
    if err is not None:
        print(f"  replay vs eager rel err {err:.2e}")
    rows = []
    for graph in (False, True):
        label = f"decode {width}x{height} graph={graph}"
        r = attempt(
            label,
            lambda g=graph: bench(
                lambda: decoder.decode(x, graph=g), args.warmup, args.iters
            ),
        )
        if r is not None:
            rows.append({"size": f"{width}x{height}", "graph": graph, **r})
            print(
                f"  graph={graph!s:5s} {r['ms']:8.2f} ms  cold {r['cold_ms'] / 1e3:6.2f} s"
            )
    return rows


def main(argv: list[str] | None = None) -> int:
    """Check every kernel path against the reference and time the model on this GPU.

    Args:
        argv: Command-line arguments. None reads ``sys.argv``.

    Returns:
        0 if nothing failed, 1 otherwise.
    """
    ap = argparse.ArgumentParser(description="Profile pneumam on this GPU.")
    ap.add_argument("--checkpoint", help="Model checkpoint. Default: random weights.")
    ap.add_argument("--vae", help="VAE file. Without it the decode is not timed.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--sizes", nargs="+", default=["1024x1024", "832x1216"], metavar="WxH"
    )
    ap.add_argument(
        "--batch", type=int, default=1, help="Images per step. Guidance doubles it."
    )
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--ctx-len", type=int, default=512, help="Padded text length.")
    ap.add_argument(
        "--prompt-tokens", type=int, default=100, help="Tokens that are not padding."
    )
    ap.add_argument(
        "--check-size", type=int, default=512, help="Image side of the checks."
    )
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--top", type=int, default=25, help="Rows of each profiler table.")
    ap.add_argument(
        "--skip",
        nargs="*",
        default=[],
        choices=["check", "graph", "steps", "ops", "trace", "decode", "train"],
    )
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--train-batches", type=int, nargs="+", default=[1, 4])
    ap.add_argument(
        "--tune",
        action="store_true",
        help="Tune the GPU kernels for every measured shape first. Slow the first time.",
    )
    ap.add_argument("--out", default="profile_out")
    args = ap.parse_args(argv)

    device = torch.device(args.device)
    report: dict[str, Any] = {"args": vars(args), "environment": environment(device)}
    print(json.dumps(report["environment"], indent=1))
    model = build_model(args, device)
    sizes = [tuple(int(v) for v in s.lower().split("x")) for s in args.sizes]
    if args.tune:
        report["tune"] = attempt("tune", lambda: tune(args, model, sizes))
    sections: list[tuple[str, Callable[[], Any]]] = [
        ("check", lambda: check(args, model)),
        ("graph", lambda: graph_check(args, model)),
        ("steps", lambda: [r for w, h in sizes for r in steps(args, model, w, h)]),
        (
            "ops",
            lambda: [
                {"size": f"{w}x{h}", **r}
                for w, h in sizes
                for r in ops(args, model, w, h)
            ],
        ),
        ("trace", lambda: trace(args, model, *sizes[0])),
        (
            "train",
            lambda: [
                train(args, model, w, h, b)
                for w, h in sizes
                for b in args.train_batches
            ],
        ),
    ]
    if args.vae:
        sections.append(
            (
                "decode",
                lambda: [r for w, h in sizes for r in decode(args, device, w, h)],
            )
        )
    for name, fn in sections:
        if name in args.skip:
            continue
        heading(name)
        report[name] = attempt(name, fn)
    report["failures"] = FAILURES
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "profile.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=1)
    print(f"\n{len(FAILURES)} failure(s). Report written to {path}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
