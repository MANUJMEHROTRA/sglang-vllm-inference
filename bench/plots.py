"""Charts, with the rules baked in so every notebook looks like one system.

Three conventions here are deliberate and should not be "fixed":

1. **Colour follows the engine, not its rank.** vLLM is the same blue in every
   chart in this repo, whether it won or lost that particular experiment. A
   reader who learned "blue = vLLM" is never misled by a filter that drops a
   series.
2. **Colour is never the only channel.** Every scatter also varies marker
   shape, and every multi-series chart carries a legend. That covers colour
   vision deficiency, greyscale printing, and the small-multiple case where a
   legend is far away.
3. **Never two y-axes.** Throughput and latency on one plot with two scales
   invents a relationship that is not in the data. When both matter, they go on
   the two axes of a *scatter* (the Pareto view) -- which is the honest version
   of that chart and also the more useful one.

The palette is a validated categorical set (adjacent-pair CVD Delta E >= 8 in
OKLab x100). The HuggingFace baseline is deliberately muted grey rather than a
series colour: it is the reference line everything else is measured against,
not a contender.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from .schema import RunResult

# Validated categorical slots, light surface.
PALETTE = {
    "blue": "#2a78d6",
    "orange": "#eb6834",
    "aqua": "#1baf7a",
    "yellow": "#eda100",
    "magenta": "#e87ba4",
    "green": "#008300",
    "violet": "#6250d6",
    "red": "#e34948",
}

INK = {
    "surface": "#fcfcfb",
    "primary": "#0b0b0b",
    "secondary": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
}

# Fixed, entity-stable assignment. Order matters: adjacent slots in this list
# are the pairs most likely to appear side by side in a bar chart.
ENGINE_COLORS: Dict[str, str] = {
    "vllm": PALETTE["blue"],
    "sglang": PALETTE["orange"],
    "llamacpp": PALETTE["aqua"],
    "tensorrt_llm": PALETTE["yellow"],
    "triton": PALETTE["magenta"],
    "onnxruntime": PALETTE["green"],
    "torch_compile": PALETTE["violet"],
    "hf": INK["muted"],          # the baseline: present, recessive, not a rival
}

ENGINE_MARKERS: Dict[str, str] = {
    "vllm": "o",
    "sglang": "s",
    "llamacpp": "^",
    "tensorrt_llm": "D",
    "triton": "v",
    "onnxruntime": "P",
    "torch_compile": "X",
    "hf": ".",
}

ENGINE_LABELS: Dict[str, str] = {
    "vllm": "vLLM",
    "sglang": "SGLang",
    "llamacpp": "llama.cpp",
    "tensorrt_llm": "TensorRT-LLM",
    "triton": "Triton Server",
    "onnxruntime": "ONNX Runtime",
    "torch_compile": "torch.compile",
    "hf": "HF transformers (baseline)",
}

FIGURE_DIR = Path("results/figures")


def color_for(engine: str) -> str:
    return ENGINE_COLORS.get(engine, PALETTE["blue"])


def marker_for(engine: str) -> str:
    return ENGINE_MARKERS.get(engine, "o")


def label_for(engine: str) -> str:
    return ENGINE_LABELS.get(engine, engine)


def use_theme() -> None:
    """Apply the house style. Call once at the top of each notebook.

    Thin marks, hairline solid grid one shade off the surface, no top/right
    spines, no dashed gridlines (dashing reads as "projection" when it is just
    a grid).
    """
    import matplotlib as mpl  # noqa: PLC0415

    mpl.rcParams.update(
        {
            "figure.figsize": (8.0, 4.8),
            "figure.dpi": 120,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
            "figure.facecolor": INK["surface"],
            "axes.facecolor": INK["surface"],
            "savefig.facecolor": INK["surface"],
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.titlepad": 12,
            "axes.labelsize": 10,
            "axes.labelcolor": INK["secondary"],
            "axes.edgecolor": INK["axis"],
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": INK["grid"],
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",
            "xtick.color": INK["muted"],
            "ytick.color": INK["muted"],
            "xtick.labelcolor": INK["secondary"],
            "ytick.labelcolor": INK["secondary"],
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 2.0,
            "lines.markersize": 6,
        }
    )


def save(fig, name: str, directory: Optional[Path] = None) -> Path:
    """Save a figure into ``results/figures/`` and return its path."""
    directory = Path(directory or FIGURE_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.png"
    fig.savefig(path)
    print(f"saved {path}")
    return path


def _group(results: Sequence[RunResult], key=lambda r: r.engine.engine) -> Dict[str, List[RunResult]]:
    groups: Dict[str, List[RunResult]] = {}
    for r in results:
        groups.setdefault(key(r), []).append(r)
    return groups


def sweep_curve(
    results: Sequence[RunResult],
    y: str = "output_throughput_tok_s",
    x: str = "concurrency",
    ylabel: Optional[str] = None,
    title: str = "",
    scale: float = 1.0,
    ax=None,
    logx: bool = True,
):
    """One line per engine across a concurrency (or rate) sweep.

    ``scale`` multiplies the y values -- pass 1000 to show seconds as ms.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415

    ax = ax or plt.subplots()[1]
    for engine, runs in _group(results).items():
        pts = []
        for r in runs:
            xv = r.workload.concurrency if x == "concurrency" else r.workload.request_rate
            if xv is None:
                continue
            pts.append((xv, r.metrics.get(y, float("nan")) * scale))
        if not pts:
            continue
        pts.sort()
        xs, ys = zip(*pts)
        ax.plot(
            xs, ys,
            color=color_for(engine),
            marker=marker_for(engine),
            markeredgecolor=INK["surface"],   # 2px surface ring on overlap
            markeredgewidth=1.5,
            label=label_for(engine),
        )
    if logx:
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(lambda v, _: f"{int(v)}")
    ax.set_xlabel("concurrent requests" if x == "concurrency" else "offered rate (req/s)")
    ax.set_ylabel(ylabel or y)
    if title:
        ax.set_title(title)
    if len(_group(results)) >= 2:
        ax.legend()
    return ax


def pareto(
    results: Sequence[RunResult],
    x: str = "output_throughput_tok_s",
    y: str = "ttft_s_p99",
    yscale: float = 1000.0,
    title: str = "Throughput vs tail latency",
    annotate: bool = True,
    ax=None,
):
    """The chart that answers "which engine should I use?".

    Up-and-left is better: more throughput at lower tail latency. Each engine
    traces a curve as concurrency rises; where the curve bends upward is the
    point past which extra load buys latency, not capacity.

    This is the honest replacement for a dual-axis throughput/latency plot:
    both measures keep their own axis and the trade-off is a *shape* you can
    read, not two lines whose alignment you invented.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415

    ax = ax or plt.subplots()[1]
    for engine, runs in _group(results).items():
        pts = sorted(
            (r.metrics.get(x, float("nan")), r.metrics.get(y, float("nan")) * yscale,
             r.workload.concurrency)
            for r in runs
        )
        if not pts:
            continue
        xs, ys, cs = zip(*pts)
        ax.plot(
            xs, ys,
            color=color_for(engine),
            marker=marker_for(engine),
            markeredgecolor=INK["surface"],
            markeredgewidth=1.5,
            label=label_for(engine),
            alpha=0.95,
        )
        if annotate:
            # Direct-label the endpoint only. A number on every point is noise.
            ax.annotate(
                f"c={cs[-1]}",
                (xs[-1], ys[-1]),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=8,
                color=INK["muted"],
            )
    ax.set_xlabel("output throughput (tok/s)  ->  better")
    ax.set_ylabel("p99 TTFT (ms)  ->  worse")
    ax.set_title(title)
    ax.grid(axis="both")
    if len(_group(results)) >= 2:
        ax.legend()
    return ax


def cdf(
    results: Sequence[RunResult],
    metric: str = "ttft",
    scale: float = 1000.0,
    title: str = "",
    ax=None,
    logx: bool = False,
):
    """Empirical CDF of per-request TTFT or per-token ITL.

    Worth more than any bar chart of means. A bar chart says "engine A is
    faster"; the CDF shows you *where* -- two engines with identical means can
    have completely different tails, and the tail is what users complain
    about. Read it as: "x% of requests finished within y ms".
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415

    ax = ax or plt.subplots()[1]
    for engine, runs in _group(results).items():
        values: List[float] = []
        for r in runs:
            for rec in r.requests:
                if not rec.success:
                    continue
                if metric == "ttft" and rec.ttft_s is not None:
                    values.append(rec.ttft_s * scale)
                elif metric == "itl":
                    values.extend(v * scale for v in rec.itls_s)
                elif metric == "e2e" and rec.e2e_s is not None:
                    values.append(rec.e2e_s * scale)
        if not values:
            continue
        values.sort()
        ys = [100.0 * (i + 1) / len(values) for i in range(len(values))]
        ax.plot(values, ys, color=color_for(engine), label=label_for(engine))
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel({"ttft": "TTFT (ms)", "itl": "inter-token latency (ms)",
                   "e2e": "end-to-end latency (ms)"}[metric])
    ax.set_ylabel("% of requests")
    ax.set_ylim(0, 100)
    ax.set_title(title or f"{metric.upper()} distribution")
    ax.grid(axis="both")
    if len(_group(results)) >= 2:
        ax.legend()
    return ax


def grouped_bar(
    labels: Sequence[str],
    series: Dict[str, Sequence[float]],
    ylabel: str = "",
    title: str = "",
    colors: Optional[Dict[str, str]] = None,
    annotate: bool = True,
    ax=None,
):
    """Grouped bars with a 2px surface gap between adjacent fills.

    ``series`` maps a series name to one value per label. Use for comparing a
    handful of discrete configurations (quant levels, flag on/off); use
    ``sweep_curve`` for anything with a continuous x.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    ax = ax or plt.subplots()[1]
    n = len(series)
    idx = np.arange(len(labels))
    width = 0.8 / max(1, n)
    for i, (name, values) in enumerate(series.items()):
        color = (colors or {}).get(name) or color_for(name) or list(PALETTE.values())[i]
        pos = idx - 0.4 + width * (i + 0.5)
        bars = ax.bar(
            pos, values, width * 0.94,       # 0.94 leaves the surface gap
            color=color,
            label=label_for(name) if name in ENGINE_LABELS else name,
            linewidth=0,
        )
        if annotate and len(labels) * n <= 24:
            for bar, value in zip(bars, values):
                ax.annotate(
                    f"{value:,.0f}" if abs(value) >= 10 else f"{value:,.2f}",
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=8, color=INK["secondary"],
                )
    ax.set_xticks(idx)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if n >= 2:
        ax.legend()
    return ax


def roofline(
    gpu_peak_tflops: float,
    gpu_bandwidth_gb_s: float,
    points: Sequence[tuple],
    title: str = "Roofline",
    ax=None,
):
    """Roofline plot: achievable FLOPS vs arithmetic intensity.

    ``points`` is a sequence of ``(label, intensity_flops_per_byte,
    achieved_tflops)``.

    The ridge point -- peak FLOPS / peak bandwidth -- splits the plot. Left of
    it you are bandwidth-limited and the only wins are moving fewer bytes
    (quantization, fusion, better data layout). Right of it you are
    compute-limited and the wins are better math (tensor cores, bigger tiles).
    Decode lands far to the left; prefill lands near or right of the ridge.
    Knowing which side you are on tells you what to optimise, which is the
    whole subject of notebook 08.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    ax = ax or plt.subplots()[1]
    ridge = gpu_peak_tflops * 1e12 / (gpu_bandwidth_gb_s * 1e9)
    xs = np.logspace(-1, 4, 200)
    ceiling = np.minimum(xs * gpu_bandwidth_gb_s * 1e9 / 1e12, gpu_peak_tflops)
    ax.plot(xs, ceiling, color=INK["secondary"], linewidth=1.5, label="roofline")
    ax.axvline(ridge, color=INK["axis"], linewidth=1.0)
    ax.annotate(
        f"ridge {ridge:.0f} FLOP/byte",
        (ridge, gpu_peak_tflops * 0.5),
        rotation=90, fontsize=8, color=INK["muted"],
        textcoords="offset points", xytext=(4, 0),
    )
    for i, (label, intensity, tflops) in enumerate(points):
        color = list(PALETTE.values())[i % len(PALETTE)]
        ax.plot(
            [intensity], [tflops],
            marker="o", markersize=9, color=color,
            markeredgecolor=INK["surface"], markeredgewidth=1.5,
        )
        ax.annotate(
            label, (intensity, tflops),
            textcoords="offset points", xytext=(8, -3),
            fontsize=9, color=INK["secondary"],
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("arithmetic intensity (FLOP / byte)")
    ax.set_ylabel("achieved TFLOP/s")
    ax.set_title(title)
    ax.grid(axis="both")
    return ax


def table(results: Sequence[RunResult], columns: Optional[Sequence[str]] = None):
    """The table view that every chart in this repo is required to have.

    Returns a pandas DataFrame. A chart without a table twin hides values
    behind hover, and nothing in a notebook should be readable only by
    pointing at it.
    """
    import pandas as pd  # noqa: PLC0415

    columns = columns or [
        "ttft_s_p50", "ttft_s_p99", "tpot_s_mean", "itl_s_p99",
        "output_throughput_tok_s", "request_throughput_req_s", "goodput_req_s",
    ]
    rows = []
    for r in results:
        row = {
            "engine": label_for(r.engine.engine),
            "version": r.engine.version,
            "dtype": r.engine.dtype,
            "workload": r.workload.name,
            "concurrency": r.workload.concurrency,
            "synthetic": r.synthetic,
        }
        for c in columns:
            name, value = _display(c, r.metrics.get(c))
            row[name] = value
        rows.append(row)
    return pd.DataFrame(rows)


# Latency distributions are stored in seconds and read in milliseconds. These
# suffixes mark a per-request/per-token latency; throughput columns end in
# `_tok_s` / `_req_s` and must never be rescaled.
_LATENCY_SUFFIXES = ("_mean", "_p50", "_p90", "_p99", "_max")


def _display(column: str, value):
    """(display_name, display_value) -- seconds become milliseconds."""
    is_latency = "_s_" in column and column.endswith(_LATENCY_SUFFIXES)
    if not is_latency:
        return column, value
    name = column.replace("_s_", "_ms_")
    return name, (value * 1000 if value is not None else None)
