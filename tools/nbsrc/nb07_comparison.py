"""Notebook 07 - The comparison: every result on one set of axes."""

from tools.nbgen import code, md, setup_cells

NAME = "07_comparison.ipynb"
TITLE = "07 - The comparison: which engine, when, and why"

CELLS = setup_cells(
    NAME,
    TITLE,
    "Everything measured in notebooks 00-06 goes on one set of axes. This notebook "
    "runs on CPU -- it only reads result files -- so you can re-plot and re-reason "
    "without burning GPU time.",
    "~5 minutes",
    needs_gpu=False,
) + [

    md("""
## Ground rules for a fair comparison

Before any chart, the filters. A comparison is only as honest as what it
refuses to put side by side.

1. **Same GPU.** A T4 run and an A100 run are different experiments. We group
   by GPU and never average across them.
2. **Same model and dtype** -- or, where they differ (llama.cpp runs Q4_K_M by
   necessity), the chart says so in the label rather than hiding it.
3. **Same workload.** `chat` and `rag` are plotted separately. Averaging a
   prefix-cache-friendly workload with a hostile one produces a number that
   describes neither.
4. **Same measurement path.** Every generative engine was driven through the
   same OpenAI-compatible streaming client, so client overhead is a shared
   constant.
5. **Generative and encoder results never share an axis.** TTFT does not exist
   for BERT.
6. **Synthetic rows are marked.** Any result with `synthetic=True` is
   illustrative, not measured here, and is drawn differently.
"""),

    code('''
from collections import Counter
from bench.schema import load_dir
from bench.plots import use_theme, label_for
import pandas as pd

use_theme()
runs = load_dir("results")
print(f"loaded {len(runs)} result files\\n")

gen = [r for r in runs if r.engine.task == "generative"]
enc = [r for r in runs if r.engine.task == "encode"]

print("generative runs by engine:")
for engine, n in Counter(r.engine.engine for r in gen).most_common():
    print(f"  {label_for(engine):<28} {n:>3}")
print("\\nencoder runs by engine:")
for engine, n in Counter(r.engine.engine for r in enc).most_common():
    print(f"  {label_for(engine):<28} {n:>3}")
print("\\nGPUs present:", dict(Counter(r.hardware.gpu_name for r in runs)))
print("workloads present:", dict(Counter(r.workload.name for r in runs)))

synthetic = [r for r in runs if r.synthetic]
if synthetic:
    print(f"\\n{len(synthetic)} rows are illustrative, not measured here.")
'''),

    code('''
# Pick the dominant GPU and keep only runs from it, so nothing is compared
# across hardware. If you have results from several GPUs, change TARGET_GPU
# and re-run -- the conclusions may genuinely differ.
gpu_counts = Counter(r.hardware.gpu_name for r in gen if r.hardware.gpu_name)
TARGET_GPU = gpu_counts.most_common(1)[0][0] if gpu_counts else None
print(f"comparing on: {TARGET_GPU}")

def on_gpu(rs):
    return [r for r in rs if r.hardware.gpu_name == TARGET_GPU and not r.synthetic]

chat_sweeps = [r for r in on_gpu(gen)
               if r.workload.name == "chat" and r.workload.mode == "closed"
               and r.workload.input_len == 256 and r.workload.output_len == 256]

print(f"\\n{len(chat_sweeps)} comparable chat-sweep points")
engines_present = sorted({r.engine.engine for r in chat_sweeps})
print("engines:", [label_for(e) for e in engines_present])
if len(engines_present) < 2:
    print("\\nOnly one engine has comparable runs. Run notebooks 01-04 on the")
    print("same GPU to populate the comparison, or read on with what you have.")
'''),

    md("""
## The Pareto view

The chart that answers "which engine should I use?". Each engine traces a curve
as concurrency rises. **Up and to the left is better**: more throughput at lower
tail latency.

Do not read this as a ranking. Read it as a shape:

- an engine whose curve stays flat and then turns sharply up has a scheduler
  that holds latency well until it saturates;
- an engine that climbs steadily from the start is trading latency for
  throughput the whole way;
- where two curves cross is the concurrency at which your answer changes.
""".rstrip()),

    code('''
import matplotlib.pyplot as plt
from bench.plots import cdf, pareto, save, sweep_curve, table

if len(engines_present) >= 1:
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    pareto(chat_sweeps, title=f"Throughput vs tail latency -- {TARGET_GPU}, chat workload",
           ax=ax)
    save(fig, "07_pareto"); plt.show()
'''),

    code('''
if chat_sweeps:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    sweep_curve(chat_sweeps, "output_throughput_tok_s", ylabel="output tok/s",
                title="Throughput", ax=axes[0][0])
    sweep_curve(chat_sweeps, "ttft_s_p99", scale=1000, ylabel="p99 TTFT (ms)",
                title="Tail time-to-first-token", ax=axes[0][1])
    sweep_curve(chat_sweeps, "tpot_s_mean", scale=1000, ylabel="mean TPOT (ms)",
                title="Per-token latency", ax=axes[1][0])
    sweep_curve(chat_sweeps, "goodput_req_s", ylabel="goodput (req/s)",
                title="Goodput (TTFT<1s and p99 ITL<50ms)", ax=axes[1][1])
    plt.tight_layout(); save(fig, "07_sweeps"); plt.show()
'''),

    code('''
# Head-to-head at a single concurrency, with the table view beside it.
TARGET_CONCURRENCY = 16
at_c = [r for r in chat_sweeps if r.workload.concurrency == TARGET_CONCURRENCY]

if len(at_c) >= 2:
    from bench.plots import grouped_bar, color_for
    names = [label_for(r.engine.engine) for r in at_c]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for ax, (key, scale, title, unit) in zip(axes, [
        ("output_throughput_tok_s", 1, "Output throughput", "tok/s"),
        ("ttft_s_p99", 1000, "p99 TTFT", "ms"),
        ("tpot_s_mean", 1000, "mean TPOT", "ms"),
    ]):
        values = [r.metrics[key] * scale for r in at_c]
        ax.bar(range(len(at_c)), values, 0.62,
               color=[color_for(r.engine.engine) for r in at_c], linewidth=0)
        ax.set_xticks(range(len(at_c)))
        ax.set_xticklabels([n.replace(" ", "\\n") for n in names], fontsize=8)
        ax.set_ylabel(unit); ax.set_title(f"{title} @ concurrency {TARGET_CONCURRENCY}")
        for i, v in enumerate(values):
            ax.annotate(f"{v:,.0f}" if v >= 10 else f"{v:,.2f}", (i, v),
                        textcoords="offset points", xytext=(0, 3), ha="center",
                        fontsize=8, color="#52514e")
    plt.tight_layout(); save(fig, "07_head_to_head"); plt.show()

    table(at_c)
else:
    print(f"need >=2 engines at concurrency {TARGET_CONCURRENCY}")
'''),

    code('''
# Where the engines really differ: the shape of the tail, not the mean.
if len(at_c) >= 2:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))
    cdf(at_c, metric="ttft", title="TTFT distribution", ax=ax1, logx=True)
    cdf(at_c, metric="itl", title="Inter-token latency distribution", ax=ax2, logx=True)
    plt.tight_layout(); save(fig, "07_distributions"); plt.show()

    print("Two engines with the same mean can have completely different tails.")
    print("A long flat right edge on the ITL chart means stalls -- preemption,")
    print("a prefill blocking decodes, or scheduler starvation.")
'''),

    md("""
## Cost per million tokens

The number that actually decides deployments. It folds throughput and hardware
price into one figure:

```
$ / 1M output tokens  =  (GPU $/hour / 3600)  /  (output tokens/s)  x  1e6
```

Two caveats that make this an *estimate*, not an invoice: it assumes 100%
utilisation (real services run at 30-60%, so multiply accordingly), and the
hourly rates in `bench/sysinfo.py` are rough on-demand cloud prices you should
replace with what you actually pay.
"""),

    code('''
from bench.aggregate import to_row

rows = [to_row(r) for r in chat_sweeps if to_row(r)["usd_per_mtok"]]
if rows:
    df = pd.DataFrame(rows)
    best = (df.sort_values("usd_per_mtok")
              .groupby("engine", as_index=False).first()
              .sort_values("usd_per_mtok"))

    fig, ax = plt.subplots(figsize=(9, 4.4))
    from bench.plots import color_for
    ax.barh([label_for(e) for e in best.engine], best.usd_per_mtok, 0.6,
            color=[color_for(e) for e in best.engine], linewidth=0)
    ax.set_xlabel("USD per 1M output tokens (at 100% utilisation)")
    ax.set_title(f"Cost at each engine's best operating point -- {TARGET_GPU}")
    ax.grid(axis="x")
    for i, (v, c) in enumerate(zip(best.usd_per_mtok, best.concurrency)):
        ax.annotate(f"${v:.3f}  (c={c})", (v, i), textcoords="offset points",
                    xytext=(6, 0), va="center", fontsize=9, color="#52514e")
    plt.tight_layout(); save(fig, "07_cost"); plt.show()

    best[["engine", "concurrency", "output_throughput_tok_s", "usd_per_mtok"]]
else:
    print("no cost data -- set Hardware.usd_per_hour or add your GPU to GPU_HOURLY_USD")
'''),

    md(r"""
## The decision framework

Benchmarks go stale. The reasoning does not. Work down this tree.

```
START: what is the output?
|
+-- A label, a score, or a vector (fixed output set)
|     -> ENCODER (BERT/MiniLM/E5/BGE) on ONNX Runtime or TensorRT
|        behind Triton if you serve several models.           [notebooks 05, 06]
|        Optimise: length-sorted batching > truncation > batch size > int8.
|
+-- Open-ended text
      |
      +-- Do you have an NVIDIA GPU?
      |     |
      |     +-- NO (CPU / Mac / AMD / Intel)
      |     |     -> llama.cpp + GGUF. Q4_K_M unless it fits bigger. [notebook 03]
      |     |
      |     +-- YES
      |           |
      |           +-- Does the model fit in VRAM?
      |                 |
      |                 +-- NO -> llama.cpp with partial offload (-ngl),
      |                 |         or quantize until it fits, or a smaller model.
      |                 |         Check the offload curve first.      [notebook 03]
      |                 |
      |                 +-- YES
      |                       |
      |                       +-- One user / local / private?
      |                       |     -> llama.cpp (simplest) or vLLM (more features)
      |                       |
      |                       +-- Many concurrent users?
      |                             |
      |                             +-- Prompts share long prefixes, branch,
      |                             |   or are agent/tree shaped?
      |                             |     -> SGLang (RadixAttention)   [notebook 02]
      |                             |
      |                             +-- Latency is the product, model is stable,
      |                             |   GPUs are fixed, and you have CI?
      |                             |     -> TensorRT-LLM (compiled)   [notebook 04]
      |                             |
      |                             +-- Otherwise, or you value flexibility
      |                                 and model coverage?
      |                                   -> vLLM                      [notebook 01]
      |
      +-- Serving several models/frameworks on shared hardware?
            -> put any of the above behind Triton Inference Server.    [notebook 05]
```

### The summary table

| | vLLM | SGLang | llama.cpp | TensorRT-LLM | Triton Server |
|---|---|---|---|---|---|
| **Category** | engine | engine | engine | engine | serving layer |
| **Hardware** | NVIDIA (+ROCm) | NVIDIA (+ROCm) | **everything** | NVIDIA only | any backend |
| **Setup** | pip, minutes | pip, minutes | compile or binary | build engine, slow | container |
| **Batch-1 latency** | good | good | **very good** | **best** | n/a |
| **Throughput @ load** | **excellent** | **excellent** | fair | **excellent** | n/a |
| **Shared prefixes** | good (hash) | **best (radix tree)** | basic | good | n/a |
| **Model > VRAM** | no | no | **yes (offload)** | no | n/a |
| **Model coverage** | **widest** | wide | lagging | narrowest | n/a |
| **Multi-model** | no | no | no | no | **yes** |
| **Quantization** | AWQ/GPTQ/FP8 | AWQ/GPTQ/FP8 | **GGUF K/I-quants** | FP8/INT8/INT4 | backend's |
| **Ops complexity** | low | low | **lowest** | **highest** | high |

### Things that are true regardless of engine

1. **Your workload shape decides more than your engine choice.** The
   prefix-caching experiments in notebooks 01 and 02 moved TTFT by more than
   any engine-vs-engine difference at the same settings.
2. **Quantization is usually the biggest single lever**, and it is
   engine-independent. 4-bit weight-only typically gives more than switching
   engines, as long as you are decode-bound.
3. **Deploy at the goodput peak, not the throughput peak.** They are different
   concurrencies, and the gap is where "fast benchmark, unhappy users" lives.
4. **Measure your own prompt distribution.** Every number in this repo is
   specific to one model on one GPU at one set of lengths.
"""),

    md("""
## Honest limitations of everything above

Worth writing down, because a comparison that does not state its limits is
marketing.

1. **One small model.** Qwen2.5-0.5B is bandwidth-bound and *CPU-overhead
   sensitive*, which flatters engines with lean schedulers. At 7B or 70B each
   forward pass is long enough that scheduler differences nearly vanish and
   kernel quality dominates. Several rankings here plausibly invert at scale.
2. **One GPU, one session.** Colab hardware is shared and thermally variable.
   Differences under ~10% should be treated as noise unless you repeat the run.
3. **Short runs.** 64-96 requests per point gives a noisy p99. The CDFs are
   more trustworthy than the p99 numbers.
4. **Default-ish configurations.** Every engine here can be tuned further. A
   motivated expert on any one of them could likely beat these numbers.
5. **No multi-GPU, no tensor parallelism, no speculative decoding, no LoRA.**
   All of these change the picture, and some change it a lot.
6. **Versions move fast.** Each result records its engine version; check them
   before quoting anything.

If you want a number you can defend, re-run with your model, your prompts, your
hardware, several repeats, and report the spread rather than a single value.
"""),

    code('''
# Publish the aggregated results for the GitHub Pages site.
from bench.aggregate import build, markdown_table

payload = build()
print()
print(markdown_table([r for r in runs if r.engine.task == "generative"], limit=20))
'''),

    md("""
**Next:** [08 - Kernel profiling and optimization](08_kernel_optimization.ipynb),
where we stop choosing between engines and start making one faster.
"""),
]
