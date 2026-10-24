"""Notebook 01 - vLLM: continuous batching and PagedAttention, measured."""

from tools.nbgen import code, md, setup_cells

NAME = "01_vllm_deep_dive.ipynb"
TITLE = "01 - vLLM: continuous batching, PagedAttention, prefix caching"

CELLS = setup_cells(
    NAME,
    TITLE,
    "vLLM is the reference implementation of modern LLM serving, and the ideas it "
    "popularised -- paged KV cache, continuous batching, prefix caching -- are now in "
    "every other engine here. We measure each one by turning it off.",
    "~35 minutes",
) + [

    md(r"""
## What vLLM actually does

Notebook 00 ended with a painful number: a hand-written decode loop reaching a
fraction of the hardware ceiling at batch 1. vLLM closes that gap with four
mechanisms. Each is measurable, and we will measure each one.

### 1. Continuous batching (the big one)

Static batching waits for every sequence in a batch to finish before starting
the next batch. Since output lengths vary wildly, the batch runs at the speed
of its slowest member and the GPU idles on finished slots:

```
STATIC BATCHING                      CONTINUOUS BATCHING
step ->                              step ->
A  ####----------                    A  ####BBBB*EEEE          * = new request
B  ########------                    B  ########FFFF             joins mid-flight
C  ##############                    C  ##############
D  ###-----------                    D  ###GGGGGGGGGG
   ^^^^ idle slots burning           every slot stays busy
        memory bandwidth
```

Finished sequences leave at the *token* boundary and waiting requests take
their slot immediately. On mixed-length traffic this alone is typically a
**2-4x throughput win**, with no change to the model or the math.

### 2. PagedAttention

The 16-token-block KV cache from notebook 00. It removes the fragmentation that
otherwise wastes most of your cache, and it makes blocks *shareable* between
sequences -- which enables mechanism 3.

### 3. Automatic prefix caching (APC)

If two requests start with the same tokens, the second one can point at the
first one's already-computed KV blocks instead of recomputing them. vLLM hashes
each block of tokens and keeps a lookup table. On a RAG or system-prompt
workload where 90% of the prompt is shared, this cuts prefill work by ~90% --
and prefill is what TTFT is made of.

### 4. CUDA graphs + chunked prefill

**CUDA graphs** record the whole decode step as a single replayable graph,
collapsing hundreds of individual kernel launches (5-10 us each) into one
submission. At small batch sizes, launch overhead is a large share of a decode
step, so this matters most exactly where decode is weakest.

**Chunked prefill** splits a long prefill into pieces and interleaves them with
decode steps. Without it, one 8k-token prompt monopolises a forward pass and
every other user's stream visibly stalls. It trades a little TTFT on the long
request for a lot of ITL stability on everyone else.

> **V1 engine.** Modern vLLM runs a rewritten core with a cleaner scheduler,
> prefix caching on by default, and chunked prefill on by default. Several
> older tutorials show flags that no longer exist or no longer default the
> same way -- which is exactly why this repo records engine versions and flags
> into every result file.
"""),

    code('''
# vLLM pulls a large dependency tree and a matching torch build. Expect 3-6
# minutes on a cold Colab runtime.
%pip install -q vllm

from bench import package_versions
VERSIONS = package_versions("vllm", "torch", "transformers", "flashinfer-python")
for name, v in VERSIONS.items():
    print(f"  {name:<22} {v}")
'''),

    code('''
import asyncio
from bench import (EndpointConfig, EngineSpec, PromptBuilder, describe,
                   free_gpu, kill_stale, preset, run, serve, sweep_concurrency,
                   vllm_cmd)
from bench.plots import save, use_theme
from transformers import AutoTokenizer
import matplotlib.pyplot as plt

use_theme()
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
PORT = 8000

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
builder = PromptBuilder(tokenizer, seed=0)

# A T4 has 16 GB and no bf16. Keep max_model_len modest so the KV cache can
# hold enough concurrent sequences to make the sweep interesting -- see the
# concurrency table you printed in notebook 00.
MAX_LEN = 4096
GPU_UTIL = 0.85 if (HW.gpu_memory_gb or 16) <= 16 else 0.90
DTYPE = "float16" if not CAPS["bf16"] else "auto"

print(f"max_model_len={MAX_LEN}  gpu_memory_utilization={GPU_UTIL}  dtype={DTYPE}")

kill_stale()   # clear anything left over from an earlier cell
free_gpu()
'''),

    md("""
### Launching the server

We drive vLLM over its OpenAI-compatible HTTP endpoint rather than the offline
`LLM()` class. The offline API skips tokenisation, scheduling and queueing, so
it reports numbers 20-40% better than anything you can deploy. Since the whole
point here is comparing engines fairly, every engine gets measured the same
way: real server, real HTTP, real streaming.

The server log goes to a file instead of the notebook -- vLLM emits thousands
of lines and would make the saved `.ipynb` unusable. Use `print_log(PORT)` if a
launch fails.
"""),

    code('''
cmd = vllm_cmd(
    MODEL_ID, port=PORT,
    max_model_len=MAX_LEN,
    gpu_memory_utilization=GPU_UTIL,
    dtype=DTYPE,
    enable_prefix_caching=True,
    max_num_seqs=256,
)
print(" ".join(cmd))
'''),

    code('''
from bench import discover_model_id, probe
from bench.servers import print_log

# Everything that needs the server runs inside this block. On exit the process
# is killed and the ~14 GB of preallocated KV cache is returned to the GPU.
server_ctx = serve(cmd, port=PORT, log_name="vllm.log")
base_url = server_ctx.__enter__()

try:
    served_model = run(discover_model_id(base_url))
    cfg = EndpointConfig(base_url=base_url, model=served_model, engine="vllm")
    print("serving:", served_model)
    print(run(probe(cfg, "The capital of France is", max_tokens=8)))
except Exception:
    print_log(PORT, 40)
    raise
'''),

    md("""
## Experiment 1 - The concurrency sweep

The curve that defines an engine. One line per concurrency level, same
workload, same model. Watch for three things:

- **where throughput stops rising** -- that is real capacity, and it is usually
  far below the "max tokens/s" number in any announcement post;
- **how steeply p99 TTFT climbs afterwards** -- that is the scheduler under
  pressure;
- **the ratio between concurrency 1 and the knee** -- that is everything
  continuous batching bought you over notebook 00's baseline.
"""),

    code('''
engine_spec = EngineSpec(
    engine="vllm", version=VERSIONS["vllm"], model_id=MODEL_ID,
    model_family="qwen", params_b=0.49, dtype=DTYPE, max_model_len=MAX_LEN,
    flags={"enable_prefix_caching": True, "max_num_seqs": 256,
           "gpu_memory_utilization": GPU_UTIL, "api": "openai-server",
           "chunked_prefill": "default"},
)

chat = preset("chat", input_len=256, output_len=256, n_requests=64)
LEVELS = (1, 2, 4, 8, 16, 32, 64)

sweep = run(sweep_concurrency(cfg, engine_spec, chat, builder, levels=LEVELS,
                              hardware=HW, notes="vLLM baseline concurrency sweep"))
for r in sweep:
    r.save("results/real")
print(f"\\nsaved {len(sweep)} runs")
'''),

    code('''
from bench.plots import pareto, sweep_curve, table

fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))

sweep_curve(sweep, "output_throughput_tok_s", ylabel="output tok/s",
            title="Throughput saturates", ax=axes[0][0])
sweep_curve(sweep, "ttft_s_p99", scale=1000, ylabel="p99 TTFT (ms)",
            title="Tail latency is the price", ax=axes[0][1])
sweep_curve(sweep, "tpot_s_mean", scale=1000, ylabel="mean TPOT (ms)",
            title="Per-token speed degrades gracefully", ax=axes[1][0])
sweep_curve(sweep, "goodput_req_s", ylabel="goodput (req/s)",
            title="Goodput peaks, then collapses", ax=axes[1][1])

plt.tight_layout(); save(fig, "01_vllm_sweep"); plt.show()
table(sweep)
'''),

    code('''
best_tp  = max(sweep, key=lambda r: r.metrics["output_throughput_tok_s"])
best_gp  = max(sweep, key=lambda r: r.metrics["goodput_req_s"])
single   = min(sweep, key=lambda r: r.workload.concurrency)

print(f"peak throughput at concurrency {best_tp.workload.concurrency}: "
      f"{best_tp.metrics['output_throughput_tok_s']:.0f} tok/s")
print(f"peak goodput    at concurrency {best_gp.workload.concurrency}: "
      f"{best_gp.metrics['goodput_req_s']:.2f} req/s")
print(f"batch-1 throughput: {single.metrics['output_throughput_tok_s']:.0f} tok/s")
print(f"\\ncontinuous batching gain: "
      f"{best_tp.metrics['output_throughput_tok_s']/single.metrics['output_throughput_tok_s']:.1f}x")
print()
print("If peak goodput happens at a LOWER concurrency than peak throughput,")
print("you have just measured the gap between 'fast' and 'usable'. Deploy at")
print("the goodput peak, not the throughput peak.")
'''),

    md(r"""
## Experiment 2 - Prefix caching, measured properly

The `rag` workload is 2048 tokens of prompt, of which 1792 (87.5%) are a shared
prefix -- a system prompt plus retrieved documents reused across requests. This
is extremely common in production and it is where prefix caching earns its
keep.

**The controlled comparison matters.** We restart the server with prefix
caching off and run the *identical* workload. Comparing two engines' defaults
on this workload would just compare "cache on" to "cache off" -- that is the
mistake that produces a lot of misleading engine-vs-engine charts.

Prediction: TTFT should fall by roughly the shared fraction (~87%), because
prefill is what TTFT is made of. TPOT should be *unchanged* -- caching does
nothing for decode, which reads weights, not prefixes. If your TPOT moves much,
something else changed and the experiment is not controlled.
"""),

    code('''
rag = preset("rag", n_requests=64, concurrency=8)
print(describe(rag))

prompts = builder.build(rag)
mean_len, min_len, max_len = builder.verify(prompts, rag)
print(f"actual prompt tokens: mean {mean_len:.0f} (min {min_len}, max {max_len})")
print(f"shared prefix is {rag.shared_prefix_tokens} tokens = "
      f"{100*rag.shared_prefix_tokens/mean_len:.0f}% of each prompt")

# Sanity check that the prefix really is byte-identical -- a single differing
# character at the front defeats the cache completely.
assert len({p[:200] for p in prompts}) == 1, "prefix differs between prompts!"
print("verified: all prompts share a byte-identical prefix")
'''),

    code('''
from bench import benchmark

spec_apc_on = EngineSpec(**{**engine_spec.__dict__,
                            "flags": {**engine_spec.flags, "enable_prefix_caching": True}})
apc_on = run(benchmark(cfg, spec_apc_on, rag, prompts, hardware=HW,
                       notes="RAG workload, prefix caching ON"))
apc_on.save("results/real")
'''),

    code('''
# Restart with prefix caching OFF. Same model, same workload, one flag.
server_ctx.__exit__(None, None, None)
free_gpu()

cmd_off = vllm_cmd(MODEL_ID, port=PORT, max_model_len=MAX_LEN,
                   gpu_memory_utilization=GPU_UTIL, dtype=DTYPE,
                   enable_prefix_caching=False, max_num_seqs=256)

server_ctx = serve(cmd_off, port=PORT, log_name="vllm-no-apc.log")
base_url = server_ctx.__enter__()
cfg = EndpointConfig(base_url=base_url, model=run(discover_model_id(base_url)), engine="vllm")

spec_apc_off = EngineSpec(**{**engine_spec.__dict__,
                             "flags": {**engine_spec.flags, "enable_prefix_caching": False}})
apc_off = run(benchmark(cfg, spec_apc_off, rag, prompts, hardware=HW,
                        notes="RAG workload, prefix caching OFF"))
apc_off.save("results/real")
'''),

    code('''
from bench.plots import cdf, grouped_bar, PALETTE

on, off = apc_on.metrics, apc_off.metrics
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6))

grouped_bar(
    ["TTFT p50 (ms)", "TTFT p99 (ms)", "TPOT mean (ms)"],
    {"prefix caching OFF": [1000*off["ttft_s_p50"], 1000*off["ttft_s_p99"],
                            1000*off["tpot_s_mean"]],
     "prefix caching ON":  [1000*on["ttft_s_p50"],  1000*on["ttft_s_p99"],
                            1000*on["tpot_s_mean"]]},
    ylabel="milliseconds", title="Prefix caching on a 87%-shared RAG prompt",
    colors={"prefix caching OFF": "#898781", "prefix caching ON": PALETTE["blue"]},
    ax=ax1,
)
cdf([apc_off, apc_on], metric="ttft", title="TTFT distribution", ax=ax2)
ax2.get_legend().set_visible(False)
ax2.text(0.98, 0.1, "left = better", transform=ax2.transAxes, ha="right",
         fontsize=9, color="#898781")

plt.tight_layout(); save(fig, "01_prefix_caching"); plt.show()

print(f"TTFT p50:  {1000*off['ttft_s_p50']:7.1f} ms -> {1000*on['ttft_s_p50']:7.1f} ms "
      f"({off['ttft_s_p50']/on['ttft_s_p50']:.2f}x faster)")
print(f"TPOT mean: {1000*off['tpot_s_mean']:7.2f} ms -> {1000*on['tpot_s_mean']:7.2f} ms "
      f"(should be ~unchanged: caching does nothing for decode)")
print(f"throughput: {off['output_throughput_tok_s']:.0f} -> "
      f"{on['output_throughput_tok_s']:.0f} tok/s")
'''),

    md("""
**Reading the result.** The first request in the ON run is a cache *miss* and
pays full price; everything after it hits. That is why the TTFT CDF has a
distinctive shape -- one slow request at the top, a tight cluster below. In
production that cold-start cost is amortised over far more requests, so the
steady-state win is larger than this 64-request run suggests.

**Where prefix caching does nothing:** a workload with no shared prefix. Run
the `chat` preset both ways and the difference disappears into the noise.
Caching is a workload-shaped optimisation, not a free speedup, and this is the
single biggest reason benchmark results disagree across blog posts.
"""),

    md("""
## Experiment 3 - `max_num_seqs`: the latency/throughput dial

`max_num_seqs` caps how many sequences share one decode step. It is the most
direct control you have over where you sit on the trade-off curve:

- **low** (8-16): small batches, fast per-token decode, excellent tail latency,
  poor GPU utilisation.
- **high** (256+): big batches, maximum throughput, worse ITL, and once the KV
  cache fills, vLLM starts **preempting** -- evicting a sequence's blocks and
  recomputing them later. Preemption shows up as vicious ITL spikes.

Watch the p99 ITL column especially. That is where preemption hides.
"""),

    code('''
results_seqs = []
for max_seqs in (8, 32, 256):
    server_ctx.__exit__(None, None, None); free_gpu()
    c = vllm_cmd(MODEL_ID, port=PORT, max_model_len=MAX_LEN,
                 gpu_memory_utilization=GPU_UTIL, dtype=DTYPE,
                 enable_prefix_caching=True, max_num_seqs=max_seqs)
    server_ctx = serve(c, port=PORT, log_name=f"vllm-seqs{max_seqs}.log")
    base_url = server_ctx.__enter__()
    cfg = EndpointConfig(base_url=base_url, model=run(discover_model_id(base_url)),
                         engine="vllm")

    spec = EngineSpec(**{**engine_spec.__dict__,
                         "flags": {**engine_spec.flags, "max_num_seqs": max_seqs}})
    w = preset("chat", input_len=512, output_len=256, n_requests=96, concurrency=64)
    res = run(benchmark(cfg, spec, w, builder.build(w), hardware=HW,
                        notes=f"max_num_seqs={max_seqs} at concurrency 64"))
    res.save("results/real")
    results_seqs.append((max_seqs, res))
'''),

    code('''
labels = [f"max_num_seqs={n}" for n, _ in results_seqs]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.4))

grouped_bar(labels,
            {"output tok/s": [r.metrics["output_throughput_tok_s"] for _, r in results_seqs]},
            ylabel="output tok/s", title="Throughput", ax=ax1)
grouped_bar(labels,
            {"p99 ITL (ms)": [1000*r.metrics["itl_s_p99"] for _, r in results_seqs],
             "mean ITL (ms)": [1000*r.metrics["itl_s_mean"] for _, r in results_seqs]},
            ylabel="milliseconds", title="Inter-token latency: mean vs tail", ax=ax2)

plt.tight_layout(); save(fig, "01_max_num_seqs"); plt.show()

print(f"{'config':<22} {'tok/s':>9} {'mean ITL':>10} {'p99 ITL':>10} {'ratio':>7}")
print("-" * 62)
for n, r in results_seqs:
    m = r.metrics
    print(f"max_num_seqs={n:<9} {m['output_throughput_tok_s']:>9.0f} "
          f"{1000*m['itl_s_mean']:>9.2f}m {1000*m['itl_s_p99']:>9.2f}m "
          f"{m['itl_s_p99']/m['itl_s_mean']:>7.1f}x")
print("\\nA p99/mean ratio above ~3x means some requests are being starved or")
print("preempted. Check the server log for 'preempt' to confirm.")
'''),

    md("""
## Experiment 4 - Open loop: finding the cliff

Every experiment so far was closed-loop, where the client waits for a response
before sending the next request. That hides overload: when the server slows
down, the client automatically slows down too.

Open loop sends requests on a Poisson process at a fixed rate regardless of
what the server is doing. Push the rate past capacity and the queue grows
without bound -- latency diverges. That divergence point is the only honest
answer to "how many requests per second can this handle?"

Use the peak-goodput concurrency from Experiment 1 to pick rates around the
capacity you measured.
"""),

    code('''
from dataclasses import replace

capacity = best_gp.metrics["request_throughput_req_s"]
rates = [round(capacity * f, 2) for f in (0.4, 0.7, 0.9, 1.1, 1.4)]
print(f"measured closed-loop capacity ~{capacity:.2f} req/s; probing {rates}")

open_results = []
for rate in rates:
    w = replace(preset("chat", input_len=256, output_len=128),
                mode="open", request_rate=rate, n_requests=max(60, int(rate * 45)))
    spec = EngineSpec(**{**engine_spec.__dict__,
                         "flags": {**engine_spec.flags, "load": "open-loop poisson"}})
    res = run(benchmark(cfg, spec, w, builder.build(w), hardware=HW,
                        notes=f"open loop at {rate} req/s", progress=False))
    res.save("results/real")
    open_results.append(res)
    m = res.metrics
    print(f"  offered {rate:>5.2f} req/s -> achieved {m['request_throughput_req_s']:>5.2f}, "
          f"p99 TTFT {1000*m['ttft_s_p99']:>8.0f} ms")
'''),

    code('''
offered  = [r.workload.request_rate for r in open_results]
achieved = [r.metrics["request_throughput_req_s"] for r in open_results]
p99      = [1000*r.metrics["ttft_s_p99"] for r in open_results]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.4))
from bench.plots import INK

ax1.plot(offered, achieved, marker="o", color=PALETTE["blue"],
         markeredgecolor=INK["surface"], markeredgewidth=1.5, label="achieved")
ax1.plot(offered, offered, color=INK["muted"], linewidth=1.2, label="offered (y=x)")
ax1.set_xlabel("offered rate (req/s)"); ax1.set_ylabel("achieved (req/s)")
ax1.set_title("Where the server stops keeping up"); ax1.legend(); ax1.grid(axis="both")

ax2.plot(offered, p99, marker="o", color=PALETTE["orange"],
         markeredgecolor=INK["surface"], markeredgewidth=1.5)
ax2.set_yscale("log")
ax2.set_xlabel("offered rate (req/s)"); ax2.set_ylabel("p99 TTFT (ms)")
ax2.set_title("Latency diverges past capacity"); ax2.grid(axis="both")

plt.tight_layout(); save(fig, "01_open_loop"); plt.show()

print("The knee in the right-hand chart is your real capacity. Publishing the")
print("throughput number from past the knee is how a system ships 'validated'")
print("at a rate it cannot actually serve.")
'''),

    code('''
# Always release the GPU before moving on. A leaked vLLM process holds ~85% of
# VRAM and makes the next notebook fail with a baffling OOM.
server_ctx.__exit__(None, None, None)
kill_stale()
free_gpu()
'''),

    md("""
## When to choose vLLM

**Strong fit**
- Open-weight decoder models served over an OpenAI-compatible API.
- You want excellent throughput without compiling anything or tuning much.
- Broad model coverage matters -- vLLM usually supports a new architecture
  within days, and that is a real operational advantage.
- You need LoRA adapters, structured output, multi-modal, or speculative
  decoding without assembling them yourself.

**Weaker fit**
- Absolute lowest latency at batch 1 on NVIDIA hardware. A compiled
  TensorRT-LLM engine wins that (notebook 04), at the cost of a build step and
  far less flexibility.
- Heavy shared-prefix / tree-structured workloads such as agent loops, where
  SGLang's RadixAttention is a better-matched design (notebook 02).
- No GPU, or a model larger than your VRAM. That is llama.cpp's territory
  (notebook 03).

**The flags that actually matter**, in order of impact:
`max_model_len` (KV budget per sequence) > `gpu_memory_utilization` (total KV
pool) > `max_num_seqs` (latency/throughput dial) > `enable_prefix_caching`
(workload-dependent, enormous when it fits) > `max_num_batched_tokens`
(chunked-prefill granularity, controls ITL stability under mixed load).

### Exercises

1. Re-run Experiment 2 with the `chat` preset (no shared prefix). Confirm the
   prefix-caching win vanishes, then explain in one sentence why.
2. Set `max_model_len=32768` and repeat the concurrency sweep. Throughput
   should collapse at high concurrency. Use the KV-cache table from notebook 00
   to predict the concurrency at which it happens *before* you run it.
3. Grep the server log for `Preemption`. Find the `max_num_seqs` value where
   preemption starts, and connect it to the p99 ITL spike in Experiment 3.
4. Add `--max-num-batched-tokens 512` and run a mixed workload (some 4k
   prompts, some short). Compare p99 ITL against the default. This is chunked
   prefill protecting the decoders from the prefillers.

**Next:** [02 - SGLang](02_sglang_deep_dive.ipynb), which takes prefix sharing
much further.
"""),
]
