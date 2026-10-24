"""Notebook 05 - Triton Inference Server: the serving layer, not the engine."""

from tools.nbgen import code, md, setup_cells

NAME = "05_triton_inference_server.ipynb"
TITLE = "05 - Triton Inference Server: dynamic batching and the serving layer"

CELLS = setup_cells(
    NAME,
    TITLE,
    "Triton is the most commonly misunderstood item on this list, because it is not "
    "an inference engine at all -- it is the layer that runs other engines. We build "
    "its core mechanism, dynamic batching, from scratch and measure the trade-off it "
    "controls.",
    "~25 minutes",
    needs_gpu=False,
) + [

    md(r"""
## Triton is a different category of thing

vLLM, SGLang, llama.cpp and TensorRT-LLM are **engines**: they take a model and
compute tokens. Triton Inference Server is a **serving layer**: it takes
engines and gives them HTTP/gRPC endpoints, batching, scheduling, versioning,
metrics and multi-model hosting.

```
                      +-------------------------------+
   HTTP / gRPC  --->  |     Triton Inference Server   |
                      |                               |
                      |  scheduler + dynamic batcher  |
                      |  model repository & versions  |
                      |  metrics / health / tracing   |
                      +---------------+---------------+
                                      |
            +-------------+-----------+-----------+-------------+
            v             v                       v             v
      TensorRT       vLLM backend           ONNX Runtime     Python
      backend        (an LLM engine)        backend          backend
                                                             (your code)
```

So "vLLM vs Triton" is a category error. The real comparisons are:

- **vLLM's own server vs Triton + the vLLM backend** -- do you want vLLM's
  built-in HTTP server, or Triton fronting it?
- **TensorRT-LLM standalone vs Triton + TensorRT-LLM backend** -- the latter is
  NVIDIA's intended production path.

### What you get by putting Triton in front

1. **Many models, one server, one GPU.** A classifier, an embedder, a reranker
   and an LLM sharing hardware with per-model instance counts. No LLM server
   does this.
2. **Many frameworks, one API.** ONNX, TensorRT, PyTorch, TensorFlow, Python,
   and LLM backends behind an identical protocol.
3. **Ensembles and BLS.** Wire models into a pipeline server-side --
   tokenize -> embed -> rerank -> generate -- without a network hop per stage.
   Business Logic Scripting lets that pipeline contain real control flow.
4. **Production plumbing.** Model versioning with hot reload, Prometheus
   metrics, health endpoints, request tracing, dynamic model load/unload.
5. **Dynamic batching** for models that do not batch themselves -- which is the
   subject of the rest of this notebook.

### When you do *not* want it

One LLM, one model, one framework? Triton is extra moving parts for no gain.
Use the engine's own server. Triton earns its complexity when you have model
*variety*, not model *scale*.

> **Running it for real.** Triton ships as a container; it is not meaningfully
> installable on Colab. The command for a real machine:
>
> ```bash
> docker run --gpus all --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
>   -v $PWD/model_repository:/models \
>   nvcr.io/nvidia/tritonserver:25.01-py3 \
>   tritonserver --model-repository=/models
> ```
>
> Below we write a real model repository you can point that at, then build and
> measure Triton's central mechanism ourselves so the concept is not left
> abstract.
"""),

    md(r"""
## The model repository

Triton's configuration is a directory layout. This is the whole contract:

```
model_repository/
├── bert_classifier/
│   ├── config.pbtxt
│   ├── 1/                     <- version 1
│   │   └── model.onnx
│   └── 2/                     <- version 2, hot-swappable
│       └── model.onnx
└── text_pipeline/             <- an ensemble
    ├── config.pbtxt
    └── 1/                     <- empty; the ensemble is pure config
```

Numbered directories are versions. Dropping in a `3/` and touching the config
loads it without a restart, and the version policy decides whether to serve the
latest, all, or a specific set. That is a genuinely useful property that LLM
servers generally lack.
"""),

    code('''
from pathlib import Path

REPO = Path("model_repository/bert_classifier")
(REPO / "1").mkdir(parents=True, exist_ok=True)

config = """
name: "bert_classifier"
platform: "onnxruntime_onnx"
max_batch_size: 32

input [
  { name: "input_ids",      data_type: TYPE_INT64, dims: [ -1 ] },
  { name: "attention_mask", data_type: TYPE_INT64, dims: [ -1 ] }
]
output [
  { name: "logits", data_type: TYPE_FP32, dims: [ -1 ] }
]

# ---- the mechanism this notebook is about -------------------------------
# Wait up to max_queue_delay_microseconds for more requests to arrive so they
# can be run as one batch. Pure latency-for-throughput trade: every request
# pays up to this delay, and in exchange the GPU does far less work per
# sample. preferred_batch_size tells the scheduler which batch sizes are
# efficient -- for a TensorRT engine, the shapes it was tuned for.
dynamic_batching {
  preferred_batch_size: [ 8, 16, 32 ]
  max_queue_delay_microseconds: 5000
}

# Two copies of the model on GPU 0. Independent execution contexts, so one
# can run while the other is being fed -- this hides host-to-device copies
# and keeps the GPU from idling between batches.
instance_group [
  { count: 2, kind: KIND_GPU, gpus: [ 0 ] }
]

version_policy: { latest: { num_versions: 1 } }
"""
(REPO / "config.pbtxt").write_text(config.strip() + "\\n")
print((REPO / "config.pbtxt").read_text())
'''),

    md(r"""
## Dynamic batching, from first principles

Here is the whole idea. An encoder model (BERT) has no autoregressive loop:
one request, one forward pass. At batch 1 a GPU is grotesquely underused -- the
same reason decode is slow in notebook 00, for the same physical reason
(weights read once per forward pass regardless of batch).

If requests arrive at 200/s and a batch-1 forward pass takes 5 ms, you can
serve 200/s and the GPU is ~99% idle. If instead you **wait 5 ms**, collect
whatever arrived, and run them as one batch of ~10, the forward pass takes
maybe 6 ms and you have served 10 requests. Throughput goes up roughly 8x; each
request paid at most 5 ms extra.

```
NO BATCHING                         DYNAMIC BATCHING (5 ms window)
req arrives -> run -> respond       req arrives -> queue
                                    ... wait up to 5 ms ...
 r1  [==]                           r1  [----wait----]
 r2    [==]                         r2    [--wait--]  } one batch [====]
 r3      [==]                       r3      [-wait-]
 r4        [==]                     r4        [wait]
      4 forward passes                   1 forward pass
      GPU idle between each             GPU does 4x the useful work
```

**The knob is `max_queue_delay_microseconds`**, and the trade is strictly
monotonic in both directions:

- larger delay -> bigger batches -> higher throughput, higher p50 latency
- smaller delay -> smaller batches -> lower latency, lower throughput

The optimum depends on your arrival rate. At high load the queue fills before
the timeout expires, so the delay costs you nothing and is pure gain. At low
load you pay the full delay for a batch of two. **Dynamic batching helps most
exactly when you need it most**, which is a rare and pleasant property.

Let's build it and measure the curve.
"""),

    code('''
%pip install -q onnxruntime onnx transformers

import numpy as np, onnxruntime as ort, time, threading, queue
from pathlib import Path
from transformers import AutoTokenizer

BERT_ID = "distilbert-base-uncased-finetuned-sst-2-english"
ONNX_PATH = Path("bert.onnx")

if not ONNX_PATH.exists():
    import torch
    from transformers import AutoModelForSequenceClassification
    model = AutoModelForSequenceClassification.from_pretrained(BERT_ID).eval()
    dummy = {"input_ids": torch.ones(1, 128, dtype=torch.long),
             "attention_mask": torch.ones(1, 128, dtype=torch.long)}
    torch.onnx.export(
        model, (dummy["input_ids"], dummy["attention_mask"]), str(ONNX_PATH),
        input_names=["input_ids", "attention_mask"], output_names=["logits"],
        # Both dimensions dynamic: batch so we can batch, sequence so padding
        # is only as long as the longest member of each batch.
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"},
                      "attention_mask": {0: "batch", 1: "seq"},
                      "logits": {0: "batch"}},
        opset_version=17)
    print(f"exported -> {ONNX_PATH} ({ONNX_PATH.stat().st_size/1024**2:.0f} MB)")

providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
             if CAPS["has_gpu"] else ["CPUExecutionProvider"])
session = ort.InferenceSession(str(ONNX_PATH), providers=providers)
tokenizer = AutoTokenizer.from_pretrained(BERT_ID)
print("providers:", session.get_providers())
'''),

    code('''
SEQ_LEN = 128

def make_batch(n):
    texts = [f"This benchmark sample number {i} is reasonably positive overall."
             for i in range(n)]
    enc = tokenizer(texts, padding="max_length", truncation=True,
                    max_length=SEQ_LEN, return_tensors="np")
    return {"input_ids": enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64)}

def infer(batch):
    return session.run(["logits"], batch)[0]

# Warm up: first calls pay for provider init, graph optimisation and
# cuDNN/cuBLAS autotuning -- exactly as in notebook 00.
for _ in range(5):
    infer(make_batch(8))

print(f"{'batch':>6} {'latency':>10} {'per-sample':>12} {'throughput':>13}")
print("-" * 46)
batch_latency = {}
for bs in (1, 2, 4, 8, 16, 32, 64):
    b = make_batch(bs)
    times = []
    for _ in range(20):
        t0 = time.perf_counter(); infer(b); times.append(time.perf_counter() - t0)
    med = sorted(times)[len(times)//2]
    batch_latency[bs] = med
    print(f"{bs:>6} {1000*med:>9.2f}m {1000*med/bs:>11.3f}m {bs/med:>12.0f}/s")

print("\\nNote how latency barely grows from batch 1 to batch 8: that spare")
print("capacity is exactly what dynamic batching harvests.")
'''),

    md("""
### Now the batcher itself

About 30 lines. Requests land on a queue; a batcher thread drains it, stopping
when it has `max_batch_size` items **or** when `max_queue_delay` has elapsed
since the first item arrived. That timeout is the entire mechanism.
"""),

    code('''
import dataclasses

@dataclasses.dataclass
class Req:
    arrive_s: float
    done: threading.Event
    latency_s: float = 0.0
    batch_size: int = 0


class DynamicBatcher:
    """A faithful miniature of Triton's dynamic batching scheduler."""

    def __init__(self, max_batch_size=32, max_queue_delay_us=5000):
        self.q: "queue.Queue[Req]" = queue.Queue()
        self.max_batch_size = max_batch_size
        self.max_delay = max_queue_delay_us / 1e6
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while not self.stop.is_set():
            try:
                first = self.q.get(timeout=0.05)
            except queue.Empty:
                continue

            batch, deadline = [first], time.perf_counter() + self.max_delay
            # Keep collecting until the batch is full or the window closes.
            while len(batch) < self.max_batch_size:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    batch.append(self.q.get(timeout=remaining))
                except queue.Empty:
                    break

            infer(make_batch(len(batch)))
            finished = time.perf_counter()
            for r in batch:
                r.latency_s = finished - r.arrive_s
                r.batch_size = len(batch)
                r.done.set()

    def submit(self) -> Req:
        r = Req(arrive_s=time.perf_counter(), done=threading.Event())
        self.q.put(r)
        return r

    def shutdown(self):
        self.stop.set(); self.thread.join(timeout=5)
'''),

    code('''
from bench.metrics import percentile
from bench.workloads import poisson_arrivals

def run_load(batcher, rate_per_s, n=400, seed=0):
    """Open-loop Poisson arrivals against the batcher."""
    offsets = poisson_arrivals(n, rate_per_s, seed=seed)
    reqs, t0 = [], time.perf_counter()
    for off in offsets:
        delay = off - (time.perf_counter() - t0)
        if delay > 0:
            time.sleep(delay)
        reqs.append(batcher.submit())
    for r in reqs:
        r.done.wait(timeout=60)
    duration = time.perf_counter() - t0
    lat = [r.latency_s for r in reqs]
    return {
        "p50_ms": 1000 * percentile(lat, 50),
        "p99_ms": 1000 * percentile(lat, 99),
        "throughput": len(reqs) / duration,
        "mean_batch": sum(r.batch_size for r in reqs) / len(reqs),
    }


RATE = 300 if CAPS["has_gpu"] else 60
DELAYS_US = [0, 500, 1000, 2000, 5000, 10000, 20000]

rows = []
for delay in DELAYS_US:
    b = DynamicBatcher(max_batch_size=32, max_queue_delay_us=delay)
    stats = run_load(b, RATE, n=400)
    b.shutdown()
    rows.append({"delay_us": delay, **stats})
    print(f"  delay {delay:>6} us -> batch {stats['mean_batch']:>5.1f}  "
          f"p50 {stats['p50_ms']:>7.1f} ms  p99 {stats['p99_ms']:>7.1f} ms  "
          f"{stats['throughput']:>6.0f} req/s")
'''),

    code('''
import matplotlib.pyplot as plt
import pandas as pd
from bench.plots import INK, PALETTE, save, use_theme
use_theme()

df = pd.DataFrame(rows)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))

# The trade-off as a Pareto curve: both measures keep their own axis.
ax1.plot(df.throughput, df.p50_ms, marker="o", color=PALETTE["magenta"],
         markeredgecolor=INK["surface"], markeredgewidth=1.5, label="p50")
ax1.plot(df.throughput, df.p99_ms, marker="s", color=PALETTE["orange"],
         markeredgecolor=INK["surface"], markeredgewidth=1.5, label="p99")
for _, r in df.iterrows():
    ax1.annotate(f"{int(r.delay_us)}us", (r.throughput, r.p50_ms),
                 textcoords="offset points", xytext=(6, 4), fontsize=8,
                 color=INK["muted"])
ax1.set_xlabel("throughput (req/s)"); ax1.set_ylabel("latency (ms)")
ax1.set_title("Queue delay buys throughput with latency")
ax1.legend(); ax1.grid(axis="both")

ax2.plot(df.delay_us, df.mean_batch, marker="^", color=PALETTE["aqua"],
         markeredgecolor=INK["surface"], markeredgewidth=1.5)
ax2.set_xlabel("max_queue_delay (us)"); ax2.set_ylabel("mean batch size")
ax2.set_title(f"Batch formation at {RATE} req/s offered")
ax2.grid(axis="both")

plt.tight_layout(); save(fig, "05_dynamic_batching"); plt.show()

base, best = df.iloc[0], df.loc[df.throughput.idxmax()]
print(f"no batching:  {base.throughput:>6.0f} req/s at p50 {base.p50_ms:.1f} ms")
print(f"best delay:   {best.throughput:>6.0f} req/s at p50 {best.p50_ms:.1f} ms "
      f"({int(best.delay_us)} us, mean batch {best.mean_batch:.1f})")
print(f"\\nthroughput gain: {best.throughput/base.throughput:.1f}x "
      f"for {best.p50_ms - base.p50_ms:+.1f} ms of p50 latency")
'''),

    md("""
### Reading the curve

The left chart is the whole decision. Pick the point where the curve bends:
past it you are paying a lot of latency for very little throughput.

Two things worth noticing:

**The delay is an upper bound, not a cost you always pay.** At high arrival
rates the batch fills before the timeout, so the real added latency is much
less than `max_queue_delay`. Look at the mean batch size: once it saturates at
`max_batch_size`, increasing the delay does nothing at all.

**p99 degrades faster than p50.** The unlucky request is the one that arrives
just after a batch closes and waits the full window plus the forward pass. If
you have a tail-latency SLO, set the delay from the p99 curve, not the p50.

### What this teaches about LLM serving

Dynamic batching is for models with a **single forward pass per request** --
encoders, classifiers, embedders, rerankers. LLMs need something strictly more
general, because requests have wildly different numbers of forward passes and
arrive and leave mid-batch. That generalisation is **continuous batching** from
notebook 01: instead of forming a batch and running it to completion, the
scheduler rebuilds the batch every single decode step.

So the progression is:

```
no batching  ->  dynamic batching  ->  continuous batching
one at a time    group arrivals,       rebuild the batch every
                 run to completion     forward pass; members join
                                       and leave mid-flight
```

Triton's LLM backends (vLLM, TensorRT-LLM) therefore do **not** use the dynamic
batcher -- they do their own in-flight batching, and Triton's config for them
disables dynamic batching accordingly. Setting `dynamic_batching` on an LLM
backend is a classic misconfiguration that adds latency and buys nothing.
"""),

    code('''
# A real config for an LLM backend, where dynamic batching is explicitly OFF.
VLLM_REPO = Path("model_repository/qwen_vllm")
(VLLM_REPO / "1").mkdir(parents=True, exist_ok=True)

(VLLM_REPO / "1" / "model.json").write_text("""{
  "model": "Qwen/Qwen2.5-0.5B-Instruct",
  "disable_log_requests": true,
  "gpu_memory_utilization": 0.85,
  "max_model_len": 4096,
  "enable_prefix_caching": true
}
""")

(VLLM_REPO / "config.pbtxt").write_text("""
name: "qwen_vllm"
backend: "vllm"

# NOTE: no dynamic_batching block. The vLLM backend does its own continuous
# batching per decode step; Triton's dynamic batcher would only add queue
# delay in front of a scheduler that is already better at this job.
instance_group [ { count: 1, kind: KIND_MODEL } ]
""".strip() + "\\n")

print((VLLM_REPO / "config.pbtxt").read_text())
print("model repository now contains:")
for p in sorted(Path("model_repository").rglob("*")):
    print("  " + str(p))
'''),

    md("""
## `perf_analyzer`, and why its numbers differ from ours

Triton ships `perf_analyzer`, its own load generator:

```bash
perf_analyzer -m bert_classifier \\
  --concurrency-range 1:64:8 \\
  --percentile=99 \\
  --input-data random \\
  --measurement-interval 10000
```

It is excellent and you should use it when you run Triton for real. Be aware
it defaults to **closed-loop** concurrency (notebook 01's Experiment 4) and
reports its own latency breakdown -- queue time, compute time, and
input/output transfer separately. That breakdown is genuinely useful: if queue
time dominates, add instances or raise the batch size; if compute dominates,
you need a faster engine or a smaller model.

Our harness reports end-to-end client-side latency instead, which is what a
user experiences and what makes engines comparable across this repo. Neither is
wrong; they answer different questions. Expect `perf_analyzer`'s numbers to be
slightly lower than ours because it excludes some client-side cost.

## When to choose Triton

**Strong fit**
- Multiple models, multiple frameworks, one serving tier.
- Non-LLM models that need batching they cannot do themselves -- embedders,
  rerankers, classifiers, vision models.
- You need ensembles/BLS to keep a pipeline server-side.
- You want NVIDIA's production path for TensorRT-LLM, with metrics,
  versioning and k8s integration already solved.

**Weak fit**
- One LLM, one framework. Use that engine's own server; Triton adds operational
  surface for no benefit.
- No Kubernetes and no appetite for containers. Triton is container-first.
- You need an OpenAI-compatible API out of the box. Triton's native protocol is
  its own (KServe v2); there is an OpenAI-compatible frontend for LLM backends,
  but it is an extra piece rather than the default.

### Exercises

1. Re-run the delay sweep at a much lower arrival rate (say 20 req/s). The
   optimal delay should move -- and dynamic batching should look much less
   attractive. Explain why in terms of queue occupancy.
2. Add a second "instance" by running two batcher threads over the same
   session. Does throughput improve? What does that tell you about whether
   ONNX Runtime releases the GIL during inference?
3. Modify `make_batch` so sequence lengths vary (32-128 tokens) and pad to the
   batch max rather than to 128. Measure the waste. Then sort the queue by
   length before batching and measure again -- this is exactly what a
   production embedding server does.
4. Write the `config.pbtxt` for an ensemble that runs a tokenizer in the Python
   backend and feeds `bert_classifier`.

**Next:** [06 - Encoder models](06_encoder_models_bert.ipynb), where we take the
BERT side seriously.
"""),
]
