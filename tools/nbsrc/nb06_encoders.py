"""Notebook 06 - Encoder models: BERT-family serving, and when not to use an LLM."""

from tools.nbgen import code, md, setup_cells

NAME = "06_encoder_models_bert.ipynb"
TITLE = "06 - Encoders (BERT) vs decoders (Qwen): picking the right model"

CELLS = setup_cells(
    NAME,
    TITLE,
    "Half of real-world inference is not generation at all -- it is classification, "
    "embedding and reranking. The cost model is completely different, the serving "
    "stack is completely different, and using a 0.5B decoder where a 22M encoder "
    "would do is the most expensive mistake on this list.",
    "~25 minutes",
) + [

    md(r"""
## The architectural fork

```
ENCODER (BERT, RoBERTa, DeBERTa, E5, BGE)
  input:  [CLS] the movie was great [SEP]
                    |
          bidirectional self-attention     every token sees every other token,
          N layers                         in both directions
                    |
  output: one vector per token  ->  pool  ->  ONE answer
  ONE forward pass. Done.

DECODER (Qwen, Llama, Mistral, GPT)
  input:  the movie was
                    |
          causal self-attention            token i sees only tokens <= i
          N layers                         (so the KV cache is reusable)
                    |
  output: next-token distribution -> sample -> append -> REPEAT
  ONE forward pass PER OUTPUT TOKEN.
```

Everything that makes LLM serving hard comes from that loop:

| | Encoder | Decoder |
|---|---|---|
| forward passes per request | **1** | 1 + output_len |
| KV cache | **none needed** | the dominant memory cost |
| attention | bidirectional | causal, incremental |
| TTFT / TPOT / ITL | **meaningless** | the headline metrics |
| what to optimise | batch size, padding | KV cache, scheduling |
| right engine | ONNX Runtime, TensorRT, Triton | vLLM, SGLang, TRT-LLM |

An encoder has no KV cache because it never extends a sequence -- there is
nothing to cache *for*. That single fact removes PagedAttention, continuous
batching, prefix caching and preemption from the problem. Encoder serving is a
much simpler engineering problem, and that is why the tooling looks so
different.

### The metrics change

TTFT, TPOT and ITL simply do not exist here -- there is no stream, no first
token, no gap between tokens. What matters instead:

- **latency at batch 1** (the interactive case),
- **throughput vs batch size** (the bulk case),
- **latency vs sequence length** (attention is still O(N²)),
- **padding waste** (the thing people forget, and often the biggest win).

So results in this notebook use `task="encode"` in the schema, and notebook 07
plots them on their own axes rather than pretending they are comparable to
generation numbers.
"""),

    md("""
## Which model, for which job

The question "Qwen base or BERT based?" has a clean answer once you ask what
the output is.

| Job | Output | Right family | Typical size |
|---|---|---|---|
| Sentiment / intent / topic | a label | **encoder** (BERT, DeBERTa-v3) | 22M-300M |
| Named entities, token tagging | per-token labels | **encoder** | 110M-300M |
| Semantic search, RAG retrieval | a vector | **embedding encoder** (E5, BGE, GTE) | 33M-560M |
| Reranking a candidate list | a score | **cross-encoder** | 22M-500M |
| Summarise, answer, write code | free text | **decoder** (Qwen, Llama) | 0.5B-70B+ |
| Extract structured JSON | structured text | **decoder + grammar** | 0.5B-7B |
| Any task with no training data | anything | **decoder, zero-shot** | 1B-70B |

The honest trade:

- **An encoder is 10-100x cheaper** for a fixed label set. A 22M MiniLM runs
  thousands of classifications per second on a CPU; a 0.5B decoder doing the
  same job costs a GPU and still produces tokens you have to parse.
- **A decoder needs no training data.** That is often worth more than the
  compute, especially early, and it is why "just prompt an LLM" is a reasonable
  first move.
- **The usual mature answer is both**: prototype with a decoder, label a few
  thousand examples with it, distil into an encoder, keep the decoder for the
  long tail. You get the LLM's coverage at the encoder's cost.

A useful rule of thumb: if the output is drawn from a **fixed, known set**, an
encoder is probably right. If the output is **open-ended**, you need a decoder.
"""),

    code('''
%pip install -q onnxruntime onnx transformers sentence-transformers

import time, numpy as np, torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from bench.plots import INK, PALETTE, save, use_theme
import matplotlib.pyplot as plt

use_theme()
device = "cuda" if torch.cuda.is_available() else "cpu"

MODELS = {
    "MiniLM-L6 (22M)":    "sentence-transformers/all-MiniLM-L6-v2",
    "DistilBERT (66M)":   "distilbert-base-uncased",
    "BERT-base (110M)":   "bert-base-uncased",
}

for label, mid in MODELS.items():
    cfg = __import__("transformers").AutoConfig.from_pretrained(mid)
    n = cfg.num_hidden_layers * (12 * cfg.hidden_size**2) + cfg.vocab_size * cfg.hidden_size
    print(f"  {label:<20} layers={cfg.num_hidden_layers:<3} hidden={cfg.hidden_size:<5} "
          f"~{n/1e6:.0f}M params")
'''),

    md(r"""
## The cost model for an encoder

Simpler than the decoder case, and worth deriving once.

```
FLOPs per sequence  ~=  2 * params * seq_len          (the dense part)
                      + 4 * layers * seq_len^2 * hidden   (attention)

Bytes moved per BATCH ~= params * dtype_bytes          (weights, read once)
                       + activations (proportional to batch * seq_len)
```

The crucial asymmetry: **weight bytes are per batch, FLOPs are per sequence.**
So arithmetic intensity grows nearly linearly with batch size, exactly as in
decode -- which means a batch-1 encoder call is just as wasteful as batch-1
decoding, and for the same reason.

The quadratic attention term is negligible at seq_len 128 but dominates by
seq_len 512+. That is why a 512-token BERT call is far more than 4x the cost of
a 128-token one, and why truncating inputs is such a cheap win.
"""),

    code('''
def bench_backend(run_fn, batch_sizes, seq_len=128, repeats=15, warmup=5):
    """Median latency per batch size. Median, not mean: a single scheduler
    hiccup or a GC pause skews a mean badly at these timescales."""
    out = {}
    for bs in batch_sizes:
        for _ in range(warmup):
            run_fn(bs, seq_len)
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            run_fn(bs, seq_len)
            times.append(time.perf_counter() - t0)
        out[bs] = sorted(times)[len(times) // 2]
    return out


MODEL_ID = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
torch_model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).to(device).eval()

def make_inputs(bs, seq_len):
    texts = [f"Benchmark input number {i} for encoder serving." for i in range(bs)]
    return tokenizer(texts, padding="max_length", truncation=True,
                     max_length=seq_len, return_tensors="pt")

@torch.no_grad()
def run_torch(bs, seq_len):
    enc = {k: v.to(device) for k, v in make_inputs(bs, seq_len).items()}
    out = torch_model(**enc)
    if device == "cuda":
        torch.cuda.synchronize()
    return out

BATCHES = [1, 2, 4, 8, 16, 32, 64]
results = {"PyTorch eager": bench_backend(run_torch, BATCHES)}
print("PyTorch eager:", {k: f"{1000*v:.2f}ms" for k, v in results["PyTorch eager"].items()})
'''),

    code('''
# ---- ONNX Runtime ---------------------------------------------------------
# The standard production path for encoders. It does graph-level optimisation
# (constant folding, operator fusion, layout changes) ahead of time, then runs
# with far less per-op overhead than eager PyTorch.
import onnxruntime as ort
from pathlib import Path

ONNX_PATH = Path("distilbert.onnx")
if not ONNX_PATH.exists():
    dummy = make_inputs(1, 128)
    torch.onnx.export(
        torch_model.cpu(), (dummy["input_ids"], dummy["attention_mask"]), str(ONNX_PATH),
        input_names=["input_ids", "attention_mask"], output_names=["logits"],
        dynamic_axes={"input_ids": {0: "b", 1: "s"}, "attention_mask": {0: "b", 1: "s"},
                      "logits": {0: "b"}},
        opset_version=17)
    torch_model.to(device)

providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
             if device == "cuda" else ["CPUExecutionProvider"])
sess = ort.InferenceSession(str(ONNX_PATH), providers=providers)

def run_onnx(bs, seq_len):
    enc = make_inputs(bs, seq_len)
    return sess.run(["logits"], {"input_ids": enc["input_ids"].numpy(),
                                 "attention_mask": enc["attention_mask"].numpy()})

results["ONNX Runtime"] = bench_backend(run_onnx, BATCHES)
print("ONNX Runtime:", {k: f"{1000*v:.2f}ms" for k, v in results["ONNX Runtime"].items()})
'''),

    code('''
# ---- torch.compile --------------------------------------------------------
# Traces the model and generates fused Triton kernels. The first call pays a
# large compilation cost, and it recompiles for every new input SHAPE -- which
# is precisely why dynamic batch sizes and encoders are an awkward pair.
# mode="reduce-overhead" additionally uses CUDA graphs.
try:
    compiled = torch.compile(torch_model, mode="reduce-overhead", dynamic=False)

    @torch.no_grad()
    def run_compiled(bs, seq_len):
        enc = {k: v.to(device) for k, v in make_inputs(bs, seq_len).items()}
        out = compiled(**enc)
        if device == "cuda":
            torch.cuda.synchronize()
        return out

    print("compiling (one build per batch size -- this is the cost)...")
    t0 = time.perf_counter()
    for bs in BATCHES:
        run_compiled(bs, 128)
    print(f"total compile time for {len(BATCHES)} shapes: {time.perf_counter()-t0:.1f}s")

    results["torch.compile"] = bench_backend(run_compiled, BATCHES, warmup=3)
    print("torch.compile:", {k: f"{1000*v:.2f}ms" for k, v in results["torch.compile"].items()})
except Exception as exc:
    print(f"torch.compile unavailable here: {exc}")
'''),

    code('''
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))
colors = {"PyTorch eager": INK["muted"], "ONNX Runtime": PALETTE["green"],
          "torch.compile": PALETTE["violet"]}
markers = {"PyTorch eager": ".", "ONNX Runtime": "P", "torch.compile": "X"}

for name, data in results.items():
    bs = sorted(data)
    ax1.plot(bs, [1000 * data[b] for b in bs], marker=markers[name], color=colors[name],
             markeredgecolor=INK["surface"], markeredgewidth=1.5, label=name)
    ax2.plot(bs, [b / data[b] for b in bs], marker=markers[name], color=colors[name],
             markeredgecolor=INK["surface"], markeredgewidth=1.5, label=name)

for ax in (ax1, ax2):
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(lambda v, _: f"{int(v)}")
    ax.set_xlabel("batch size"); ax.grid(axis="both"); ax.legend()
ax1.set_ylabel("latency (ms)"); ax1.set_title("Latency grows slowly with batch")
ax2.set_yscale("log"); ax2.set_ylabel("sequences/s")
ax2.set_title("...so throughput scales almost linearly")

plt.tight_layout(); save(fig, "06_encoder_backends"); plt.show()

base = results["PyTorch eager"]
print(f"{'batch':>6} " + " ".join(f"{n:>16}" for n in results))
print("-" * (7 + 17 * len(results)))
for b in BATCHES:
    row = f"{b:>6} "
    for name, data in results.items():
        row += f" {base[b]/data[b]:>15.2f}x"
    print(row)
print("\\n(speedup vs PyTorch eager; the batch-1 column is where graph")
print("optimisation matters most, because per-op overhead dominates there)")
'''),

    md(r"""
## The padding tax

This is the optimisation people miss, and it is frequently larger than
switching backends.

A batch must be rectangular, so every sequence is padded to the longest member.
Real text has a long-tailed length distribution, so a batch of mostly-short
texts with one long one wastes most of its compute:

```
batch sorted by arrival                batch sorted by length
  "ok"                [##..............]   "ok"            [##]
  "this is longer"    [#####...........]   "hi there"      [##]
  "hi there"          [##..............]   "this is..."    [#####]
  "a very long one"   [################]   "a very long"   [################]
   padded to 16 -> 75% waste              two small batches -> ~20% waste
```

**Length-sorted batching** (sometimes "bucketing") groups similar lengths
together. In an offline/bulk job you can sort the whole dataset and the win is
nearly free. In an online server you sort within the queue window -- which is
exactly what the dynamic batcher in notebook 05 should be doing.

Let's measure the tax on a realistic length distribution.
"""),

    code('''
rng = np.random.default_rng(0)
# Log-normal lengths: the shape real text actually has -- many short, a few long.
lengths = np.clip(rng.lognormal(mean=3.4, sigma=0.8, size=512).astype(int), 8, 512)
print(f"lengths: median {int(np.median(lengths))}, mean {lengths.mean():.0f}, "
      f"p99 {int(np.percentile(lengths, 99))}, max {lengths.max()}")

def padded_tokens(batch_lengths):
    """Total token slots processed = batch_size * longest member."""
    return len(batch_lengths) * max(batch_lengths)

BS = 32
naive = sum(padded_tokens(lengths[i:i+BS]) for i in range(0, len(lengths), BS))
sorted_lengths = np.sort(lengths)
bucketed = sum(padded_tokens(sorted_lengths[i:i+BS]) for i in range(0, len(lengths), BS))
useful = int(lengths.sum())

print(f"\\n  useful tokens          {useful:>8,}")
print(f"  arrival-order batching {naive:>8,}  ({naive/useful:.2f}x, "
      f"{100*(1-useful/naive):.0f}% wasted)")
print(f"  length-sorted batching {bucketed:>8,}  ({bucketed/useful:.2f}x, "
      f"{100*(1-useful/bucketed):.0f}% wasted)")
print(f"\\n  speedup from sorting alone: {naive/bucketed:.2f}x -- no model change,")
print("  no new backend, no quantization. Just ordering the queue.")

fig, ax = plt.subplots(figsize=(8.5, 4.2))
from bench.plots import grouped_bar
grouped_bar(["useful", "arrival order", "length sorted"],
            {"token slots processed": [useful, naive, bucketed]},
            ylabel="token slots", title=f"The padding tax at batch size {BS}",
            colors={"token slots processed": PALETTE["green"]}, ax=ax)
save(fig, "06_padding_tax"); plt.show()
'''),

    md("""
## Sequence length: where the quadratic term bites
"""),

    code('''
seq_results = {}
for seq_len in (64, 128, 256, 512):
    d = bench_backend(run_onnx, [8], seq_len=seq_len, repeats=10)
    seq_results[seq_len] = d[8]

fig, ax = plt.subplots(figsize=(8.5, 4.4))
lens = sorted(seq_results)
measured = [1000 * seq_results[s] for s in lens]
linear = [measured[0] * s / lens[0] for s in lens]
quad = [measured[0] * (s / lens[0]) ** 2 for s in lens]

ax.plot(lens, measured, marker="o", color=PALETTE["green"],
        markeredgecolor=INK["surface"], markeredgewidth=1.5, label="measured")
ax.plot(lens, linear, color=INK["muted"], linewidth=1.3, label="linear O(N)")
ax.plot(lens, quad, color=INK["axis"], linewidth=1.3, label="quadratic O(N^2)")
ax.set_xlabel("sequence length (tokens)"); ax.set_ylabel("latency at batch 8 (ms)")
ax.set_title("Encoder cost sits between linear and quadratic")
ax.legend(); ax.grid(axis="both")
save(fig, "06_seqlen_scaling"); plt.show()

print(f"{'seq_len':>8} {'latency':>10} {'vs 64':>8}")
for s in lens:
    print(f"{s:>8} {1000*seq_results[s]:>9.2f}m {seq_results[s]/seq_results[64]:>7.2f}x")
print("\\nAt short lengths the dense (linear) term dominates; by 512 the")
print("quadratic attention term is clearly bending the curve upward. This is")
print("why truncating inputs to what you actually need is such a cheap win.")
'''),

    code('''
# Save encoder results in the shared schema so notebook 07 can place them on
# their own axes. task="encode" marks them as non-generative: TTFT and TPOT
# are left unset rather than filled with meaningless zeros.
from bench import EngineSpec, Hardware, RequestRecord, RunResult, WorkloadSpec
from bench.metrics import summarize
from bench import package_versions

backend_versions = package_versions("onnxruntime", "torch")

for name, data in results.items():
    engine = {"ONNX Runtime": "onnxruntime", "torch.compile": "torch_compile"}.get(name, "hf")
    for bs, latency in data.items():
        records = [RequestRecord(start_offset_s=0.0, ttft_s=None, e2e_s=latency,
                                 prompt_tokens=128 * bs, output_tokens=None,
                                 token_count_method="exact")
                   for _ in range(bs)]
        metrics = {
            "n_requests": bs, "n_success": bs, "n_failed": 0,
            "duration_s": latency,
            "e2e_s_mean": latency, "e2e_s_p50": latency, "e2e_s_p99": latency,
            "request_throughput_req_s": bs / latency,
            "total_input_tokens": 128 * bs, "total_output_tokens": 0,
            "output_throughput_tok_s": 0.0,
            "total_throughput_tok_s": 128 * bs / latency,
            "latency_per_sample_ms": 1000 * latency / bs,
        }
        RunResult(
            engine=EngineSpec(engine=engine, version=backend_versions.get("onnxruntime", "-"),
                              model_id=MODEL_ID, task="encode", model_family="bert",
                              params_b=0.066, dtype="float32",
                              flags={"backend": name, "batch_size": bs, "seq_len": 128}),
            workload=WorkloadSpec(name="classify", mode="closed", concurrency=bs,
                                  n_requests=bs, input_len=128, output_len=0,
                                  ignore_eos=False),
            hardware=HW, metrics=metrics, requests=[], duration_s=latency,
            notes=f"encoder classification, {name}, batch {bs}",
        ).save("results/real")

print("saved encoder results")
'''),

    md("""
## Decision guide

**Use an encoder when** the output comes from a fixed set -- labels, scores,
vectors. Serve it with ONNX Runtime (CPU or GPU) or TensorRT, behind Triton if
you have several models. Optimise in this order:

1. **Sort by length before batching.** Usually the biggest single win, and it
   costs nothing.
2. **Truncate inputs** to the length you actually need. The quadratic term
   punishes generosity.
3. **Batch as hard as your latency budget allows** -- see notebook 05's queue
   delay curve.
4. **Export to ONNX**, then consider int8 dynamic quantization (encoders
   tolerate it well; 2-3x on CPU is typical).
5. **Then** consider a smaller model. MiniLM-L6 is 5x smaller than BERT-base
   and within a point or two on many tasks.

**Use a decoder when** the output is open-ended, or when you have no labelled
data and need zero-shot coverage. Serve it with vLLM or SGLang, and constrain
the output with a grammar if you need structure (notebook 02).

**A note on embeddings.** vLLM and SGLang can both serve embedding models now,
and if you are already running one of them for generation, reusing it avoids a
second serving tier -- a real operational benefit. But for a dedicated
embedding service at scale, ONNX Runtime or TensorRT is leaner: no KV cache
machinery, no scheduler, lower memory, lower latency.

### Exercises

1. Export DistilBERT to int8 with `onnxruntime.quantization.quantize_dynamic`
   and compare latency and accuracy on a real dataset. Encoders usually survive
   int8 with almost no loss -- confirm it rather than trusting it.
2. Implement length-sorted batching inside notebook 05's `DynamicBatcher` and
   measure the combined win against the baseline here.
3. Time Qwen2.5-0.5B doing the same sentiment classification by generating
   "positive"/"negative", and compare cost per 1000 classifications against
   DistilBERT. Include the GPU you need for each.
4. Compare MiniLM-L6, DistilBERT and BERT-base on the same task: latency,
   throughput, and accuracy. Plot accuracy against latency and find the knee.

**Next:** [07 - The comparison](07_comparison.ipynb), where everything measured
so far goes on one set of axes.
"""),
]
