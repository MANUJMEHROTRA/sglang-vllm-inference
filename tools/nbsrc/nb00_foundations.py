"""Notebook 00 - Foundations: the mental model and the measuring rig."""

from tools.nbgen import code, md, setup_cells

NAME = "00_foundations_and_setup.ipynb"
TITLE = "00 - Foundations: what you are actually measuring"

CELLS = setup_cells(
    NAME,
    TITLE,
    "Before comparing engines, build the model of *why* inference costs what it "
    "costs. By the end you will be able to predict, on paper, roughly how fast a "
    "model can possibly decode on a given GPU -- and then measure how close a real "
    "stack gets. Every later notebook is a variation on that one move.",
    "~15 minutes",
) + [

    md(r"""
## Part 1 - A model answers in two completely different phases

This is the single most important idea in the whole repo. Everything else --
PagedAttention, continuous batching, chunked prefill, RadixAttention,
quantization choice -- is a response to the gap between these two phases.

```
PREFILL                                      DECODE
"read the prompt"                            "write the answer"

one forward pass over ALL N prompt tokens    one forward pass PER output token

 t0  t1  t2  t3  ...  tN                      tN+1     tN+2     tN+3
  |   |   |   |        |                        |        |        |
  +---+---+---+--------+                        |        |        |
       all at once                             one      one      one
                                               at a     at a     at a
  * every token attends to every other         time     time     time
    token -> O(N^2) attention work
  * the GPU gets a big fat matrix multiply    * batch dimension is 1 token
  * COMPUTE-bound: tensor cores are busy      * MEMORY-bound: reads all the
  * fills the KV cache                          weights to produce 1 token
                                              * appends 1 entry to the KV cache
  => determines TTFT                          => determines TPOT / ITL
```

**Why decode is so much worse than it looks.** To produce one token for one
sequence, the GPU must read *every weight in the model* from HBM. A 0.5B model
in fp16 is ~1 GB, so one token costs ~1 GB of memory traffic -- and about
1 GFLOP of arithmetic. That ratio, **~1 FLOP per byte**, is the number to
remember. A T4 needs about **200 FLOPs per byte** to keep its tensor cores
busy. So at batch size 1, decode uses well **under 1% of the GPU's math
capability**. The rest sits idle waiting for memory.

**This is why batching works.** The weights get read once per *step*, not once
per *sequence*. Serve 32 sequences in one step and you amortise that 1 GB read
across 32 tokens. The arithmetic intensity rises ~32x and throughput rises
almost as much -- which is the entire economic foundation of vLLM, SGLang,
TensorRT-LLM and every other serving engine here.

And it is why **prefill and decode want opposite things**: prefill wants to be
left alone with the whole GPU; decode wants as many friends in the batch as
possible. Scheduling that conflict is what separates a good engine from a bad
one.
"""),

    md("""
### The model we will use throughout

`Qwen/Qwen2.5-0.5B-Instruct` is the workhorse of this repo:

- **Small enough** to load in seconds and fit many copies on a free T4, so you
  can run a full sweep in a Colab session rather than watching a progress bar.
- **Modern architecture** -- grouped-query attention, SwiGLU, RMSNorm, RoPE,
  tied embeddings. Everything you learn transfers directly to Llama-3,
  Mistral, Gemma and the larger Qwens, which differ only in dimensions.
- **Supported by every engine here**, including TensorRT-LLM and llama.cpp
  (with GGUF conversions on the Hub).

Notebook 07 repeats the key experiments on `Qwen2.5-1.5B-Instruct` and
`Qwen2.5-7B-Instruct` (A100 only) so you can see how the conclusions scale.
Scaling is not a detail: several rankings here *invert* with model size.
"""),

    code('''
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

%pip install -q transformers accelerate

from bench.metrics import ModelShape

shape = ModelShape.from_hf_config(MODEL_ID)
params = shape.params()

print(f"{MODEL_ID}")
print(f"  layers           {shape.n_layers}")
print(f"  hidden size      {shape.hidden_size}")
print(f"  query heads      {shape.n_heads}")
print(f"  KV heads         {shape.n_kv_heads}   <- grouped-query attention")
print(f"  head dim         {shape.head_dim}")
print(f"  FFN size         {shape.intermediate_size}")
print(f"  vocab            {shape.vocab_size:,}")
print(f"  tied embeddings  {shape.tie_word_embeddings}")
print()
print(f"  parameters       {params/1e9:.3f} B")
print(f"  weights @ fp16   {params*2/1024**3:.2f} GB")
print(f"  weights @ int4   {params*0.5/1024**3:.2f} GB   (plus scales, ~10% more)")
print()
print(f"  GQA ratio        {shape.n_heads // shape.n_kv_heads}x  "
      f"-> KV cache is {shape.n_heads // shape.n_kv_heads}x smaller than with plain MHA")
'''),

    md(r"""
## Part 2 - The roofline: the speed limit you cannot argue with

A kernel is limited by one of two things: how fast the GPU can do math
(FLOP/s), or how fast it can move bytes (GB/s). The **roofline** model says:

```
achievable FLOP/s = min( peak_FLOPs ,  arithmetic_intensity x peak_bandwidth )
                          \____/                               \________/
                        compute roof                          memory roof
```

**Arithmetic intensity** = FLOPs performed / bytes moved. The **ridge point**
is where the two roofs meet (peak FLOPs / peak bandwidth). Left of the ridge
you are memory-bound; right of it, compute-bound.

Ridge points worth committing to memory:

| GPU | HBM bandwidth | fp16 dense | ridge point |
|---|---:|---:|---:|
| T4 (sm_75) | 320 GB/s | 65 TFLOP/s | ~203 FLOP/byte |
| L4 (sm_89) | 300 GB/s | 121 TFLOP/s | ~403 FLOP/byte |
| A100-40GB (sm_80) | 1555 GB/s | 312 TFLOP/s | ~201 FLOP/byte |
| H100 SXM (sm_90) | 3350 GB/s | 989 TFLOP/s | ~295 FLOP/byte |

Notice the ridge barely moves across four generations while absolute
performance grows ~15x. **Memory bandwidth has not kept up with arithmetic**,
and that divergence is why inference engineering is almost entirely an
exercise in moving fewer bytes.

The cell below places prefill and decode on that roofline. The distance between
them is the gap that every engine in this repo is trying to close.
"""),

    code('''
import matplotlib.pyplot as plt
from bench.metrics import arithmetic_intensity, roofline_decode_tok_s
from bench.plots import roofline, save

peak_tflops = SPECS.get("fp16_tflops", 65.0)
bandwidth   = SPECS.get("hbm_gb_s", 320.0)
gpu_label   = HW.gpu_name or "T4 (assumed - no GPU detected)"

points = []

# Decode at several batch sizes. Weights are read once per step no matter the
# batch, so intensity scales ~linearly with batch size.
for bs in (1, 8, 32, 128):
    flops = 2 * shape.params() * bs
    bytes_moved = shape.params() * 2 + shape.kv_cache_bytes(1024, batch=bs)
    ai = arithmetic_intensity(flops, bytes_moved)
    achieved = min(peak_tflops, ai * bandwidth * 1e9 / 1e12)
    points.append((f"decode bs={bs}", ai, achieved))

# Prefill of a 2048-token prompt: one huge matmul, far to the right.
pre_flops = shape.prefill_flops(2048)
pre_bytes = shape.params() * 2 + shape.kv_cache_bytes(2048)
pre_ai = arithmetic_intensity(pre_flops, pre_bytes)
points.append(("prefill 2048 tok", pre_ai, min(peak_tflops, pre_ai * bandwidth * 1e9 / 1e12)))

fig, ax = plt.subplots(figsize=(9, 5.2))
roofline(peak_tflops, bandwidth, points, title=f"Roofline - {gpu_label}", ax=ax)
save(fig, "00_roofline")
plt.show()

print(f"\\n{'regime':<20} {'FLOP/byte':>10} {'% of peak FLOPS':>18}")
print("-" * 50)
for label, ai, achieved in points:
    print(f"{label:<20} {ai:>10.1f} {100*achieved/peak_tflops:>17.1f}%")
'''),

    md("""
**Read the chart like this.** Every decode point sits on the *sloped* part of
the roof -- the memory roof. That means its speed is set purely by bandwidth,
and the only ways to go faster are to move fewer bytes:

- **quantize the weights** (int4 moves a quarter of the bytes -> up to ~4x
  faster decode; this is why llama.cpp on CPU is viable at all),
- **batch harder** (amortise the same bytes over more tokens),
- **shrink the KV cache** (GQA, MQA, quantized KV, shorter context),
- **fuse kernels** so intermediate tensors never round-trip to HBM (notebook 08).

Making the math faster -- better tensor-core utilisation, FP8 arithmetic -- does
*nothing* for a point on the sloped roof. This is the single most useful
diagnostic in the repo: **find out which roof you are under before optimising
anything.**

Prefill sits far to the right, near or past the ridge. It is the one part of
inference where raw FLOP/s matters, which is exactly why TensorRT-LLM's
aggressively tuned GEMMs show their biggest advantage on long prompts.
"""),

    code('''
# Paper prediction vs. the hardware's hard ceiling.
print(f"Decode ceiling on {gpu_label} ({bandwidth:.0f} GB/s), fp16 weights:\\n")
print(f"{'batch':>6} {'ceiling tok/s':>15} {'per-seq tok/s':>15}")
print("-" * 40)
for bs in (1, 4, 16, 64, 256):
    total = roofline_decode_tok_s(shape, bandwidth, batch=bs, seq_len=1024)
    print(f"{bs:>6} {total:>15,.0f} {total/bs:>15,.1f}")

print("\\nNote how per-sequence speed *falls* as batch grows while total rises.")
print("That trade -- individual latency for aggregate throughput -- is the whole")
print("game. An engine's scheduler is the thing choosing where you sit on it.")
'''),

    md(r"""
## Part 3 - The KV cache is your real memory budget

The weights are a fixed cost. The **KV cache** is the variable one, and it is
what actually decides how many users you can serve at once.

For every token in every sequence, attention must keep a key and a value vector
per layer:

```
KV bytes = 2 (K and V) x layers x kv_heads x head_dim x tokens x dtype_bytes
```

Two consequences people get wrong constantly:

1. **It scales with total tokens across all sequences**, not with sequence
   count. 100 users at 500 tokens costs the same as 10 users at 5000.
2. **`max_model_len` is a memory setting, not a politeness setting.** Engines
   reserve cache per sequence based on it. Leaving a 0.5B model at its full 32k
   context on a 16 GB T4 is the most common reason "only a few requests fit".

The cell below converts your actual GPU memory into a concurrency budget.
"""),

    code('''
import math

weights_gb = shape.params() * 2 / 1024**3
total_gb   = HW.gpu_memory_gb or SPECS.get("mem_gb", 16.0)
overhead_gb = 1.2            # CUDA context, activations, fragmentation, cuBLAS workspace
kv_budget_gb = max(0.0, total_gb * 0.90 - weights_gb - overhead_gb)

print(f"GPU total            {total_gb:>7.1f} GB")
print(f"  weights (fp16)     {weights_gb:>7.2f} GB")
print(f"  runtime overhead   {overhead_gb:>7.2f} GB  (approx)")
print(f"  usable for KV      {kv_budget_gb:>7.2f} GB  (at gpu_memory_utilization=0.90)")
print()
print(f"{'context':>9} {'KV / sequence':>15} {'concurrent sequences':>22}")
print("-" * 50)
for ctx in (512, 1024, 2048, 4096, 8192, 16384, 32768):
    per_seq = shape.kv_cache_bytes(ctx)
    fits = int(kv_budget_gb * 1024**3 / per_seq) if per_seq else 0
    print(f"{ctx:>9} {per_seq/1024**2:>12.1f} MB {fits:>22,}")

print("\\nSame table with int8 KV cache (halves every number above):")
for ctx in (4096, 32768):
    per_seq = shape.kv_cache_bytes(ctx, dtype_bytes=1)
    fits = int(kv_budget_gb * 1024**3 / per_seq) if per_seq else 0
    print(f"{ctx:>9} {per_seq/1024**2:>12.1f} MB {fits:>22,}")
'''),

    md(r"""
### Why PagedAttention exists

Naively, you reserve `max_model_len` tokens of cache per sequence up front. If
`max_model_len` is 4096 and the average conversation is 300 tokens, you waste
**over 90% of your KV memory** on space that will never be written -- and that
waste translates directly into 10x fewer concurrent users.

vLLM's PagedAttention fixes this by borrowing virtual memory's oldest trick:
split the cache into fixed-size **blocks** (16 tokens each) and keep a per-
sequence **block table** mapping logical positions to physical blocks.

```
Without paging -- contiguous reservation per sequence
  seq A  [####........................]  4096 reserved, 400 used
  seq B  [##..........................]  4096 reserved, 200 used
         \_____________ 90%+ wasted ____________/

With paging -- 16-token blocks, allocated on demand
  physical blocks  [b0][b1][b2][b3][b4][b5][b6][b7] ...
  seq A table      b0 -> b3 -> b7
  seq B table      b1 -> b2
  free list        b4, b5, b6, ...       <- immediately reusable
```

The payoffs compound:

- **near-zero internal fragmentation** (worst case: 15 wasted tokens per
  sequence instead of thousands),
- **copy-on-write sharing**: two sequences with the same prefix point at the
  *same* physical blocks, which is the mechanism behind prefix caching
  (notebook 01) and SGLang's RadixAttention (notebook 02),
- **preemption becomes possible**: under pressure the scheduler can evict a
  sequence's blocks and recompute later, instead of OOMing.

The cost is an indirection on every attention read, which needs a custom kernel
-- you cannot express a block table in stock PyTorch attention. That kernel is
one of the things we profile in notebook 08.
"""),

    md(r"""
## Part 4 - The metrics, defined precisely

```
 client sends                 first token                         last token
      |                            |                                   |
      v                            v                                   v
      |========= TTFT ============>|--+--+--+--+--+--+--+--+--+--+--+-->|
      |   queue wait + prefill     |  |  |                              |
      |                            |  |  +-- ITL: one sample per gap    |
      |                            |  +----- ITL                        |
      |                            +-------- ITL                        |
      |                                                                 |
      |<======================= E2E latency ===========================>|

      TPOT = (E2E - TTFT) / (output_tokens - 1)       == mean ITL
```

| Metric | What it is | What it feels like | Driven by |
|---|---|---|---|
| **TTFT** | send -> first token | "did it hang?" | prompt length, queue depth, prefill throughput |
| **ITL** | gap between tokens | "is it stuttering?" | batch size, scheduler, preemption |
| **TPOT** | mean ITL | reading speed | memory bandwidth, batch size |
| **E2E** | send -> last token | total wait | all of the above + output length |
| **Output throughput** | generated tok/s | $ per token | batch size, quantization |
| **Goodput** | req/s meeting an SLO | *actual* capacity | everything |

**Human reference points.** Comfortable reading is ~5-10 tokens/s, so a TPOT
under ~100 ms already outruns the reader. Below ~30 ms there is no perceptible
benefit for a chat UI -- but a *single* 500 ms stall is immediately visible.
This is why the **p99 of ITL matters far more than the mean**, and why a
benchmark that reports only averages tells you almost nothing about how a
system feels.

**Goodput is the honest headline.** An engine can post a magnificent throughput
number by batching 512 requests and making every one of them wait four seconds
for its first token. Throughput says "excellent"; every user says "broken".
Goodput -- requests/s that met *both* a TTFT and an ITL bound -- is the metric
that cannot be gamed this way. The harness keeps raw per-request records
precisely so you can re-evaluate goodput under a different SLO without re-running
anything.
"""),

    md("""
## Part 5 - Six ways an inference benchmark lies

Every one of these has produced a widely-shared, completely wrong blog post.
`bench.runner.validate()` checks for numbers 2, 3 and 5 automatically after
every run in this repo.

**1. No warmup.** The first request pays for CUDA context creation, cuBLAS
autotuning, Triton JIT, memory-pool growth and often CUDA graph capture. On a
T4 it can be 10x the steady-state latency. In a 32-request run that single
request moves the mean by a third. *Fix: discard warmup requests.*

**2. The client is the bottleneck.** One Python event loop parsing SSE tops out
around a few thousand chunks/s. You measure your own client and attribute it to
the server. *Fix: apply Little's Law -- throughput x latency should equal the
concurrency you actually requested. If it doesn't, you are measuring yourself.*

**3. Different output lengths.** Engine A stops at 40 tokens, engine B rambles
to 200. You just measured the sampler's verbosity, not the engine. *Fix:
`ignore_eos=True` and a fixed `max_tokens` everywhere.*

**4. "Tokens/s" without saying which.** Counting prompt tokens as throughput
makes a summarisation workload look 10x faster than a chat workload on
identical hardware. *Fix: report output throughput and total throughput
separately -- this harness always reports both.*

**5. Closed-loop only.** Workers that wait for a response before sending the
next one automatically back off when the server slows down, so the server can
never fall behind and you never find the cliff. *Fix: also run open-loop
Poisson arrivals, where overload is visible.*

**6. Comparing default configurations.** SGLang enables its radix cache by
default; vLLM's prefix caching has been opt-in in some versions. Benchmark the
defaults on a shared-prefix workload and you are comparing "cache on" to
"cache off" and calling it an architecture comparison. *Fix: control the flags
explicitly, and record them in the result -- which the schema forces.*

> A seventh, subtler one: **tokenizer differences**. "512 tokens" means
> different text to different tokenizers. We sidestep it by generating prompts
> with the target model's own tokenizer and verifying the actual lengths
> before every run.
"""),

    md("""
## Part 6 - The baseline: plain HuggingFace `transformers`

Before touching a serving engine, measure the naive thing. This is the number
every engine's marketing is implicitly quoting against, and understanding *why*
it is slow is more instructive than any speedup chart.

We write the decode loop by hand rather than calling `.generate()`, because the
loop *is* the lesson: one forward pass per token, the KV cache threaded through
as `past_key_values`, and a `torch.cuda.synchronize()` before every timestamp.

That synchronize is not optional. CUDA is asynchronous: without it the CPU
races ahead of the GPU and you time how fast Python can enqueue work, which
produces beautifully fast, completely fictional numbers.
"""),

    code('''
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=dtype).to(device).eval()
print(f"loaded on {device} as {dtype}")


def sync():
    if device == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def generate_timed(prompt_ids, max_new_tokens):
    """Manual greedy loop, timing prefill and every decode step separately."""
    sync(); t0 = time.perf_counter()

    out = model(input_ids=prompt_ids, use_cache=True)          # PREFILL
    next_id = out.logits[:, -1, :].argmax(-1, keepdim=True)
    sync(); ttft = time.perf_counter() - t0

    past, itls, prev = out.past_key_values, [], time.perf_counter()
    for _ in range(max_new_tokens - 1):                        # DECODE
        out = model(input_ids=next_id, past_key_values=past, use_cache=True)
        next_id = out.logits[:, -1, :].argmax(-1, keepdim=True)
        past = out.past_key_values
        sync()
        now = time.perf_counter()
        itls.append(now - prev)
        prev = now

    return ttft, itls
'''),

    code('''
import random

from bench.workloads import PromptBuilder
from bench.metrics import percentile

builder = PromptBuilder(tokenizer, seed=0)
PROMPT_TOKENS, OUTPUT_TOKENS = 256, 64

prompt = builder.text_of_tokens(PROMPT_TOKENS, random.Random(0))
ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
print(f"prompt is {ids.shape[1]} tokens (asked for {PROMPT_TOKENS})")

print("\\nwarmup (discarded)...")
for _ in range(2):
    generate_timed(ids, 8)

print("measuring...")
runs = [generate_timed(ids, OUTPUT_TOKENS) for _ in range(5)]

ttfts = [r[0] for r in runs]
itls  = [v for r in runs for v in r[1]]

print(f"\\n  TTFT       mean {1000*sum(ttfts)/len(ttfts):7.1f} ms   "
      f"(prefill of {ids.shape[1]} tokens)")
print(f"  ITL        mean {1000*sum(itls)/len(itls):7.2f} ms   "
      f"p99 {1000*percentile(itls, 99):7.2f} ms")
print(f"  decode     {len(itls)/sum(itls):7.1f} tok/s  at batch size 1")
'''),

    code('''
# How close is that to the hardware ceiling?
measured = len(itls) / sum(itls)
ceiling  = roofline_decode_tok_s(shape, bandwidth, batch=1, seq_len=PROMPT_TOKENS)

print(f"  measured           {measured:7.1f} tok/s")
print(f"  roofline ceiling   {ceiling:7.1f} tok/s")
print(f"  efficiency         {100*measured/ceiling:7.1f}% of theoretical peak")
print()
print("A plain transformers loop typically lands at 20-40% of the ceiling.")
print("The gap is overhead that has nothing to do with the math:")
print("  - a few hundred separate kernel launches per token (~5-10 us each)")
print("  - Python dispatch between every one of them")
print("  - unfused elementwise ops (RMSNorm, RoPE, SiLU) that each round-trip")
print("    their tensors through HBM")
print("  - attention that re-reads the whole KV cache without a fused kernel")
print()
print("CUDA graphs close the launch-overhead part; kernel fusion closes the")
print("round-trip part. Those two are most of what a serving engine *is*.")
'''),

    md("""
### Now the batching experiment

Batch size is the single biggest lever in inference, so let's watch it work.
Identical prompts, identical output length, only the batch dimension changes.

Predictions from Part 2 before you run it:

1. **Total throughput rises steeply**, because the per-step weight read is
   amortised across the batch.
2. **Per-sequence speed falls slowly**, because each sequence now waits for a
   bigger matmul.
3. **The curve flattens** once arithmetic intensity reaches the ridge point, or
   once the KV cache fills -- whichever comes first.

Watch which of those two actually bites on your GPU.
"""),

    code('''
import gc

@torch.no_grad()
def batch_decode_throughput(batch_size, prompt_tokens=256, new_tokens=32):
    ids_1 = tokenizer(builder.text_of_tokens(prompt_tokens, random.Random(1)),
                      return_tensors="pt").input_ids
    batch = ids_1.repeat(batch_size, 1).to(device)

    out = model(input_ids=batch, use_cache=True)              # prefill, untimed
    next_id = out.logits[:, -1, :].argmax(-1, keepdim=True)
    past = out.past_key_values

    sync(); t0 = time.perf_counter()
    for _ in range(new_tokens):
        out = model(input_ids=next_id, past_key_values=past, use_cache=True)
        next_id = out.logits[:, -1, :].argmax(-1, keepdim=True)
        past = out.past_key_values
    sync()
    elapsed = time.perf_counter() - t0
    return batch_size * new_tokens / elapsed


BATCHES = [1, 2, 4, 8, 16, 32, 64]
measured_tp, ceiling_tp, used = [], [], []

for bs in BATCHES:
    try:
        tp = batch_decode_throughput(bs)
        measured_tp.append(tp)
        ceiling_tp.append(roofline_decode_tok_s(shape, bandwidth, batch=bs, seq_len=288))
        used.append(bs)
        print(f"  batch {bs:>3}: {tp:>8.1f} tok/s total   {tp/bs:>7.1f} tok/s per sequence")
    except torch.cuda.OutOfMemoryError:
        print(f"  batch {bs:>3}: OOM - this is the KV cache wall, exactly as predicted")
        gc.collect(); torch.cuda.empty_cache()
        break
'''),

    code('''
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
from bench.plots import INK, PALETTE

ax1.plot(used, measured_tp, marker="o", color=PALETTE["blue"],
         markeredgecolor=INK["surface"], markeredgewidth=1.5, label="measured (HF)")
ax1.plot(used, ceiling_tp, marker="s", color=INK["muted"], linestyle="-",
         markeredgecolor=INK["surface"], markeredgewidth=1.5, label="roofline ceiling")
ax1.set_xscale("log", base=2); ax1.set_yscale("log")
ax1.xaxis.set_major_formatter(lambda v, _: f"{int(v)}")
ax1.set_xlabel("batch size"); ax1.set_ylabel("total decode tok/s")
ax1.set_title("Batching amortises the weight read")
ax1.grid(axis="both"); ax1.legend()

ax2.plot(used, [t / b for t, b in zip(measured_tp, used)], marker="o",
         color=PALETTE["orange"], markeredgecolor=INK["surface"], markeredgewidth=1.5)
ax2.set_xscale("log", base=2)
ax2.xaxis.set_major_formatter(lambda v, _: f"{int(v)}")
ax2.set_xlabel("batch size"); ax2.set_ylabel("tok/s per sequence")
ax2.set_title("...at the cost of individual latency")
ax2.grid(axis="both")

plt.tight_layout(); save(fig, "00_batching"); plt.show()

speedup = measured_tp[-1] / measured_tp[0]
print(f"\\nbatch 1 -> batch {used[-1]}: {speedup:.1f}x total throughput, "
      f"{measured_tp[-1]/used[-1]/measured_tp[0]:.2f}x per-sequence speed")
print("\\nThat trade is the entire reason continuous batching exists: a serving")
print("engine tries to keep the batch full WITHOUT making any one request wait")
print("for the others to finish. Notebook 01 shows how.")
'''),

    md("""
## Part 7 - Record the baseline

Everything gets saved in the same schema, including this hand-rolled baseline,
so notebook 07 can place `transformers` on the same axes as the real engines.
Being able to say "vLLM is 8x the naive loop *on my hardware, on this
workload*" is worth much more than any number from someone else's blog post.
"""),

    code('''
from bench import EngineSpec, RequestRecord, RunResult, WorkloadSpec, package_versions
from bench.metrics import summarize

records, total_time = [], 0.0
for ttft, step_itls in runs:
    e2e = ttft + sum(step_itls)
    total_time += e2e
    records.append(RequestRecord(
        start_offset_s=0.0, ttft_s=ttft, e2e_s=e2e, itls_s=step_itls,
        prompt_tokens=int(ids.shape[1]), output_tokens=OUTPUT_TOKENS,
        token_count_method="exact",
    ))

baseline = RunResult(
    engine=EngineSpec(
        engine="hf",
        version=package_versions("transformers")["transformers"],
        model_id=MODEL_ID,
        model_family="qwen",
        params_b=round(shape.params() / 1e9, 3),
        dtype=str(dtype).replace("torch.", ""),
        flags={"api": "manual decode loop", "batch_size": 1,
               "cuda_graphs": False, "kv_cache": "transformers DynamicCache"},
    ),
    workload=WorkloadSpec(name="chat", mode="closed", concurrency=1,
                          n_requests=len(records), input_len=int(ids.shape[1]),
                          output_len=OUTPUT_TOKENS),
    hardware=HW,
    metrics=summarize(records, total_time),
    requests=records,
    duration_s=total_time,
    notes="Sequential baseline: plain transformers, batch 1, no serving stack. "
          "The number every engine is implicitly compared against.",
)

path = baseline.save("results/real")
print(f"saved -> {path}")
print(f"\\nbaseline: {baseline.metrics['output_throughput_tok_s']:.1f} output tok/s, "
      f"TTFT p50 {1000*baseline.metrics['ttft_s_p50']:.0f} ms")
'''),

    md("""
## What you should have learned

1. **Prefill and decode are different problems.** Compute-bound vs
   memory-bound, and they fight each other for the GPU.
2. **At batch 1, decode uses ~1% of a GPU's math.** Everything else follows
   from that embarrassing fact.
3. **Batching is the primary lever**, and its cost is per-request latency.
4. **The KV cache, not the weights, decides your concurrency**, and
   `max_model_len` is the knob that controls it.
5. **Which roof you are under determines what optimisation will work.** Moving
   fewer bytes helps memory-bound work; faster math does not.
6. **Measurement is adversarial.** Warmup, client limits, output lengths,
   which-tokens-per-second, closed vs open loop, and default flags will each
   happily hand you a wrong answer.

### Exercises

1. Re-run the batching sweep with `dtype=torch.bfloat16` (A100/L4 only -- a T4
   is sm_75 and has no bf16). Does throughput change? Should it, given the
   roofline? *(It shouldn't move much: same bytes per weight.)*
2. Change `PROMPT_TOKENS` from 256 to 2048. Predict the TTFT change before
   running. Is it 8x, more, or less -- and what does the answer tell you about
   where the quadratic attention term starts to matter?
3. Compute, on paper, how many concurrent 2048-token sequences a 40 GB A100
   could hold for Qwen2.5-7B with GQA, then again pretending it used plain MHA.
   The ratio is why GQA shipped.
4. Use `generate_timed` to plot ITL *against token index* within a single long
   generation. The slow upward drift is the KV cache growing -- attention reads
   more of it every step.

**Next:** [01 - vLLM](01_vllm_deep_dive.ipynb), where continuous batching and
PagedAttention turn that baseline into a server.
"""),
]
