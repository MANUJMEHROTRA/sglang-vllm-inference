"""Notebook 03 - llama.cpp and GGUF: quantization, CPU inference, hybrid offload."""

from tools.nbgen import code, md, setup_cells

NAME = "03_llamacpp_gguf.ipynb"
TITLE = "03 - llama.cpp & GGUF: quantization from first principles"

CELLS = setup_cells(
    NAME,
    TITLE,
    "The only engine here that does not assume a GPU, and the only one that will "
    "run a model bigger than your VRAM. Understanding GGUF's quantization formats "
    "is also the clearest way to understand why quantization works at all.",
    "~45 minutes (includes a ~8 minute CUDA build)",
) + [

    md(r"""
## Why llama.cpp is a different kind of thing

vLLM, SGLang and TensorRT-LLM are *datacenter* servers: they assume one or more
NVIDIA GPUs, they preallocate most of VRAM, and they optimise aggregate
throughput across many concurrent users.

llama.cpp optimises a different objective: **run this model on the hardware I
actually have**, which might be a laptop CPU, an M-series Mac, an old GPU with
6 GB, or a GPU plus CPU together. It is a single C++ binary with no Python and
no CUDA requirement. That is a genuinely different engineering problem, and it
leads to different answers.

Three consequences worth internalising:

1. **Quantization is not optional, it is the point.** On CPU you have ~50-100
   GB/s of memory bandwidth instead of 300-3000. From notebook 00, decode speed
   is bandwidth divided by bytes-per-token. The only lever that moves the
   needle by 4x is making the weights 4x smaller.
2. **Partial offload is a first-class feature.** No other engine here will put
   20 of 32 layers on the GPU and the rest on the CPU.
3. **Single-stream latency matters more than throughput.** It is usually
   serving one person, so the whole design points at low batch-1 latency rather
   than at high aggregate tokens/s.
"""),

    md(r"""
## GGUF, the file format

GGUF (GGML Universal Format) is a single-file container: weights, metadata,
tokenizer, chat template, everything. The layout:

```
+-------------------------------------------------------------+
| magic "GGUF" | version | tensor_count | metadata_kv_count    |
+-------------------------------------------------------------+
| metadata key-value pairs                                     |
|   general.architecture        = "qwen2"                      |
|   qwen2.block_count           = 24                           |
|   qwen2.attention.head_count  = 14                           |
|   tokenizer.ggml.tokens       = [...]      <- whole tokenizer|
|   tokenizer.chat_template     = "{% for ..."                 |
+-------------------------------------------------------------+
| tensor info: name, shape, ggml_type, offset   (x N)          |
+-------------------------------------------------------------+
| padding to alignment                                         |
+-------------------------------------------------------------+
| tensor data  (quantized blocks, laid out back to back)       |
+-------------------------------------------------------------+
```

Two design choices do most of the work:

**Everything is in one file.** No separate tokenizer.json, config.json or chat
template. You can hand someone a `.gguf` and they can run it. Compare with the
HuggingFace layout, which is a directory of files that must agree with each
other.

**The tensor data is mmap-able.** Because quantized blocks are stored
contiguously and aligned, llama.cpp maps the file into memory rather than
reading it. The OS pages in weights on demand and shares them between processes.
This is why a 4 GB model can "load" in milliseconds, and why running two
llama.cpp processes on the same model does not double resident memory.
"""),

    md(r"""
## How the quantization actually works

This is the part worth understanding properly, because it explains why a 4-bit
model is not 4x worse than a 16-bit one -- in fact it is usually almost
indistinguishable.

### The naive version (Q4_0, legacy)

Take 32 consecutive weights. Find the max absolute value. Store one fp16 scale
plus 32 four-bit integers:

```
block of 32 weights -> [ scale: fp16 ][ q0 q1 q2 ... q31 : 4 bits each ]
                          2 bytes          16 bytes        = 18 bytes
                       vs 64 bytes at fp16  ->  4.5 bits per weight
```

Dequantization is `w[i] = scale * q[i]`. The insight is that **weights within a
small block have similar magnitudes**, so one shared scale loses little. The
error is dominated by the few outliers in each block.

### K-quants (Q4_K, Q5_K, Q6_K) -- what you should actually use

K-quants use a two-level scale hierarchy. A **super-block** of 256 weights is
split into 8 sub-blocks of 32. Each sub-block gets its own 6-bit scale and
6-bit minimum; those per-sub-block parameters are themselves quantized against
one fp16 super-block scale:

```
super-block (256 weights)
  +-- fp16 d, fp16 dmin                         <- super-block scales
  +-- 8 x (6-bit scale, 6-bit min)              <- per sub-block, quantized
  +-- 256 x 4-bit quants
  = 144 bytes for 256 weights -> 4.5 bits/weight
```

Having a **minimum** as well as a scale means each sub-block is mapped onto an
affine range `[min, min + 15*scale]` rather than a symmetric one around zero.
Weight distributions are rarely symmetric, so this alone removes a lot of
error at no extra cost.

### The `_M` / `_S` / `_L` suffix: mixed precision

`Q4_K_M` does not mean "everything in 4 bits". It is a *recipe* that assigns
different precision to different tensors:

- `attn_v` and the `ffn_down` projections -> **Q6_K** (more sensitive)
- most other weights -> **Q4_K**
- `token_embd` and `output` -> often Q6_K or left larger

Empirically some tensors matter far more than others. Attention value
projections and the down-projection of the FFN carry more signal per bit, so
spending extra bits there and saving them elsewhere beats uniform precision at
the same average bits/weight. `_S` is the leaner recipe, `_L` the richer one.

### I-quants (IQ2_XXS ... IQ4_XS) and the importance matrix

I-quants use lattice/codebook quantization to reach 2-3 bits/weight. At that
size you cannot afford uniform treatment, so they are normally produced with an
**importance matrix** (`imatrix`): you run a calibration corpus through the
model, record the activation magnitude each weight actually sees, and let
quantization error land preferentially on weights that barely fire.

This is the same idea as AWQ and GPTQ on the GPU side -- *quantization error
should be allocated by how much a weight matters, not uniformly* -- just with a
different codebook and file format.

The practical cost: I-quants are slower to dequantize than K-quants. On a
bandwidth-starved CPU that trade is usually still worth it; on a GPU with
bandwidth to spare it often is not.

### What to pick

| Quant | bits/weight | size (0.5B) | when |
|---|---:|---:|---|
| `F16` | 16.0 | ~1.0 GB | reference for quality comparisons |
| `Q8_0` | 8.5 | ~0.5 GB | effectively lossless; use if it fits |
| `Q6_K` | 6.6 | ~0.4 GB | very close to Q8 at smaller size |
| **`Q5_K_M`** | 5.7 | ~0.35 GB | best quality/size for most people |
| **`Q4_K_M`** | 4.9 | ~0.3 GB | **the default recommendation** |
| `Q3_K_M` | 3.9 | ~0.25 GB | noticeable degradation begins |
| `IQ2_XXS` | 2.1 | ~0.15 GB | last resort to fit at all |

The rule of thumb that holds up remarkably well: **a larger model at Q4_K_M
beats a smaller model at F16 of the same file size.** We will test that claim
rather than repeat it.
"""),

    code('''
%pip install -q gguf huggingface_hub

from huggingface_hub import hf_hub_download
from pathlib import Path

GGUF_REPO = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
GGUF_DIR = Path("/content/gguf") if Path("/content").exists() else Path("./gguf")
GGUF_DIR.mkdir(parents=True, exist_ok=True)

# Quants to compare. Names follow the repo's own file naming.
QUANTS = ["q4_k_m", "q5_k_m", "q8_0", "fp16"]
paths = {}
for q in QUANTS:
    try:
        paths[q] = Path(hf_hub_download(GGUF_REPO,
                        filename=f"qwen2.5-0.5b-instruct-{q}.gguf",
                        local_dir=GGUF_DIR))
        print(f"  {q:<8} {paths[q].stat().st_size/1024**2:>8.1f} MB")
    except Exception as exc:
        print(f"  {q:<8} unavailable: {exc}")
'''),

    code('''
# Look inside a GGUF file: metadata, then the per-tensor quantization types.
from gguf import GGUFReader

reader = GGUFReader(str(paths["q4_k_m"]))

print("=== metadata (selected) ===")
for field in reader.fields.values():
    name = field.name
    if any(k in name for k in ("architecture", "block_count", "head_count",
                               "embedding_length", "feed_forward", "file_type",
                               "context_length", "rope")):
        try:
            print(f"  {name:<42} {field.contents()}")
        except Exception:
            pass

print(f"\\n=== tensors: {len(reader.tensors)} ===")
from collections import Counter
types = Counter(t.tensor_type.name for t in reader.tensors)
for tname, count in types.most_common():
    print(f"  {tname:<12} {count:>4} tensors")

print("\\n=== the mixed-precision recipe, tensor by tensor (layer 0) ===")
for t in reader.tensors:
    if "blk.0." in t.name or t.name in ("token_embd.weight", "output.weight",
                                        "output_norm.weight"):
        mb = t.n_bytes / 1024**2
        print(f"  {t.name:<32} {t.tensor_type.name:<10} {str(list(t.shape)):<18} {mb:>7.2f} MB")
'''),

    md("""
Look carefully at that last table. In a `Q4_K_M` file you should see `Q6_K` on
`attn_v` and `ffn_down` while `attn_q`, `attn_k` and `ffn_gate`/`ffn_up` are
`Q4_K`, with the norms left in `F32` (they are tiny and extremely sensitive).
That is the mixed-precision recipe from the explanation above, visible in the
actual file.

Norms staying F32 is worth noting: they are a few thousand parameters out of
500 million, so their size is irrelevant, but quantizing them would be
disastrous. **Bits should go where the sensitivity is, not where the parameters
are.** The same principle drives every quantization scheme in this repo.
"""),

    md("""
## Building llama.cpp with CUDA

We build from source rather than using a prebuilt wheel, because we want the
whole toolbox: `llama-server` (the OpenAI-compatible endpoint our harness
drives), `llama-bench` (its own micro-benchmark), `llama-perplexity` (quality
measurement) and `llama-quantize`.

The build takes roughly 8 minutes on a Colab runtime. `-DGGML_CUDA=ON` is what
enables GPU offload; without it you get a CPU-only binary, which is also a
perfectly valid thing to benchmark and we will do that too.
"""),

    code('''
import subprocess, os
from pathlib import Path

LCPP = Path("/content/llama.cpp") if Path("/content").exists() else Path("./llama.cpp")

if not (LCPP / "build" / "bin" / "llama-server").exists():
    if not LCPP.exists():
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/ggml-org/llama.cpp", str(LCPP)], check=True)
    cuda_flag = "ON" if CAPS["has_gpu"] else "OFF"
    print(f"configuring with GGML_CUDA={cuda_flag} (this takes a few minutes)")
    subprocess.run(["cmake", "-B", str(LCPP / "build"), "-S", str(LCPP),
                    f"-DGGML_CUDA={cuda_flag}",
                    "-DCMAKE_CUDA_ARCHITECTURES=native",
                    "-DCMAKE_BUILD_TYPE=Release",
                    "-DLLAMA_CURL=OFF"], check=True)
    subprocess.run(["cmake", "--build", str(LCPP / "build"), "--config", "Release",
                    "-j", str(os.cpu_count())], check=True)

BIN = LCPP / "build" / "bin"
os.environ["PATH"] = f"{BIN}:{os.environ['PATH']}"
print("\\nbuilt:", sorted(p.name for p in BIN.glob("llama-*"))[:12])
'''),

    md("""
## Experiment 1 - The quantization trade-off curve

Three axes, measured on the same model:

- **size** (bytes on disk, which is also bytes moved per token),
- **speed** (decode tok/s -- should track size almost exactly, because decode
  is bandwidth-bound),
- **quality** (perplexity on held-out text -- lower is better).

The prediction from notebook 00 is sharp: if decode is purely bandwidth-bound,
**tok/s should be inversely proportional to file size**. A Q4_K_M file that is
3.2x smaller than F16 should decode ~3.2x faster. Deviations from that line are
dequantization overhead, and measuring them is how you find out whether a quant
format is actually paying for itself.
"""),

    code('''
# llama-bench is llama.cpp's own micro-benchmark: pp = prompt processing
# (prefill), tg = text generation (decode). It reports tok/s with error bars
# and handles its own warmup, so it is the right tool for this comparison.
import json, subprocess

bench_rows = []
for q, path in paths.items():
    print(f"benchmarking {q} ...")
    out = subprocess.run(
        [str(BIN / "llama-bench"), "-m", str(path),
         "-p", "512", "-n", "128", "-ngl", "99" if CAPS["has_gpu"] else "0",
         "-r", "3", "-o", "json"],
        capture_output=True, text=True, check=True).stdout
    for row in json.loads(out):
        bench_rows.append({
            "quant": q,
            "size_mb": path.stat().st_size / 1024**2,
            "test": "prefill" if row["n_prompt"] else "decode",
            "tok_s": row["avg_ts"],
            "stddev": row.get("stddev_ts", 0.0),
        })

import pandas as pd
bench = pd.DataFrame(bench_rows)
bench.pivot_table(index="quant", columns="test", values="tok_s").round(1)
'''),

    code('''
# Quality: perplexity on held-out text. Lower is better.
# Perplexity = exp(mean negative log-likelihood) -- "how surprised is the model
# by text it has not seen". It is a blunt instrument (it does not measure
# instruction following or reasoning) but it is sensitive to exactly the kind
# of damage quantization does, and it is cheap and reproducible.
%pip install -q datasets

from datasets import load_dataset
corpus = Path("wikitext_test.txt")
if not corpus.exists():
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\\n".join(ds["text"])[:600_000]     # ~150k tokens is plenty
    corpus.write_text(text)
print(f"corpus: {corpus.stat().st_size/1024:.0f} KB")

ppl = {}
for q, path in paths.items():
    print(f"perplexity {q} ...", end=" ", flush=True)
    out = subprocess.run(
        [str(BIN / "llama-perplexity"), "-m", str(path), "-f", str(corpus),
         "-ngl", "99" if CAPS["has_gpu"] else "0", "-c", "512", "--chunks", "40"],
        capture_output=True, text=True)
    # llama-perplexity prints "Final estimate: PPL = 12.3456 +/- 0.1234"
    line = [ln for ln in out.stderr.splitlines() if "Final estimate" in ln]
    if line:
        ppl[q] = float(line[0].split("PPL =")[1].split("+/-")[0].strip())
        print(f"{ppl[q]:.4f}")
    else:
        print("failed"); print(out.stderr[-500:])
'''),

    code('''
import matplotlib.pyplot as plt
from bench.plots import INK, PALETTE, save, use_theme
use_theme()

decode = bench[bench.test == "decode"].set_index("quant")
order = [q for q in ["fp16", "q8_0", "q5_k_m", "q4_k_m"] if q in decode.index]

sizes  = [decode.loc[q, "size_mb"] for q in order]
speeds = [decode.loc[q, "tok_s"] for q in order]
quals  = [ppl.get(q) for q in order]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))

# Measured speed vs the pure-bandwidth prediction.
ref_size, ref_speed = sizes[0], speeds[0]
predicted = [ref_speed * ref_size / s for s in sizes]
ax1.plot(sizes, speeds, marker="o", color=PALETTE["aqua"],
         markeredgecolor=INK["surface"], markeredgewidth=1.5, label="measured")
ax1.plot(sizes, predicted, color=INK["muted"], linewidth=1.3,
         label="pure bandwidth prediction")
for q, s, v in zip(order, sizes, speeds):
    ax1.annotate(q, (s, v), textcoords="offset points", xytext=(6, 4),
                 fontsize=9, color=INK["secondary"])
ax1.set_xlabel("model size (MB)"); ax1.set_ylabel("decode tok/s")
ax1.set_title("Smaller weights decode proportionally faster")
ax1.legend(); ax1.grid(axis="both")

if all(q is not None for q in quals):
    ax2.plot(sizes, quals, marker="s", color=PALETTE["orange"],
             markeredgecolor=INK["surface"], markeredgewidth=1.5)
    for q, s, v in zip(order, sizes, quals):
        ax2.annotate(q, (s, v), textcoords="offset points", xytext=(6, 4),
                     fontsize=9, color=INK["secondary"])
    ax2.set_xlabel("model size (MB)"); ax2.set_ylabel("perplexity (lower = better)")
    ax2.set_title("...and quality barely moves until 4 bits")
    ax2.grid(axis="both")

plt.tight_layout(); save(fig, "03_quant_tradeoff"); plt.show()

if ppl.get("fp16") and ppl.get("q4_k_m"):
    delta = 100 * (ppl["q4_k_m"] / ppl["fp16"] - 1)
    ratio = decode.loc["q4_k_m", "size_mb"] / decode.loc["fp16", "size_mb"]
    print(f"Q4_K_M vs F16: {1/ratio:.2f}x smaller, "
          f"{decode.loc['q4_k_m','tok_s']/decode.loc['fp16','tok_s']:.2f}x faster decode, "
          f"perplexity {delta:+.2f}%")
    print("\\nA perplexity increase under ~2% is generally imperceptible in use.")
    print("That is the whole case for 4-bit quantization, in one line.")
'''),

    md(r"""
## Experiment 2 - The offload curve (`-ngl`)

llama.cpp's defining feature: put the first N layers on the GPU and run the
rest on the CPU. The model does not need to fit in VRAM.

The naive expectation is a straight line -- half the layers on the GPU should
give half the speedup. It does not work that way, and the reason is instructive.

```
-ngl 0                      -ngl 12 (half)                -ngl 24 (all)

all 24 layers on CPU        12 on GPU, 12 on CPU          all on GPU
                            every token crosses PCIe
~50 GB/s bandwidth          at the boundary                ~300 GB/s
                            + CPU half still runs at
                            CPU speed
```

The CPU half still runs at CPU speed, and Amdahl's law applies hard: if the CPU
portion is 80% of the time, moving 50% of the layers to a GPU that is 6x faster
still leaves you dominated by the CPU. Real offload curves are therefore
**convex** -- most of the benefit arrives only in the last stretch, when the
whole model finally fits.

The practical conclusion matters: **a quant level that fully fits in VRAM
usually beats a larger quant that half-fits**, even though the larger quant has
better perplexity. Q4_K_M fully offloaded can be several times faster than
Q8_0 at 60% offload.
"""),

    code('''
if CAPS["has_gpu"]:
    n_layers = 24
    levels = [0, 6, 12, 18, 24]
    offload_rows = []
    for ngl in levels:
        out = subprocess.run(
            [str(BIN / "llama-bench"), "-m", str(paths["q4_k_m"]),
             "-p", "256", "-n", "128", "-ngl", str(ngl), "-r", "2", "-o", "json"],
            capture_output=True, text=True, check=True).stdout
        for row in json.loads(out):
            offload_rows.append({"ngl": ngl,
                                 "test": "prefill" if row["n_prompt"] else "decode",
                                 "tok_s": row["avg_ts"]})
        print(f"  -ngl {ngl:>2}: " + "  ".join(
            f"{r['test']} {r['tok_s']:.1f} tok/s" for r in offload_rows[-2:]))

    off = pd.DataFrame(offload_rows)
    dec = off[off.test == "decode"].sort_values("ngl")

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.plot(dec.ngl, dec.tok_s, marker="o", color=PALETTE["aqua"],
            markeredgecolor=INK["surface"], markeredgewidth=1.5, label="measured")
    cpu, gpu = dec.tok_s.iloc[0], dec.tok_s.iloc[-1]
    ax.plot(dec.ngl, [cpu + (gpu - cpu) * n / n_layers for n in dec.ngl],
            color=INK["muted"], linewidth=1.3, label="naive linear expectation")
    ax.set_xlabel("layers offloaded to GPU (-ngl)")
    ax.set_ylabel("decode tok/s")
    ax.set_title("Partial offload is convex, not linear")
    ax.legend(); ax.grid(axis="both")
    save(fig, "03_offload_curve"); plt.show()

    print(f"\\n  CPU only (-ngl 0):  {cpu:6.1f} tok/s")
    print(f"  full offload:       {gpu:6.1f} tok/s  ({gpu/cpu:.1f}x)")
    half = dec[dec.ngl == 12].tok_s.iloc[0]
    print(f"  half offload:       {half:6.1f} tok/s  "
          f"({100*(half-cpu)/(gpu-cpu):.0f}% of the way there at 50% of the layers)")
else:
    print("No GPU -- skipping the offload curve. The CPU numbers above still stand.")
'''),

    md("""
## Experiment 3 - llama-server through the shared harness

Now put llama.cpp on the same axes as vLLM and SGLang: same client, same
workload, same metric definitions, same result schema.

Expect it to lose badly on throughput at concurrency, and that is not a
criticism -- llama.cpp is not trying to win that. Watch instead what happens at
**concurrency 1**, where its lean C++ path and absent Python overhead make it
genuinely competitive, sometimes better than the datacenter servers on TTFT for
a short prompt.
"""),

    code('''
from bench import (EndpointConfig, EngineSpec, PromptBuilder, discover_model_id,
                   free_gpu, kill_stale, llamacpp_cmd, preset, run, serve,
                   sweep_concurrency)
from bench.plots import sweep_curve, table
from transformers import AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
builder = PromptBuilder(tokenizer, seed=0)
PORT = 8000

kill_stale(); free_gpu()

# ctx_size is shared across the n_parallel slots, so each concurrent request
# gets ctx_size/n_parallel tokens. Size it for the sweep's top concurrency.
N_PARALLEL = 16
cmd = llamacpp_cmd(str(paths["q4_k_m"]), port=PORT,
                   n_gpu_layers=99 if CAPS["has_gpu"] else 0,
                   ctx_size=1024 * N_PARALLEL, n_parallel=N_PARALLEL,
                   binary=str(BIN / "llama-server"))
print(" ".join(cmd))

server_ctx = serve(cmd, port=PORT, log_name="llamacpp.log")
base_url = server_ctx.__enter__()
cfg = EndpointConfig(base_url=base_url,
                     model=run(discover_model_id(base_url)), engine="llamacpp")
print("serving:", cfg.model)
'''),

    code('''
engine_spec = EngineSpec(
    engine="llamacpp", version="build-from-source", model_id="qwen2.5-0.5b-instruct-q4_k_m",
    model_family="qwen", params_b=0.49, dtype="Q4_K_M", quantization="gguf-Q4_K_M",
    max_model_len=1024,
    flags={"n_gpu_layers": 99 if CAPS["has_gpu"] else 0, "n_parallel": N_PARALLEL,
           "flash_attn": True, "cont_batching": True, "api": "openai-server"},
)

chat = preset("chat", input_len=256, output_len=256, n_requests=48)
sweep = run(sweep_concurrency(cfg, engine_spec, chat, builder,
                              levels=(1, 2, 4, 8, 16), hardware=HW,
                              notes="llama.cpp Q4_K_M concurrency sweep"))
for r in sweep:
    r.save("results/real")

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
sweep_curve(sweep, "output_throughput_tok_s", ylabel="output tok/s",
            title="llama.cpp throughput vs concurrency", ax=axes[0])
sweep_curve(sweep, "ttft_s_p50", scale=1000, ylabel="p50 TTFT (ms)",
            title="TTFT", ax=axes[1])
plt.tight_layout(); save(fig, "03_llamacpp_sweep"); plt.show()
table(sweep)
'''),

    code('''
server_ctx.__exit__(None, None, None)
kill_stale(); free_gpu()
'''),

    md("""
## When to choose llama.cpp

**Strong fit**
- **No NVIDIA GPU.** CPU, Apple Silicon (Metal), AMD (ROCm/Vulkan), Intel. No
  other engine here covers that range.
- **The model does not fit in VRAM.** Partial offload is unique to it.
- **Single user, local, private.** A desktop assistant, an on-device feature, an
  air-gapped deployment. Batch-1 latency is where it is strongest.
- **Deployment simplicity.** One static binary, no Python, no CUDA toolkit, no
  container. This is worth more operationally than most benchmarks capture.
- **You need obscure quantization levels** to fit a specific memory budget.

**Weak fit**
- **Many concurrent users.** It has continuous batching, but it is not the
  design centre and it will lose to vLLM/SGLang by a wide margin under load.
- **Datacenter GPU throughput.** The CUDA kernels are good, not state of the
  art; TensorRT-LLM and vLLM have far more tuning effort behind them.
- **Fast-moving model support.** New architectures need C++ work plus a
  conversion script, so support lags the Python engines by weeks or months.

### The quantization decision, condensed

1. If the F16 or Q8_0 model fits comfortably, use it -- there is nothing to win.
2. Otherwise **Q4_K_M** is the default. ~4.9 bits/weight, typically under 2%
   perplexity degradation, and it fits almost anywhere.
3. If it still does not fit, prefer **a bigger model at a lower quant over a
   smaller model at a higher quant** -- down to about 3 bits, below which
   degradation becomes obvious.
4. Full offload beats a better quant that only half-fits. Check the `-ngl`
   curve before choosing.

### Exercises

1. Download `Qwen2.5-1.5B-Instruct-GGUF` at Q4_K_M (~1.0 GB) and compare it
   against the 0.5B at F16 (~1.0 GB) -- same file size, different trade. Compare
   perplexity *and* decode speed. Which would you deploy?
2. Use `llama-quantize` to make a Q3_K_M from the F16 yourself, then measure its
   perplexity. Where does degradation become visible to you in actual chat?
3. Run `llama-bench` with `-t 1,2,4,8` on CPU only. Where do extra threads stop
   helping, and what does that tell you about whether CPU decode is bandwidth-
   or compute-bound?
4. Compare `-fa on` against `-fa off` on a 2048-token prompt. Flash attention
   should move prefill far more than decode -- explain why using notebook 00.

**Next:** [04 - TensorRT-LLM](04_tensorrt_llm.ipynb), the opposite philosophy:
compile everything ahead of time.
"""),
]
