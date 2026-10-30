"""Notebook 04 - TensorRT-LLM: ahead-of-time compilation."""

from tools.nbgen import code, md, setup_cells

NAME = "04_tensorrt_llm.ipynb"
TITLE = "04 - TensorRT-LLM: compile the model, then run it"

CELLS = setup_cells(
    NAME,
    TITLE,
    "Every other engine here interprets the model graph at run time. TensorRT-LLM "
    "compiles it first -- fusing, autotuning and specialising for your exact GPU and "
    "shape bounds. That buys the lowest latency available on NVIDIA hardware, and "
    "costs flexibility in ways worth feeling for yourself.",
    "~40 minutes (A100/L4 recommended; see the hardware gate below)",
) + [

    md(r"""
## A different philosophy: compile, don't interpret

vLLM and SGLang execute a PyTorch graph. Every forward pass dispatches Python,
looks up kernels, and launches them. CUDA graphs remove much of the launch
overhead, but the kernels themselves are generic: written once to work for any
shape, any head count, any batch size.

TensorRT-LLM goes the other way. You hand it a model plus **bounds** -- max
batch size, max input length, max output length -- and it produces a compiled
`.engine` file specialised to those bounds *and to your exact GPU*. During the
build it:

- **fuses aggressively**: QKV projections into one GEMM, bias+activation into
  the preceding matmul, RMSNorm folded into neighbours. Each fusion removes an
  HBM round trip, which is exactly the currency that matters for the
  memory-bound decode phase (notebook 00).
- **autotunes kernels**: for each matmul shape, it actually benchmarks several
  tile sizes, scheduling strategies and instruction mixes on *your* GPU and
  keeps the fastest. This is why build time is minutes rather than seconds, and
  why an engine is not portable to another GPU model.
- **picks optimal memory layouts** and inserts the fewest possible transposes.
- **specialises on the shape bounds**, so there is no generic-shape overhead.

```
vLLM / SGLang                        TensorRT-LLM
---------------                      ------------
load weights       (seconds)         build engine      (minutes, ONCE)
   |                                     |
   v                                     v
interpret graph per request          replay compiled engine
generic kernels                      kernels tuned for THIS gpu + shapes
any shape, any time                  shapes within the compiled bounds
```

### What it costs you

1. **Build time.** Minutes to hours. A CI step, not a startup step.
2. **Engine files are not portable.** Built for sm_80? It will not run on
   sm_89. Different TensorRT version? Rebuild. This is operationally real: your
   artifact registry now holds one engine per GPU type.
3. **Shape bounds are hard limits.** Exceed `max_input_len` and the request
   fails. Set the bounds generously and you waste memory; set them tightly and
   you reject traffic.
4. **Model support lags.** A new architecture needs a TensorRT-LLM
   implementation, not just a `transformers` class.

### Where it wins

Lowest **time-to-first-token** and lowest **batch-1 latency** on NVIDIA
hardware, consistently. If you are serving a fixed model at scale on known
hardware and latency is the product, the build step is a small price. If you
are iterating on models weekly, it is a tax you pay every week.
"""),

    code('''
# ---- hardware gate --------------------------------------------------------
# TensorRT-LLM's interesting features have real compute-capability floors:
#   sm_75 (T4)  : fp16 and int8 only. No bf16, no FP8, and INT4 AWQ/GPTQ are
#                 NOT supported below sm_80. Recent releases are developed and
#                 tested on sm_89/sm_90; Turing is community-support at best.
#   sm_80 (A100): bf16, INT4 AWQ/GPTQ, SmoothQuant.
#   sm_89 (L4)  : adds FP8.
#   sm_90 (H100): FP8 with the fastest paths, FlashAttention-3.
sm = float(HW.gpu_compute_capability or 0)

print(f"GPU: {HW.gpu_name}  (sm_{HW.gpu_compute_capability})")
print()
if sm >= 8.9:
    print("  Full support: fp16, bf16, FP8, INT4 AWQ. Everything in this notebook runs.")
elif sm >= 8.0:
    print("  Good support: fp16, bf16, INT4 AWQ, SmoothQuant. No FP8 -- that")
    print("  section will be skipped.")
elif sm >= 7.5:
    print("  LIMITED. fp16/int8 only; no bf16, no FP8, no INT4 AWQ/GPTQ.")
    print("  Recent TensorRT-LLM releases are not tested on Turing and the")
    print("  install may simply fail. Expect to skip most of this notebook.")
    print("  On Colab Pro, switch to an L4 or A100 runtime to run it properly.")
else:
    print("  No supported GPU. Read the notebook; the measurements will not run.")

TRT_OK = sm >= 8.0
'''),

    code('''
# TensorRT-LLM is a large install (several GB: TensorRT, its plugins, and a
# pinned torch). Expect 8-15 minutes on a cold runtime, and expect it to fail
# on an unsupported architecture -- which is itself a finding worth recording.
TRT_INSTALLED = False
if TRT_OK:
    import subprocess, sys
    proc = subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                           "tensorrt_llm"], capture_output=True, text=True)
    TRT_INSTALLED = proc.returncode == 0
    if not TRT_INSTALLED:
        print("install failed -- last 25 lines:")
        print("\\n".join((proc.stderr or proc.stdout).splitlines()[-25:]))

if TRT_INSTALLED:
    from bench import package_versions
    for name, v in package_versions("tensorrt_llm", "tensorrt", "torch").items():
        print(f"  {name:<18} {v}")
else:
    print("\\nSkipping the measured sections. The explanations below still apply,")
    print("and notebook 07 handles a missing engine without breaking.")
'''),

    md("""
## The two backends

Modern TensorRT-LLM ships two execution paths, and knowing which you are using
matters when you read any benchmark of it.

**`--backend pytorch`** (the newer default). No ahead-of-time engine build. It
behaves much like vLLM: load weights, serve immediately, flexible shapes. You
get TensorRT-LLM's kernels and scheduler but not the full compile-time fusion
and autotuning. Startup is seconds.

**`--backend tensorrt`** (the classic path). Builds a `.engine` first via
`trtllm-build`. This is where the ahead-of-time advantages actually come from,
and where the published "fastest inference" numbers are measured.

A lot of casual comparisons run the pytorch backend and report the result as
"TensorRT-LLM", which understates it; others quote compiled-engine numbers
against other engines' default configs, which overstates it. We will run the
pytorch backend for the apples-to-apples sweep (it is what fits in a Colab
session) and record the backend in the result file, so notebook 07 can label it
honestly rather than silently.
"""),

    code('''
import matplotlib.pyplot as plt

from bench import (EndpointConfig, EngineSpec, PromptBuilder, discover_model_id,
                   free_gpu, kill_stale, load_tokenizer, preset, probe, run,
                   serve, sweep_concurrency, trtllm_cmd)
from bench.plots import save, sweep_curve, table, use_theme
from bench.servers import print_log

use_theme()
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
PORT = 8000
builder = PromptBuilder(load_tokenizer(MODEL_ID), seed=0)

trt_sweep = []
if TRT_INSTALLED:
    kill_stale(); free_gpu()
    cmd = trtllm_cmd(MODEL_ID, port=PORT, backend="pytorch",
                     max_batch_size=64, max_num_tokens=8192, max_seq_len=4096,
                     kv_cache_fraction=0.75)
    print(" ".join(cmd))

    server_ctx = serve(cmd, port=PORT, log_name="trtllm.log", timeout_s=1800)
    base_url = server_ctx.__enter__()
    try:
        cfg = EndpointConfig(base_url=base_url,
                             model=run(discover_model_id(base_url)),
                             engine="tensorrt_llm")
        print("serving:", cfg.model)
        print(run(probe(cfg, "The capital of France is", max_tokens=8)))
    except Exception:
        print_log(PORT, 40); raise
'''),

    code('''
if TRT_INSTALLED:
    from bench import package_versions
    engine_spec = EngineSpec(
        engine="tensorrt_llm", version=package_versions("tensorrt_llm")["tensorrt_llm"],
        model_id=MODEL_ID, model_family="qwen", params_b=0.49,
        dtype="bfloat16" if CAPS["bf16"] else "float16", max_model_len=4096,
        flags={"backend": "pytorch", "max_batch_size": 64,
               "kv_cache_free_gpu_memory_fraction": 0.75, "api": "openai-server",
               "aot_compiled": False},
    )
    chat = preset("chat", input_len=256, output_len=256, n_requests=64)
    trt_sweep = run(sweep_concurrency(cfg, engine_spec, chat, builder,
                                      levels=(1, 2, 4, 8, 16, 32, 64), hardware=HW,
                                      notes="TensorRT-LLM pytorch backend sweep"))
    for r in trt_sweep:
        r.save("results/real")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    sweep_curve(trt_sweep, "output_throughput_tok_s", ylabel="output tok/s",
                title="Throughput", ax=axes[0])
    sweep_curve(trt_sweep, "ttft_s_p50", scale=1000, ylabel="p50 TTFT (ms)",
                title="TTFT -- where TensorRT-LLM is strongest", ax=axes[1])
    plt.tight_layout(); save(fig, "04_trtllm_sweep"); plt.show()
    table(trt_sweep)
'''),

    md("""
## Building an actual engine

This is the part that distinguishes TensorRT-LLM from everything else, so it is
worth doing once even though the pytorch backend is more convenient.

The flow has two stages:

```
HF checkpoint ──convert_checkpoint.py──▶ TRT-LLM checkpoint ──trtllm-build──▶ .engine
                (reshape weights,                            (fuse, autotune,
                 apply quantization)                          specialise shapes)
```

The build flags that matter most:

| Flag | What it does | Getting it wrong |
|---|---|---|
| `--max_batch_size` | upper bound on concurrent sequences | too low: rejected load; too high: wasted memory |
| `--max_input_len` | longest prompt | exceeded -> request fails outright |
| `--max_seq_len` | input + output | same |
| `--max_num_tokens` | tokens per scheduled batch | the in-flight-batching granularity knob |
| `--gemm_plugin` | use tuned GEMM plugins | leaving it off loses much of the point |
| `--use_paged_context_fmha` | paged attention + chunked context | needed for prefix reuse |

Note how `max_input_len` is a *compile-time* constant here, while in vLLM
`--max-model-len` is a runtime setting. That single difference accounts for
most of the operational friction: changing it means a rebuild, a new artifact,
and a redeploy.
"""),

    code('''
# The build is memory- and time-hungry (10-20 minutes for a 0.5B model,
# much longer for 7B). Set BUILD_ENGINE = True to actually run it.
BUILD_ENGINE = False

if TRT_INSTALLED and BUILD_ENGINE:
    import subprocess, time
    from pathlib import Path

    work = Path("/content/trt") if Path("/content").exists() else Path("./trt")
    ckpt, engine_dir = work / "ckpt", work / "engine"
    work.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    subprocess.run([
        "trtllm-build",
        "--checkpoint_dir", str(ckpt),
        "--output_dir", str(engine_dir),
        "--gemm_plugin", "auto",
        "--max_batch_size", "64",
        "--max_input_len", "2048",
        "--max_seq_len", "4096",
        "--max_num_tokens", "8192",
        "--use_paged_context_fmha", "enable",
    ], check=True)
    build_s = time.perf_counter() - t0

    size_mb = sum(f.stat().st_size for f in engine_dir.rglob("*")) / 1024**2
    print(f"\\nbuild took {build_s/60:.1f} minutes, engine is {size_mb:.0f} MB")
    print("That build time is the number to remember: it is a CI cost you pay")
    print("on every model change, every shape-bound change, and every GPU type.")
else:
    print("BUILD_ENGINE is False -- skipping the ahead-of-time build.")
    print("Flip it to True if you have the session time and want to feel the cost.")
'''),

    md(r"""
## Quantization on the TensorRT-LLM side

The same principle as GGUF's K-quants from notebook 03 -- spend bits where
sensitivity is -- expressed with different machinery:

**FP8 (sm_89+)**: an 8-bit *floating point* format (e4m3), not an integer. It
keeps an exponent, so it handles the wide dynamic range of activations far
better than int8, which is why FP8 usually needs no calibration while int8
does. Halves weight bytes and, critically, can halve the **KV cache** too --
directly doubling concurrency. On an L4 or H100 this is usually the first thing
to try.

**INT8 SmoothQuant**: activations have outlier channels that wreck naive int8
quantization. SmoothQuant migrates the difficulty from activations into
weights by rescaling both sides -- `Y = (X/s)(sW)` leaves the product
unchanged while making `X/s` easier to quantize. Needs a calibration pass. Not
supported below sm_80.

**INT4 AWQ**: activation-aware weight quantization. Identifies the ~1% of
weight channels that matter most (by activation magnitude) and protects them
with per-channel scaling before quantizing the rest to 4 bits. Same insight as
llama.cpp's imatrix, different implementation. Weight-only, so it helps the
memory-bound decode phase a lot and prefill very little. Requires sm_80+.

The decision rule follows straight from the roofline:

- **Decode-dominated** (chat, long generations) -> weight-only 4-bit is the big
  win. You are on the memory roof; moving fewer bytes is the only lever.
- **Prefill-dominated** (summarisation, RAG) -> FP8 or int8 *activation*
  quantization helps, because you are near the compute roof and the tensor
  cores run faster on narrower types. Weight-only 4-bit barely helps here and
  can even hurt, since dequantization costs arithmetic you did not have to
  spare.

That asymmetry is the single most useful thing to know about choosing a
quantization scheme, and it is why "which quantization is best" has no
workload-independent answer.
"""),

    code('''
if TRT_INSTALLED:
    import matplotlib.pyplot as plt
    from bench.plots import INK, PALETTE

    # The decision rule above, drawn. Not measured data -- a model of where
    # each scheme helps, derived from which roof the phase sits under.
    schemes = ["fp16\\n(baseline)", "FP8\\n(sm_89+)", "INT8\\nSmoothQuant", "INT4 AWQ\\n(weight-only)"]
    decode_gain  = [1.00, 1.70, 1.60, 2.90]
    prefill_gain = [1.00, 1.60, 1.45, 1.05]

    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    from bench.plots import grouped_bar
    grouped_bar(schemes, {"decode (memory-bound)": decode_gain,
                          "prefill (compute-bound)": prefill_gain},
                ylabel="relative speed vs fp16",
                title="Where each quantization scheme pays off (model, not measurement)",
                colors={"decode (memory-bound)": PALETTE["blue"],
                        "prefill (compute-bound)": PALETTE["orange"]}, ax=ax)
    ax.axhline(1.0, color=INK["axis"], linewidth=1.0)
    save(fig, "04_quant_regimes"); plt.show()

    print("Read it as a shape, not as numbers: INT4 weight-only transforms")
    print("decode and does nothing for prefill, because prefill was never")
    print("waiting on weight bytes. Measure your own workload before choosing.")
'''),

    code('''
if TRT_INSTALLED:
    server_ctx.__exit__(None, None, None)
from bench import kill_stale, free_gpu
kill_stale(); free_gpu()
'''),

    md("""
## When to choose TensorRT-LLM

**Strong fit**
- NVIDIA hardware you control and will not change often.
- A stable model that you serve at scale -- the build cost amortises.
- Latency is the product: TTFT and batch-1 latency are where it reliably wins.
- You want FP8 on Ada/Hopper with first-class support.
- You are already deploying through Triton Inference Server (notebook 05); the
  TensorRT-LLM backend is the intended pairing.

**Weak fit**
- Rapid model iteration. Every change means a rebuild.
- Heterogeneous or unknown GPUs. Engines are not portable.
- Unpredictable shapes. Hard compile-time bounds become a capacity planning
  problem rather than a tuning knob.
- Non-NVIDIA hardware. Nothing transfers.
- Small teams without CI for artifacts. Managing one engine per GPU type per
  model version is real operational work that benchmarks never show.

> **Honest summary of the performance gap.** On a well-tuned comparison with
> the *same* quantization and the *same* workload, a compiled TensorRT-LLM
> engine is typically modestly ahead of vLLM and SGLang on latency, and roughly
> comparable on throughput. It is not the 3-5x that marketing charts sometimes
> suggest -- those usually compare a tuned engine against another framework's
> defaults, often with different quantization. Which is exactly why this repo
> records flags and dtypes in every result file.

### Exercises

1. Set `BUILD_ENGINE = True` and time the build. Then compute: how many
   requests must you serve before the build time pays for itself at the latency
   improvement you measured?
2. Build with `--max_input_len 512`, then send a 1024-token prompt. Read the
   error. This is the failure mode that does not exist in vLLM.
3. If you are on an L4 or better, build an FP8 engine and compare TTFT and KV
   cache capacity against fp16. The concurrency increase should roughly match
   the KV halving.
4. Compare the pytorch backend against a compiled engine on identical
   workloads. How much of TensorRT-LLM's advantage comes from compilation, and
   how much from its kernels and scheduler?

**Next:** [05 - Triton Inference Server](05_triton_inference_server.ipynb),
which is not an engine at all.
"""),
]
