# Inference engines, from first principles

A study project comparing **vLLM**, **SGLang**, **llama.cpp / GGUF**,
**TensorRT-LLM** and **Triton Inference Server** — plus encoder serving
(BERT-family) and hands-on CUDA kernel optimization in Triton.

The goal is not a leaderboard. It is to be able to *derive* roughly how fast a
model can possibly run on a given GPU, then measure how close each stack gets,
and understand every gap.

Everything runs on **Colab** (free T4 or Pro L4/A100). No local NVIDIA GPU
required.

---

## The one-paragraph version

Generating a token requires reading **every weight in the model** from memory.
That is about **1 FLOP per byte**, against a GPU that needs ~200 FLOPs per byte
to keep its tensor cores busy — so at batch size 1 a GPU runs at well under 1%
of its arithmetic capability. Every engine here is a different strategy for
fixing that: batch more requests per weight-read (continuous batching), store
fewer bytes per weight (quantization), or avoid recomputing bytes you already
have (prefix caching). Which strategy wins depends on your workload, not on the
engine's marketing.

## Quick start

```bash
git clone <this repo> && cd sglang-vllm-inference

# 1. Verify the harness (no GPU, no network, ~2 seconds)
python tests/test_bench.py
python tests/test_kernel_math.py

# 2. Regenerate the notebooks and stamp your repo slug into the site
REPO_SLUG=your-user/your-repo python tools/build_notebooks.py

# 3. Open notebooks/00_foundations_and_setup.ipynb in Colab and work forward.

# 4. After running notebooks, publish results to the site
python -m bench.aggregate
```

Or run everything checkable locally in one go:

```bash
make check      # tests + notebook build + site math + Python/JS parity
```

## The notebooks

Run them in order; each one saves results into `results/` using a shared schema
so notebook 07 can compare them honestly.

| # | Notebook | What you get |
|---|---|---|
| 00 | [Foundations](notebooks/00_foundations_and_setup.ipynb) | prefill vs decode, the roofline, KV-cache math, metric definitions, six ways benchmarks lie, and a plain `transformers` baseline |
| 01 | [vLLM](notebooks/01_vllm_deep_dive.ipynb) | continuous batching, PagedAttention, prefix caching measured by turning it off, `max_num_seqs` as a latency dial, open-loop overload |
| 02 | [SGLang](notebooks/02_sglang_deep_dive.ipynb) | RadixAttention on an agent workload — plus the control workload where it does nothing |
| 03 | [llama.cpp & GGUF](notebooks/03_llamacpp_gguf.ipynb) | GGUF format internals, K-quant block structure, size/speed/perplexity curve, the `-ngl` partial-offload curve |
| 04 | [TensorRT-LLM](notebooks/04_tensorrt_llm.ipynb) | ahead-of-time compilation, what the build buys and costs, FP8/INT8/INT4 and which phase each helps |
| 05 | [Triton Inference Server](notebooks/05_triton_inference_server.ipynb) | why it is not an engine; a dynamic batcher built from scratch and its latency/throughput curve |
| 06 | [Encoders vs decoders](notebooks/06_encoder_models_bert.ipynb) | BERT-family serving, ONNX Runtime vs eager vs `torch.compile`, the padding tax, which model family for which job |
| 07 | [The comparison](notebooks/07_comparison.ipynb) | every result on one set of axes, Pareto curves, cost per million tokens, the decision tree, stated limitations |
| 08 | [Kernel optimization](notebooks/08_kernel_optimization.ipynb) | how to decide *what* to optimize, then a fused int4 dequant-GEMV in Triton, validated and autotuned |

## The website

`docs/` is a GitHub Pages site: the concepts as diagrams, an interactive
roofline, an interactive KV-cache/concurrency calculator, live charts of your
own results, and a decision tree.

To publish: **Settings → Pages → Source: `main` branch, `/docs` folder.**

Locally:

```bash
cd docs && python -m http.server 8080    # then open http://localhost:8080
```

(Opening `docs/index.html` directly from disk works, but the browser blocks the
`fetch` of `data/results.json`, so the results section will show an explanation
instead of charts.)

## Why there is a shared harness

The `bench/` package exists so that "vLLM is N× faster than X" means something.

- **Every generative engine is driven through the same OpenAI-compatible
  `/v1/completions` endpoint with streaming on.** Not each engine's native
  Python API. Native APIs define "first token" differently and skip the
  scheduling and queueing a real server does, which makes them 20–40%
  optimistic. The HTTP overhead is a constant that cancels out of comparisons.
- **`ignore_eos=True` with a fixed `max_tokens`** everywhere, so you measure the
  engine rather than the model's verbosity.
- **Prompts are generated to exact token lengths using the target model's own
  tokenizer**, and the actual lengths are verified before every run.
- **Raw per-request records are saved**, not just summaries, so any percentile
  or any goodput SLO stays recomputable later without re-running anything.
- **Engine version, dtype and every flag that moves the number** are recorded in
  the result file. Prefix caching alone can swing a result 10×, so the flags are
  part of the measurement.
- **`validate()` runs after every benchmark** and warns about client-bound runs
  (via Little's Law), length confounds, and offered-vs-achieved rate gaps.

```
bench/
  schema.py      the result document every notebook writes
  metrics.py     TTFT / ITL / TPOT / goodput + the roofline & KV-cache models
  workloads.py   chat, summarize, rag, agent, longgen — exact token lengths
  client.py      one async streaming client for every engine
  runner.py      closed-loop and open-loop (Poisson) load, plus validate()
  servers.py     launch/teardown with log capture; per-engine argv builders
  plots.py       the house chart style (entity-stable colours, table twins)
  aggregate.py   results/ -> docs/data/results.json
  sysinfo.py     GPU capture, specs, and compute-capability gating
```

## Reading a result file

```jsonc
{
  "schema_version": "1.0",
  "engine":   { "engine": "vllm", "version": "0.32.0", "dtype": "float16",
                "flags": { "enable_prefix_caching": true, "max_num_seqs": 256 } },
  "workload": { "name": "rag", "input_len": 2048, "output_len": 128,
                "shared_prefix_tokens": 1792, "concurrency": 8, "mode": "closed" },
  "hardware": { "gpu_name": "Tesla T4", "gpu_compute_capability": "7.5" },
  "metrics":  { "ttft_s_p99": 0.41, "tpot_s_mean": 0.012,
                "output_throughput_tok_s": 1870, "goodput_req_s": 6.2 },
  "requests": [ /* raw per-request timelines */ ],
  "synthetic": false
}
```

`synthetic: true` marks a number that was **not** measured on this hardware. The
comparison notebook and the website both render those differently. No synthetic
number is ever presented as measured.

## Hardware notes

Compute capability gates real features, and a lot of tutorials silently assume
sm_80+:

| GPU | sm | bf16 | FlashAttn-2 | FP8 | INT4 AWQ/Marlin |
|---|---|---|---|---|---|
| T4 (Colab free) | 7.5 | no | no | no | no |
| L4 (Colab Pro) | 8.9 | yes | yes | **yes** | yes |
| A100 (Colab Pro+) | 8.0 | yes | yes | no | yes |
| H100 | 9.0 | yes | yes | **yes** | yes |

On a **free T4**, notebooks 00–03 and 05–08 all run. Notebook 04
(TensorRT-LLM) needs sm_80+ for its interesting paths and gates itself with an
explanation rather than failing halfway.

`bench.capability_report()` prints this for whatever GPU you were assigned —
worth running first, since Colab assigns the GPU and it changes between
sessions.

## Testing

Everything that can be verified without a GPU, is:

```bash
make check                          # all of the below

python tests/test_bench.py          # 21 tests: metrics, schema, validate, aggregate
python tests/test_kernel_math.py    #  7 tests: int4 packing + the Triton kernel's
                                    #           exact index arithmetic, in numpy
python tools/build_notebooks.py     # regenerates + syntax-checks every code cell
python tools/check_html.py          # site anchors, assets, SVG a11y, theme contract
node   tools/check_site.mjs         # the site's roofline / KV math
python tools/check_parity.py        # Python and JavaScript agree on every formula
node   tools/check_responsive.mjs   # no horizontal scroll at phone width (Chrome)
```

The parity check matters: the roofline and KV formulas exist in both
`bench/metrics.py` and `docs/assets/models.js`. Two copies of a formula drift,
and when they do the website starts contradicting the measurements next to it.

## Troubleshooting on Colab

**`ImportError: cannot import name '_chunk_or_narrow_cat' from 'torch._utils'`**
(or any error where torch fails to import its own internals, usually surfacing
from inside `transformers`)

This is **not** a version-pin problem, and changing transformers versions will
not fix it. torch is failing to import its own helper, which a self-consistent
release never does. `pip install vllm` pins an exact torch build
(`torch==2.13.0` for vLLM 0.32.0); Colab ships a different one; the downgrade
replaced some torch files and left others. Fix:

```bash
!pip uninstall -y -q torch torchvision torchaudio torchcodec
!pip install -q vllm
```

then **Runtime → Restart session** and run from the top.

Notebooks 01–04 call `install_engine()` and `print_torch_health()`, which
detect this at the install cell and print the remedy rather than letting it
surface later from a library that did nothing wrong.

**Always start each engine notebook on a fresh runtime.** vLLM, SGLang and
TensorRT-LLM pin different torch builds; installing one over another is how
you get the mixed install above. `Runtime → Disconnect and delete runtime`
between notebooks 01, 02 and 04.

**Why the engine notebooks never import transformers.** They drive a server
over HTTP and need a tokenizer only to build prompts of an exact token length.
`bench.load_tokenizer()` uses the Rust `tokenizers` package, which imports no
torch at all, so the client process cannot be broken by whatever the engine
did to torch. Notebooks 00 and 06 do load models and do use transformers —
they just do not install an engine.

## Editing the notebooks

**Edit `tools/nbsrc/*.py`, not `notebooks/*.ipynb`** — the `.ipynb` files are
generated and regenerating overwrites them. `.ipynb` is JSON with escaped
strings: painful to write prose in, impossible to review in a diff, and prone to
conflicts over execution counts and output blobs. The sources are plain Python
with the prose in triple-quoted strings.

```bash
python tools/build_notebooks.py
```

## What this project does not do

Stated up front, because a comparison that hides its limits is marketing:

- **One small model.** Qwen2.5-0.5B is bandwidth-bound and CPU-overhead
  sensitive, which flatters engines with lean schedulers. At 7B+ each forward
  pass is long enough that scheduling differences largely vanish. **Some
  rankings here plausibly invert at scale.**
- **One GPU, shared hardware.** Colab GPUs are thermally variable. Differences
  under ~10% are noise without repeats.
- **Short runs.** 64–96 requests per point makes p99 figures noisy. Trust the
  CDFs over the p99s.
- **Near-default configurations.** A specialist in any one of these engines
  could beat these numbers.
- **No multi-GPU, tensor parallelism, speculative decoding, or LoRA.** All of
  these change the picture.

Treat the *shapes* and the *reasoning* as the output. The absolute numbers are
yours to re-measure.

## License

MIT.
