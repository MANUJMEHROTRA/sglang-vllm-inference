"""Notebook 02 - SGLang: RadixAttention and the structured-generation runtime."""

from tools.nbgen import code, md, setup_cells

NAME = "02_sglang_deep_dive.ipynb"
TITLE = "02 - SGLang: RadixAttention, tree-shaped prefix reuse"

CELLS = setup_cells(
    NAME,
    TITLE,
    "SGLang takes the one idea vLLM uses a lookup table for -- sharing KV blocks "
    "between requests with a common prefix -- and makes it the organising principle "
    "of the whole scheduler. We measure where that pays and where it does not.",
    "~35 minutes",
) + [

    md(r"""
## RadixAttention: prefix sharing as a data structure

vLLM's prefix cache is a hash table: *does this block of tokens already exist?*
That handles the common case (a shared system prompt) well.

SGLang keeps the KV cache in a **radix tree** (a compressed trie) keyed on token
sequences. The difference is not a micro-optimisation -- it changes which
workloads can share.

```
Three agent requests sharing a growing conversation:

  "You are an agent. Tools: search, calc."     <- system prompt, 40 tokens
            |
            +-- "User: what is 17*23?"                       <- turn 1
            |         |
            |         +-- "Assistant: calc(17*23)  Tool: 391"
            |                   |
            |                   +-- "Assistant: 391"          <- branch A
            |                   +-- "Assistant: The answer"    <- branch B (retry)
            |
            +-- "User: search for X"                           <- different turn 1

  Every node is a run of tokens whose KV blocks are computed ONCE.
  A new request walks down from the root and only pays for the tokens
  below its deepest matching node.
```

What this buys you over a flat hash table:

- **Partial and nested matches.** A request sharing 300 of 400 prefix tokens
  reuses those 300, rather than needing a whole block-aligned exact match.
- **Branching.** Agent retries, beam search, self-consistency sampling and
  tree-of-thought all fork from a common ancestor. The tree shares the trunk
  across every branch automatically.
- **LRU eviction that respects structure.** Leaves are evicted first, so the
  heavily shared trunk survives memory pressure -- which is precisely the part
  worth keeping.

The costs are real: tree bookkeeping on every request, and matching is a
token-prefix operation rather than a hash lookup. On a workload with **no
shared prefix** the tree is pure overhead, and you should expect SGLang and
vLLM to land within noise of each other. We will verify exactly that, because
"it depends on your workload" is only useful if you can say *how much*.

### The other half: a zero-overhead scheduler

SGLang overlaps CPU scheduling with GPU compute, so the scheduler builds the
next batch while the current one is still running. In notebook 00 we saw that
a decode step on a small model can be under a millisecond; at that scale, CPU
work that is not overlapped *is* the bottleneck. This is why SGLang's advantage
is often largest on small models at high concurrency, which is exactly the
regime this notebook runs in.
"""),

    md("""
> **Start this notebook on a fresh runtime.** SGLang and vLLM pin different
> torch builds, so installing one on top of the other leaves a mix of files
> from two versions and torch stops being able to import its own internals.
> `Runtime -> Disconnect and delete runtime` before running notebook 02 after
> notebook 01. The check below catches it either way.
"""),

    code('''
from bench import install_engine, print_torch_health, restart_runtime

TORCH_CHANGED = install_engine("sglang[all]", engine_name="'sglang[all]'")
'''),

    code('''
if not print_torch_health("'sglang[all]'"):
    raise SystemExit("torch install is inconsistent -- see the remedy above")

if TORCH_CHANGED:
    restart_runtime("torch was swapped by the SGLang install.")

from bench import package_versions
VERSIONS = package_versions("sglang", "torch", "transformers", "flashinfer-python")
for name, v in VERSIONS.items():
    print(f"  {name:<22} {v}")
'''),

    code('''
import matplotlib.pyplot as plt
from dataclasses import replace

from bench import (EndpointConfig, EngineSpec, PromptBuilder, benchmark,
                   describe, discover_model_id, free_gpu, kill_stale,
                   load_tokenizer, preset, probe, run, serve, sglang_cmd,
                   sweep_concurrency)
from bench.plots import PALETTE, cdf, grouped_bar, save, sweep_curve, table, use_theme
from bench.servers import print_log

use_theme()
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
PORT = 8000
MAX_LEN = 4096
MEM_FRACTION = 0.80 if (HW.gpu_memory_gb or 16) <= 16 else 0.88
DTYPE = "float16" if not CAPS["bf16"] else "auto"

# Rust `tokenizers`, not transformers -- see notebook 01 on why the client
# process should not import torch just to count tokens.
builder = PromptBuilder(load_tokenizer(MODEL_ID), seed=0)

kill_stale(); free_gpu()
'''),

    md("""
### Matching the vLLM configuration

For the comparison in notebook 07 to mean anything, the two servers must be
configured equivalently. The parameter names differ; the concepts do not:

| vLLM | SGLang | what it controls |
|---|---|---|
| `--gpu-memory-utilization` | `--mem-fraction-static` | size of the KV pool |
| `--max-model-len` | `--context-length` | KV reserved per sequence |
| `--max-num-seqs` | `--max-running-requests` | decode batch ceiling |
| `--enable-prefix-caching` | radix cache (default on) | prefix reuse |
| `--max-num-batched-tokens` | `--chunked-prefill-size` | prefill chunking |

One caveat on `mem_fraction_static`: SGLang's number is the fraction used for
*weights plus KV pool*, while vLLM's covers the whole allocation. They are
close enough for a fair comparison but not identical, so we record both values
in the result files rather than claiming they are the same knob.
"""),

    code('''
cmd = sglang_cmd(MODEL_ID, port=PORT, context_length=MAX_LEN,
                 mem_fraction_static=MEM_FRACTION, dtype=DTYPE,
                 max_running_requests=256)
print(" ".join(cmd))

server_ctx = serve(cmd, port=PORT, log_name="sglang.log", timeout_s=1200)
base_url = server_ctx.__enter__()
try:
    served = run(discover_model_id(base_url))
    cfg = EndpointConfig(base_url=base_url, model=served, engine="sglang")
    print("serving:", served)
    print(run(probe(cfg, "The capital of France is", max_tokens=8)))
except Exception:
    print_log(PORT, 40); raise
'''),

    code('''
engine_spec = EngineSpec(
    engine="sglang", version=VERSIONS["sglang"], model_id=MODEL_ID,
    model_family="qwen", params_b=0.49, dtype=DTYPE, max_model_len=MAX_LEN,
    flags={"radix_cache": True, "max_running_requests": 256,
           "mem_fraction_static": MEM_FRACTION, "api": "openai-server"},
)

chat = preset("chat", input_len=256, output_len=256, n_requests=64)
sweep = run(sweep_concurrency(cfg, engine_spec, chat, builder,
                              levels=(1, 2, 4, 8, 16, 32, 64), hardware=HW,
                              notes="SGLang baseline concurrency sweep"))
for r in sweep:
    r.save("results/real")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
sweep_curve(sweep, "output_throughput_tok_s", ylabel="output tok/s",
            title="Throughput", ax=axes[0])
sweep_curve(sweep, "ttft_s_p99", scale=1000, ylabel="p99 TTFT (ms)",
            title="Tail latency", ax=axes[1])
sweep_curve(sweep, "goodput_req_s", ylabel="goodput (req/s)",
            title="Goodput", ax=axes[2])
plt.tight_layout(); save(fig, "02_sglang_sweep"); plt.show()
table(sweep)
'''),

    md("""
## Experiment 1 - The agent workload: where the radix tree earns its name

The `agent` preset models a tool-using loop: a long system prompt plus a
growing conversation, with each request sharing ~87% of its tokens with its
siblings. This is the shape RadixAttention was designed for.

We run it three ways:
1. radix cache **on** (SGLang's default),
2. radix cache **off** (`--disable-radix-cache`),
3. the same workload against vLLM with prefix caching on, from notebook 01.

(1) vs (2) isolates the mechanism. (1) vs (3) is the honest engine comparison,
both caching, which is the number worth quoting.
"""),

    code('''
agent = preset("agent", n_requests=96, concurrency=16)
print(describe(agent))

prompts = builder.build(agent)
mean_len, lo, hi = builder.verify(prompts, agent)
print(f"actual prompt tokens: mean {mean_len:.0f} (min {lo}, max {hi})")
print(f"shared prefix: {agent.shared_prefix_tokens} tokens "
      f"({100*agent.shared_prefix_tokens/mean_len:.0f}% of each prompt)")

radix_on = run(benchmark(cfg, engine_spec, agent, prompts, hardware=HW,
                         notes="agent workload, radix cache ON"))
radix_on.save("results/real")
'''),

    code('''
server_ctx.__exit__(None, None, None); free_gpu()

cmd_off = sglang_cmd(MODEL_ID, port=PORT, context_length=MAX_LEN,
                     mem_fraction_static=MEM_FRACTION, dtype=DTYPE,
                     max_running_requests=256, disable_radix_cache=True)
server_ctx = serve(cmd_off, port=PORT, log_name="sglang-no-radix.log", timeout_s=1200)
base_url = server_ctx.__enter__()
cfg = EndpointConfig(base_url=base_url, model=run(discover_model_id(base_url)),
                     engine="sglang")

spec_off = EngineSpec(**{**engine_spec.__dict__,
                         "flags": {**engine_spec.flags, "radix_cache": False}})
radix_off = run(benchmark(cfg, spec_off, agent, prompts, hardware=HW,
                          notes="agent workload, radix cache OFF"))
radix_off.save("results/real")
'''),

    code('''
on, off = radix_on.metrics, radix_off.metrics

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6))
grouped_bar(
    ["TTFT p50 (ms)", "TTFT p99 (ms)", "TPOT mean (ms)"],
    {"radix OFF": [1000*off["ttft_s_p50"], 1000*off["ttft_s_p99"], 1000*off["tpot_s_mean"]],
     "radix ON":  [1000*on["ttft_s_p50"],  1000*on["ttft_s_p99"],  1000*on["tpot_s_mean"]]},
    ylabel="milliseconds", title="RadixAttention on an 87%-shared agent workload",
    colors={"radix OFF": "#898781", "radix ON": PALETTE["orange"]}, ax=ax1)
cdf([radix_off, radix_on], metric="ttft", title="TTFT distribution", ax=ax2)
ax2.get_legend().set_visible(False)
plt.tight_layout(); save(fig, "02_radix_attention"); plt.show()

print(f"TTFT p50:   {1000*off['ttft_s_p50']:8.1f} -> {1000*on['ttft_s_p50']:8.1f} ms  "
      f"({off['ttft_s_p50']/on['ttft_s_p50']:.2f}x)")
print(f"throughput: {off['output_throughput_tok_s']:8.0f} -> "
      f"{on['output_throughput_tok_s']:8.0f} tok/s  "
      f"({on['output_throughput_tok_s']/off['output_throughput_tok_s']:.2f}x)")
print(f"TPOT:       {1000*off['tpot_s_mean']:8.2f} -> {1000*on['tpot_s_mean']:8.2f} ms  "
      f"(expect ~no change -- caching does not touch decode)")
'''),

    md("""
## Experiment 2 - The control: a workload with nothing to share

This is the experiment most comparisons skip, and it is the one that tells you
whether you are reading an architecture result or a workload result.

The `chat` preset has no shared prefix. Prediction: radix on and radix off land
within a few percent of each other, with *off* possibly a hair faster because
the tree bookkeeping is pure cost when nothing matches.

If you see a large difference here, something is wrong with the setup -- check
that the prompts really are distinct.
"""),

    code('''
control = preset("chat", input_len=512, output_len=128, n_requests=64, concurrency=16)
control_prompts = builder.build(control)
assert len({p[:200] for p in control_prompts}) == len(control_prompts), \\
    "control workload should have NO shared prefix"

ctrl_off = run(benchmark(cfg, spec_off, control, control_prompts, hardware=HW,
                         notes="control (no shared prefix), radix OFF"))
ctrl_off.save("results/real")

server_ctx.__exit__(None, None, None); free_gpu()
server_ctx = serve(cmd, port=PORT, log_name="sglang.log", timeout_s=1200)
base_url = server_ctx.__enter__()
cfg = EndpointConfig(base_url=base_url, model=run(discover_model_id(base_url)),
                     engine="sglang")

ctrl_on = run(benchmark(cfg, engine_spec, control, control_prompts, hardware=HW,
                        notes="control (no shared prefix), radix ON"))
ctrl_on.save("results/real")
'''),

    code('''
fig, ax = plt.subplots(figsize=(9, 4.4))
grouped_bar(
    ["agent (87% shared)", "chat (0% shared)"],
    {"radix OFF": [1000*radix_off.metrics["ttft_s_p50"], 1000*ctrl_off.metrics["ttft_s_p50"]],
     "radix ON":  [1000*radix_on.metrics["ttft_s_p50"],  1000*ctrl_on.metrics["ttft_s_p50"]]},
    ylabel="TTFT p50 (ms)",
    title="Prefix reuse is a property of the workload, not of the engine",
    colors={"radix OFF": "#898781", "radix ON": PALETTE["orange"]}, ax=ax)
save(fig, "02_radix_control"); plt.show()

agent_gain = radix_off.metrics["ttft_s_p50"] / radix_on.metrics["ttft_s_p50"]
chat_gain  = ctrl_off.metrics["ttft_s_p50"] / ctrl_on.metrics["ttft_s_p50"]
print(f"  agent workload: {agent_gain:.2f}x TTFT improvement from the radix cache")
print(f"  chat workload:  {chat_gain:.2f}x  <- should be ~1.0")
print()
print("This pair of bars is the most important result in the notebook. Any")
print("claim of the form 'engine X is N times faster' that does not state the")
print("workload shape is unfalsifiable.")
'''),

    md("""
## Experiment 3 - Structured output: constrained decoding

SGLang ships a fast grammar backend (xgrammar) for forcing output to match a
JSON schema or regex. The mechanism: at each decode step, mask the logits of
every token that cannot legally come next, then sample from what remains.

The interesting question is the **cost**. A naive implementation recomputes the
allowed-token mask every step on the CPU and stalls the GPU -- in bad
implementations constrained decoding can cost 2-5x. A good one precompiles the
grammar into a pushdown automaton and overlaps mask computation with the
forward pass, bringing overhead close to zero.

We measure it: same prompts, same output length, with and without a schema.
"""),

    code('''
import json

schema = {
    "type": "object",
    "properties": {
        "name":      {"type": "string"},
        "latency_ms": {"type": "number"},
        "engine":    {"type": "string", "enum": ["vllm", "sglang", "llamacpp"]},
        "notes":     {"type": "string"},
    },
    "required": ["name", "latency_ms", "engine", "notes"],
}

json_prompts = [
    "Produce a JSON object describing an inference benchmark result. "
    f"Variant {i}. JSON:" for i in range(48)
]
w = replace(preset("chat", output_len=64), n_requests=len(json_prompts), concurrency=8)

free_spec = EngineSpec(**{**engine_spec.__dict__,
                          "flags": {**engine_spec.flags, "structured_output": None}})
free_run = run(benchmark(cfg, free_spec, w, json_prompts, hardware=HW,
                         notes="unconstrained generation"))
free_run.save("results/real")

# Constrained: the schema goes in extra_body, which our client forwards verbatim.
cfg_json = EndpointConfig(base_url=base_url, model=cfg.model, engine="sglang",
                          extra_body={"response_format": {
                              "type": "json_schema",
                              "json_schema": {"name": "result", "schema": schema}}})
grammar_spec = EngineSpec(**{**engine_spec.__dict__,
                             "flags": {**engine_spec.flags,
                                       "structured_output": "json_schema (xgrammar)"}})
json_run = run(benchmark(cfg_json, grammar_spec, w, json_prompts, hardware=HW,
                         notes="constrained to a JSON schema"))
json_run.save("results/real")

a, b = free_run.metrics, json_run.metrics
print(f"\\n  unconstrained TPOT {1000*a['tpot_s_mean']:6.2f} ms   "
      f"{a['output_throughput_tok_s']:6.0f} tok/s")
print(f"  JSON-constrained   {1000*b['tpot_s_mean']:6.2f} ms   "
      f"{b['output_throughput_tok_s']:6.0f} tok/s")
print(f"  overhead: {100*(b['tpot_s_mean']/a['tpot_s_mean'] - 1):+.1f}% per token")
print()
print("Under ~10% means the grammar is overlapped with the forward pass.")
print("Much more than that means mask computation is stalling the GPU.")
'''),

    md("""
### Why this matters more than it looks

Constrained decoding is not just a convenience. It changes the *reliability
engineering* of a system: with a schema enforced at the sampler, a malformed
response becomes structurally impossible rather than something you retry on.
If the overhead is near zero, you can stop writing parse-and-retry loops
entirely -- and retries are usually far more expensive than the grammar.

vLLM has equivalent support (also via xgrammar or outlines); TensorRT-LLM has
it too. The capability is not a differentiator any more, but the *overhead*
still varies between engines, so measure it on your own schema. Deeply nested
schemas with many alternations are much harder on the grammar engine than the
flat one used here.
"""),

    code('''
server_ctx.__exit__(None, None, None)
kill_stale(); free_gpu()
'''),

    md("""
## vLLM or SGLang?

They have converged a great deal. Both do paged KV, continuous batching, prefix
caching, chunked prefill, CUDA graphs, structured output, speculative decoding
and tensor parallelism. Choosing between them on a generic chat workload is
mostly a coin toss, and anyone claiming a large general-purpose gap is quoting
a workload, not an architecture.

**Pick SGLang when** your prompts have *tree* structure -- agents that branch
and retry, multi-turn sessions with a growing shared history, self-consistency
sampling, beam-like search. The radix tree handles partial and nested sharing
that a block-hash cache misses. Also when you want its frontend DSL to express
multi-call programs server-side, keeping the KV cache hot across the whole
program rather than between separate HTTP calls.

**Pick vLLM when** you want the broadest model and feature coverage and the
largest ecosystem. It tends to support new architectures first, has more
third-party integrations, and is the one most deployment tooling assumes.

**Measure, do not assume.** Run your own prompt distribution through both. The
numbers you just produced are specific to a 0.5B model on your GPU at these
concurrencies; at 70B on an H100 the balance shifts, because CPU scheduling
overhead stops mattering when each forward pass takes 30 ms.

### Exercises

1. Build a workload where requests share a prefix *with each other* but in a
   branching pattern (turn 1 shared by 8 requests, turn 2 shared by 2). Predict
   which engine wins, then check.
2. Run the agent workload at concurrency 1. Does the radix cache still help?
   (It should -- sequential requests still hit the tree -- which tells you the
   mechanism is reuse across *time*, not just across concurrent batches.)
3. Use `--chunked-prefill-size 512` and run a mixed workload. Compare p99 ITL
   to the default, as in notebook 01's exercise 4.
4. Nest the JSON schema three levels deep with several `anyOf` branches, and
   re-measure grammar overhead. Where does it stop being free?

**Next:** [03 - llama.cpp and GGUF](03_llamacpp_gguf.ipynb), where we leave the
GPU-only world entirely.
"""),
]
