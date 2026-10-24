# Results

Every notebook writes here. One JSON file per
**(engine, model, workload, hardware)** point.

```
results/
  real/      measured on your hardware by the notebooks
  demo/      anything illustrative (synthetic: true) -- never presented as measured
  figures/   PNGs saved by the notebooks (gitignored; regenerate by re-running)
```

## Filenames

```
<engine>__<model>__<workload>__<gpu>__<run_id>.json
vllm__Qwen2.5-0.5B-Instruct__chat__Tesla-T4__a35e7d4e7d56.json
```

The name encodes the identity of the point, so a directory listing is already a
readable index and two runs differing only in a flag cannot overwrite each
other.

## What is in a file

See `bench/schema.py` for the authoritative definition. The parts that matter:

- **`engine`** — engine name, **version**, model id, dtype, quantization, and
  **`flags`**: every setting that moves the number. A result without its flags
  is a rumour. `enable_prefix_caching` alone can swing TTFT by 10×.
- **`workload`** — the traffic shape, including `shared_prefix_tokens` and
  whether load was closed-loop (fixed concurrency) or open-loop (Poisson
  arrivals). Two shapes are two different experiments, never something to
  average.
- **`hardware`** — captured automatically, never typed by hand. Colab assigns
  the GPU and it changes between sessions.
- **`metrics`** — the summary, including all three throughputs kept separate
  (output / total / request) and goodput against the SLO recorded alongside it.
- **`requests`** — the **raw per-request timelines**: TTFT, every inter-token
  gap, token counts. Kept so that any percentile, or goodput under a *different*
  SLO, stays recomputable later without re-running anything.
- **`synthetic`** — `true` means this number was **not** measured on this
  hardware. Both the comparison notebook and the website render those
  differently and label them.

## Re-analysing without re-running

```python
from bench.schema import load_dir
from bench.metrics import goodput

runs = load_dir("results")

# A stricter SLO than the one the run was summarised with -- no re-running.
for r in runs:
    g = goodput(r.requests, r.duration_s, {"ttft_s": 0.3, "p99_itl_s": 0.03})
    print(f"{r.engine.engine:<14} c={r.workload.concurrency:<4} "
          f"{100 * g['goodput_fraction']:.0f}% met the tighter SLO")
```

## Schema versioning

`load_dir()` **skips files whose `schema_version` does not match** the current
one, printing the filename. That is deliberate: a stale file silently plotted
beside new ones is worse than a loud skip. If you see skips after upgrading,
re-run those benchmarks rather than editing the version field.

## Publishing to the website

```bash
python -m bench.aggregate     # -> docs/data/results.json
```

Raw per-request records are dropped from the published file (size), but stay
here in full.
