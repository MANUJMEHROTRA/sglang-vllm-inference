"""Collapse every result file into the one JSON the website reads.

Run it after any benchmarking session:

    python -m bench.aggregate

It walks ``results/``, drops the raw per-request records (they are large and
the site only ever plots summaries), and writes ``docs/data/results.json``.
The GitHub Pages site reads that file at load time, so refreshing the site is
exactly "re-run the notebooks, re-run this, commit".

Keeping raw records out of the published file is a size decision, not a
transparency one: the full records stay in ``results/`` and travel with the
repo, so anyone can recompute a different percentile or a different SLO.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from .metrics import DEFAULT_SLO
from .schema import SCHEMA_VERSION, RunResult, load_dir

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
SITE_DATA = REPO_ROOT / "docs" / "data" / "results.json"

# Metrics the site charts. Keeping this list explicit means adding a chart is a
# deliberate act rather than "whatever happened to be in the dict".
SITE_METRICS = [
    "ttft_s_mean", "ttft_s_p50", "ttft_s_p90", "ttft_s_p99",
    "tpot_s_mean", "tpot_s_p99",
    "itl_s_p50", "itl_s_p99",
    "e2e_s_mean", "e2e_s_p99",
    "output_throughput_tok_s", "total_throughput_tok_s",
    "request_throughput_req_s",
    "goodput_req_s", "goodput_fraction",
    "n_requests", "n_success", "n_failed", "duration_s",
    "total_input_tokens", "total_output_tokens",
]


def to_row(r: RunResult) -> Dict[str, Any]:
    """One flat record per run -- the shape the site's JS expects."""
    row: Dict[str, Any] = {
        "run_id": r.run_id,
        "timestamp": r.timestamp,
        "synthetic": r.synthetic,
        "notes": r.notes,
        "engine": r.engine.engine,
        "engine_version": r.engine.version,
        "model_id": r.engine.model_id,
        "model_family": r.engine.model_family,
        "params_b": r.engine.params_b,
        "task": r.engine.task,
        "dtype": r.engine.dtype,
        "quantization": r.engine.quantization,
        "flags": r.engine.flags,
        "workload": r.workload.name,
        "mode": r.workload.mode,
        "concurrency": r.workload.concurrency,
        "request_rate": r.workload.request_rate,
        "input_len": r.workload.input_len,
        "output_len": r.workload.output_len,
        "shared_prefix_tokens": r.workload.shared_prefix_tokens,
        "gpu": r.hardware.gpu_name,
        "gpu_memory_gb": r.hardware.gpu_memory_gb,
        "compute_capability": r.hardware.gpu_compute_capability,
        "usd_per_hour": r.hardware.usd_per_hour,
    }
    for key in SITE_METRICS:
        row[key] = r.metrics.get(key)

    # Cost per million output tokens -- the number that decides deployments.
    rate = r.hardware.usd_per_hour
    tok_s = r.metrics.get("output_throughput_tok_s") or 0
    row["usd_per_mtok"] = (rate / 3600.0) / tok_s * 1e6 if rate and tok_s > 0 else None
    return row


def build(results_dir: Path | None = None, out: Path | None = None) -> Dict[str, Any]:
    results_dir = Path(results_dir or RESULTS_DIR)
    out = Path(out or SITE_DATA)

    runs = load_dir(results_dir)
    rows = [to_row(r) for r in runs]

    real = [r for r in rows if not r["synthetic"]]
    synthetic = [r for r in rows if r["synthetic"]]
    facets = _facets(rows)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.time(),
        "generated_at_iso": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "default_slo": DEFAULT_SLO,
        "counts": {
            "total": len(rows),
            "measured": len(real),
            "synthetic": len(synthetic),
        },
        "facets": facets,
        "runs": rows,
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1))

    print(f"{len(rows)} runs -> {out}")
    if not rows:
        # First run, before any notebook has been executed. Write the valid
        # empty payload anyway so the site loads and shows its own
        # "run the notebooks" state rather than a fetch error.
        print(f"  no result files found under {results_dir}/ yet -- "
              "run the notebooks, then re-run this.")
        return payload

    print(f"  measured: {len(real)}   illustrative/synthetic: {len(synthetic)}")
    if synthetic:
        print("  (the site marks synthetic rows with a dagger and a warning)")
    for engine, count in sorted(facets.get("engine", {}).items()):
        print(f"  {engine:<16} {count}")
    return payload


def _facets(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    """Counts per engine / workload / gpu / model, for the site's filter row."""
    facets: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        for field in ("engine", "workload", "gpu", "model_id", "dtype", "task"):
            value = row.get(field)
            if value:
                facets[field][str(value)] += 1
    return {k: dict(v) for k, v in facets.items()}


def markdown_table(runs: List[RunResult], limit: int = 40) -> str:
    """A README-ready table. Latencies in ms, throughput in tok/s."""
    header = (
        "| engine | model | dtype | workload | conc | TTFT p50 | TTFT p99 | "
        "TPOT mean | out tok/s | $/Mtok |\n"
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|"
    )
    lines = [header]
    for r in sorted(runs, key=lambda x: -(x.metrics.get("output_throughput_tok_s") or 0))[:limit]:
        m = r.metrics
        cost = to_row(r)["usd_per_mtok"]
        cost_str = f"{cost:.2f}" if cost else "-"
        mark = " *" if r.synthetic else ""
        lines.append(
            f"| {r.engine.engine}{mark} | {r.engine.model_id.split('/')[-1]} | "
            f"{r.engine.dtype} | {r.workload.name} | {r.workload.concurrency or '-'} | "
            f"{(m.get('ttft_s_p50') or 0) * 1000:.0f} ms | "
            f"{(m.get('ttft_s_p99') or 0) * 1000:.0f} ms | "
            f"{(m.get('tpot_s_mean') or 0) * 1000:.1f} ms | "
            f"{m.get('output_throughput_tok_s') or 0:.0f} | "
            f"{cost_str} |"
        )
    if any(r.synthetic for r in runs):
        lines.append("\n`*` = illustrative, not measured on this hardware.")
    return "\n".join(lines)


if __name__ == "__main__":
    build()
