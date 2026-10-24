"""Check that the website's JavaScript and the Python harness agree.

    python tools/check_parity.py

The same formulas exist twice in this repo: in ``bench/metrics.py`` (used by
the notebooks) and in ``docs/assets/models.js`` (used by the live site). Two
implementations of one formula will drift, and when they do the site starts
quietly contradicting the measurements it sits next to.

So: ask node for the JavaScript's answers, compute the Python answers, and
compare them exactly. Requires node on PATH; skips cleanly without it.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from bench.metrics import ModelShape, roofline_decode_tok_s  # noqa: E402
from bench.sysinfo import GPU_SPECS  # noqa: E402

# Must mirror docs/assets/models.js exactly.
MODELS = {
    "qwen2.5-0.5b": ModelShape(24, 896, 14, 2, 4864, 151936, 64, True),
    "qwen2.5-1.5b": ModelShape(28, 1536, 12, 2, 8960, 151936, 128, True),
    "qwen2.5-7b": ModelShape(28, 3584, 28, 4, 18944, 152064, 128, False),
    "llama3.2-1b": ModelShape(16, 2048, 32, 8, 8192, 128256, 64, True),
    "llama3.1-8b": ModelShape(32, 4096, 32, 8, 14336, 128256, 128, False),
}

CONTEXTS = [512, 1024, 2048, 4096, 8192, 16384, 32768]
BATCHES = [1, 2, 4, 8, 16, 32, 64, 128, 256]

# The JS-side script. Dumps every value we want to compare as JSON.
DUMP_JS = """
import { MODELS, GPUS, params, kvBytes, prefillFlops, ridgePoint,
         decodeCeilingTokS } from "./docs/assets/models.js";

const out = { params: {}, kv: {}, prefill: {}, ridge: {}, ceiling: {}, gpus: {} };
for (const [k, m] of Object.entries(MODELS)) {
  out.params[k] = params(m);
  out.kv[k] = {};
  out.prefill[k] = {};
  for (const ctx of [512, 1024, 2048, 4096, 8192, 16384, 32768]) {
    out.kv[k][ctx] = kvBytes(m, ctx, 2);
    out.prefill[k][ctx] = prefillFlops(m, ctx);
  }
}
for (const [k, g] of Object.entries(GPUS)) {
  out.ridge[k] = ridgePoint(g);
  out.gpus[k] = { memGb: g.memGb, bwGbs: g.bwGbs, tflops: g.tflops };
}
const m = MODELS["qwen2.5-0.5b"];
for (const bs of [1, 2, 4, 8, 16, 32, 64, 128, 256]) {
  out.ceiling[bs] = decodeCeilingTokS(m, GPUS.t4, bs, 1024);
}
console.log(JSON.stringify(out));
"""


def js_values() -> dict:
    script = REPO_ROOT / ".parity_dump.mjs"
    script.write_text(DUMP_JS)
    try:
        raw = subprocess.check_output(["node", str(script)], cwd=REPO_ROOT, text=True)
    finally:
        script.unlink(missing_ok=True)
    return json.loads(raw)


def main() -> int:
    if not shutil.which("node"):
        print("node not found on PATH -- skipping JS/Python parity check.")
        print("Install node to run it; tools/check_site.mjs needs it too.")
        return 0

    js = js_values()
    failures = 0

    def cmp(name: str, py, j, tol: float = 0.0) -> None:
        nonlocal failures
        ok = (py == j) if tol == 0 else (abs(py - j) <= tol * max(abs(py), 1))
        if ok:
            print(f"  ok  {name}")
        else:
            print(f"  FAIL {name}: python={py!r} js={j!r}")
            failures += 1

    print("parameter counts")
    for key, shape in MODELS.items():
        cmp(f"params[{key}]", shape.params(), js["params"][key])

    print("\nKV cache bytes (all contexts)")
    for key, shape in MODELS.items():
        mismatch = [
            c for c in CONTEXTS
            if shape.kv_cache_bytes(c, dtype_bytes=2) != js["kv"][key][str(c)]
        ]
        cmp(f"kv_cache_bytes[{key}] across {len(CONTEXTS)} contexts",
            [], mismatch)

    print("\nprefill FLOPs (all contexts)")
    for key, shape in MODELS.items():
        mismatch = [
            c for c in CONTEXTS
            if abs(shape.prefill_flops(c) - js["prefill"][key][str(c)])
            > 1e-6 * shape.prefill_flops(c)
        ]
        cmp(f"prefill_flops[{key}] across {len(CONTEXTS)} contexts", [], mismatch)

    print("\nGPU specs vs bench/sysinfo.py")
    # The site's keys map onto sysinfo's substring keys.
    spec_map = {"t4": "T4", "l4": "L4", "a100_40": "A100",
                "a100_80": "A100-SXM4-80", "h100": "H100", "rtx4090": "RTX 4090"}
    for site_key, spec_key in spec_map.items():
        spec = GPU_SPECS[spec_key]
        got = js["gpus"][site_key]
        cmp(f"{site_key} bandwidth", spec["hbm_gb_s"], got["bwGbs"])
        cmp(f"{site_key} fp16 TFLOPS", spec["fp16_tflops"], got["tflops"])
        cmp(f"{site_key} memory GB", spec["mem_gb"], got["memGb"])

    print("\nridge points")
    for site_key, spec_key in spec_map.items():
        spec = GPU_SPECS[spec_key]
        py_ridge = spec["fp16_tflops"] * 1e12 / (spec["hbm_gb_s"] * 1e9)
        cmp(f"ridge[{site_key}]", py_ridge, js["ridge"][site_key], tol=1e-9)

    print("\ndecode ceiling on a T4 (Qwen2.5-0.5B, 1k ctx)")
    shape = MODELS["qwen2.5-0.5b"]
    for bs in BATCHES:
        py = roofline_decode_tok_s(shape, 320.0, dtype_bytes=2, batch=bs, seq_len=1024)
        cmp(f"ceiling[batch={bs}]", py, js["ceiling"][str(bs)], tol=1e-9)

    if failures:
        print(f"\n{failures} PARITY FAILURE(S) -- the site and the notebooks disagree.")
        return 1
    print("\nPython and JavaScript agree on every value.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
