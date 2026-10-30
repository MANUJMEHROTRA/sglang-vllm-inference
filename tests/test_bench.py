"""Tests for the benchmark harness. No GPU, no network, no engine required.

    python -m pytest tests/ -q        (or just: python tests/test_bench.py)

Run these before trusting a single number out of the notebooks. The harness is
the one piece of code every result depends on, so a silent bug here would
poison the whole repo rather than one experiment.
"""

from __future__ import annotations

import json
import math
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.aggregate import build, to_row
from bench.metrics import (
    ModelShape,
    arithmetic_intensity,
    goodput,
    littles_law_concurrency,
    percentile,
    roofline_decode_tok_s,
    summarize,
    tpot_of,
)
from bench.runner import validate
from bench.schema import (
    EngineSpec,
    Hardware,
    RequestRecord,
    RunResult,
    WorkloadSpec,
    load_dir,
)
from bench.sysinfo import capability_report, capture, gpu_specs_for
from bench.workloads import poisson_arrivals, preset

# Real config.json values, so the napkin math is anchored to real models.
QWEN_0_5B = ModelShape(
    n_layers=24, hidden_size=896, n_heads=14, n_kv_heads=2,
    intermediate_size=4864, vocab_size=151936, head_dim=64,
    tie_word_embeddings=True,
)
QWEN_1_5B = ModelShape(
    n_layers=28, hidden_size=1536, n_heads=12, n_kv_heads=2,
    intermediate_size=8960, vocab_size=151936, head_dim=128,
    tie_word_embeddings=True,
)


def test_percentile():
    vals = list(range(1, 101))
    assert abs(percentile(vals, 50) - 50.5) < 1e-9
    assert percentile(vals, 0) == 1
    assert percentile(vals, 100) == 100
    assert math.isnan(percentile([], 50))
    assert percentile([7.0], 99) == 7.0


def test_tpot_excludes_first_token():
    # 11 tokens: 1 prefill + 10 decode steps over 1.0s => 100ms each.
    r = RequestRecord(start_offset_s=0, ttft_s=0.5, e2e_s=1.5, output_tokens=11)
    assert abs(tpot_of(r) - 0.1) < 1e-9
    # A single-token response has no inter-token gap; folding in a 0 would
    # silently drag every aggregate down.
    assert tpot_of(RequestRecord(start_offset_s=0, ttft_s=0.5, e2e_s=0.5, output_tokens=1)) is None
    assert tpot_of(RequestRecord(start_offset_s=0, success=False)) is None


def _fake_run(n=50, out_tokens=64, seed=0):
    rng = random.Random(seed)
    records = []
    for i in range(n):
        ttft = 0.2 + rng.random() * 0.3
        itls = [0.01 + rng.random() * 0.01 for _ in range(out_tokens - 1)]
        records.append(
            RequestRecord(
                start_offset_s=i * 0.1, ttft_s=ttft, e2e_s=ttft + sum(itls), itls_s=itls,
                prompt_tokens=256, output_tokens=out_tokens, token_count_method="usage",
            )
        )
    return records


def test_summarize_separates_the_three_throughputs():
    records = _fake_run()
    records.append(RequestRecord(start_offset_s=5.0, success=False, error="boom"))
    m = summarize(records, duration_s=10.0)

    assert m["n_requests"] == 51 and m["n_success"] == 50 and m["n_failed"] == 1
    # Output throughput counts generated tokens only...
    assert abs(m["output_throughput_tok_s"] - 50 * 64 / 10.0) < 1e-6
    # ...while total throughput adds the prompt, and is 5x larger here. This
    # gap is exactly how a prefill-heavy benchmark inflates its headline.
    assert abs(m["total_throughput_tok_s"] - 50 * (256 + 64) / 10.0) < 1e-6
    assert m["total_throughput_tok_s"] / m["output_throughput_tok_s"] == 5.0
    # ITLs are pooled across requests, not averaged per request first.
    assert m["itl_s_n"] == 50 * 63
    assert m["ttft_s_p50"] <= m["ttft_s_p99"] <= m["ttft_s_max"]


def test_goodput_recomputable_under_a_new_slo():
    """The reason raw records live in the schema."""
    records = _fake_run()
    tight = goodput(records, 10.0, {"ttft_s": 0.25})
    loose = goodput(records, 10.0, {"ttft_s": 10.0})
    assert loose["n_good"] == 50
    assert 0 < tight["n_good"] < 50
    # An SLO on the per-request ITL tail, not just on TTFT.
    stall = goodput(records, 10.0, {"p99_itl_s": 0.001})
    assert stall["n_good"] == 0


def test_littles_law():
    assert abs(littles_law_concurrency(20.0, 2.0) - 40.0) < 1e-9


def test_param_counts_match_published_models():
    """Within 1% of the real published parameter counts."""
    assert abs(QWEN_0_5B.params() - 494_032_768) / 494_032_768 < 0.01
    assert abs(QWEN_1_5B.params() - 1_543_714_304) / 1_543_714_304 < 0.01
    # Untying the embeddings must add exactly one vocab x hidden matrix.
    untied = ModelShape(**{**QWEN_0_5B.__dict__, "tie_word_embeddings": False})
    assert untied.params() - QWEN_0_5B.params() == 151936 * 896


def test_kv_cache_math_and_gqa_saving():
    """GQA is the reason a 0.5B model serves hundreds of concurrent users."""
    kv = QWEN_0_5B.kv_cache_bytes(1024)
    assert kv == 2 * 24 * 2 * 64 * 1024 * 2        # 2(K,V) x L x kv_heads x d_h x T x fp16
    assert abs(kv / 1024**2 - 12.0) < 0.1          # 12 MB per 1k-token sequence

    # Without GQA (14 KV heads instead of 2) the same cache is 7x bigger,
    # which is 7x fewer concurrent sequences in the same HBM.
    mha = ModelShape(**{**QWEN_0_5B.__dict__, "n_kv_heads": 14})
    assert mha.kv_cache_bytes(1024) / kv == 7.0


def test_decode_is_memory_bound_at_batch_1():
    """The single most important fact about LLM inference."""
    p = QWEN_0_5B.params()
    ai = arithmetic_intensity(2 * p, p * 2)        # 2 FLOP/param, 2 bytes/param
    assert abs(ai - 1.0) < 0.01
    # A T4's ridge point is 65 TFLOP/s / 320 GB/s = ~203 FLOP/byte. At ~1,
    # decode uses well under 1% of the tensor cores and is bandwidth-starved.
    t4_ridge = 65e12 / 320e9
    assert ai / t4_ridge < 0.01


def test_roofline_batching_improves_throughput_sublinearly():
    """Weights are read once per step regardless of batch; KV is per sequence."""
    bs1 = roofline_decode_tok_s(QWEN_0_5B, 320.0, batch=1, seq_len=1024)
    bs32 = roofline_decode_tok_s(QWEN_0_5B, 320.0, batch=32, seq_len=1024)
    assert bs32 > bs1 * 10            # batching is a huge win...
    assert bs32 < bs1 * 32            # ...but not linear, because KV grows


def test_schema_round_trip_and_version_gate():
    records = _fake_run()
    res = RunResult(
        engine=EngineSpec(engine="vllm", version="0.32.0", model_id="Qwen/Qwen2.5-0.5B",
                          params_b=0.49, flags={"enable_prefix_caching": True}),
        workload=WorkloadSpec(name="chat", concurrency=8, n_requests=50),
        hardware=Hardware(gpu_name="Tesla T4", gpu_compute_capability="7.5", usd_per_hour=0.35),
        metrics=summarize(records, 10.0), requests=records, duration_s=10.0,
    )
    with tempfile.TemporaryDirectory() as td:
        path = res.save(td)
        back = RunResult.load(path)
        assert back.run_id == res.run_id
        assert len(back.requests) == 50
        assert back.engine.flags["enable_prefix_caching"] is True

        # A stale schema version is skipped, never silently plotted.
        bad = json.loads(path.read_text())
        bad["schema_version"] = "0.1"
        (Path(td) / "bad.json").write_text(json.dumps(bad))
        assert len(load_dir(td)) == 1


def test_validate_catches_a_client_bound_run():
    records = _fake_run()
    res = RunResult(
        engine=EngineSpec(engine="vllm", version="0.32.0", model_id="Qwen/Qwen2.5-0.5B"),
        workload=WorkloadSpec(name="chat", concurrency=64, n_requests=50, output_len=64),
        hardware=Hardware(), metrics=summarize(records, 10.0), requests=records, duration_s=10.0,
    )
    assert any("Little's Law" in w for w in validate(res))

    res.workload.concurrency = 8
    res.workload.output_len = 999          # vs 64 actually generated
    assert any("ignore_eos" in w for w in validate(res))


def test_cost_per_mtok():
    records = _fake_run()
    res = RunResult(
        engine=EngineSpec(engine="vllm", version="0.32.0", model_id="Qwen/Qwen2.5-0.5B"),
        workload=WorkloadSpec(name="chat", concurrency=8, n_requests=50),
        hardware=Hardware(gpu_name="Tesla T4", usd_per_hour=0.36),
        metrics=summarize(records, 10.0), requests=records, duration_s=10.0,
    )
    row = to_row(res)
    # 320 tok/s at $0.36/h = $0.0001/s -> 1e6 tokens takes 3125s -> $0.3125
    assert abs(row["usd_per_mtok"] - 0.3125) < 1e-6


def test_poisson_arrivals_are_bursty():
    offsets = poisson_arrivals(2000, rate=10.0, seed=1)
    assert offsets[0] == 0.0
    assert all(b >= a for a, b in zip(offsets, offsets[1:]))
    mean_gap = offsets[-1] / (len(offsets) - 1)
    assert abs(mean_gap - 0.1) < 0.02
    # Exponential gaps have stdev ~= mean; uniform spacing would give ~0.
    gaps = [b - a for a, b in zip(offsets, offsets[1:])]
    mean = sum(gaps) / len(gaps)
    stdev = (sum((g - mean) ** 2 for g in gaps) / len(gaps)) ** 0.5
    assert stdev > 0.6 * mean


def test_presets_are_distinguishable():
    chat, summ, rag = preset("chat"), preset("summarize"), preset("rag")
    assert summ.input_len > 8 * chat.input_len      # prefill-heavy
    assert chat.output_len > 2 * summ.output_len    # decode-heavy
    assert rag.shared_prefix_tokens / rag.input_len > 0.8   # cache-friendly
    assert preset("chat", concurrency=16).concurrency == 16


class _FakeFastTokenizer:
    """Stands in for tokenizers.Tokenizer: encode() returns an object with .ids"""

    __module__ = "tokenizers"

    class _Enc:
        def __init__(self, ids):
            self.ids = ids

    def encode(self, text, add_special_tokens=False):
        return self._Enc([ord(c) for c in text])

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


class _FakeHFTokenizer:
    """Stands in for a transformers tokenizer: encode() returns a list."""

    __module__ = "transformers.tokenization_utils"

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


def test_prompt_builder_accepts_either_tokenizer_backend():
    """The engine notebooks must not need transformers (and so torch) to
    count tokens; notebooks 00/06 already hold a transformers tokenizer.
    Both have to work through the same adapter."""
    from bench.tokenizer import as_tokenizer  # noqa: PLC0415
    from bench.workloads import PromptBuilder  # noqa: PLC0415

    for raw in (_FakeFastTokenizer(), _FakeHFTokenizer()):
        adapted = as_tokenizer(raw)
        assert adapted.encode("abc") == [97, 98, 99]
        assert adapted.decode([97, 98, 99]) == "abc"

        builder = PromptBuilder(raw, seed=0)
        # Exact length is the whole point of the builder.
        text = builder.text_of_tokens(32, random.Random(0))
        assert len(adapted.encode(text)) == 32

    # Wrapping an already-wrapped adapter must be a no-op, not double-wrapping.
    once = as_tokenizer(_FakeFastTokenizer())
    assert as_tokenizer(once) is once


def test_prompt_builder_builds_exact_shared_prefixes():
    """A prefix that differs by one byte defeats a radix cache completely."""
    from bench.workloads import PromptBuilder  # noqa: PLC0415

    builder = PromptBuilder(_FakeFastTokenizer(), seed=0)
    spec = preset("rag", n_requests=8, concurrency=4)
    prompts = builder.build(spec)

    assert len(prompts) == 8
    assert len({p[: spec.shared_prefix_tokens] for p in prompts}) == 1
    mean, lo, hi = builder.verify(prompts, spec)
    assert lo == hi == spec.input_len, f"lengths drifted: {lo}..{hi}"


def test_torch_health_reports_cleanly_without_torch():
    """On a machine with no torch this must report, not raise."""
    from bench.sysinfo import torch_health  # noqa: PLC0415

    health = torch_health()
    assert set(health) == {"installed_version", "importable", "self_consistent", "error"}
    if not health["importable"]:
        assert health["error"] and not health["self_consistent"]


def test_repo_slug_parsing():
    """Colab badges 404 silently when this is wrong, so pin both remote forms."""
    from tools.nbgen import slug_from_url  # noqa: PLC0415

    expected = "MANUJMEHROTRA/sglang-vllm-inference"
    for url in [
        "https://github.com/MANUJMEHROTRA/sglang-vllm-inference.git",
        "https://github.com/MANUJMEHROTRA/sglang-vllm-inference",
        "git@github.com:MANUJMEHROTRA/sglang-vllm-inference.git",
        "ssh://git@github.com/MANUJMEHROTRA/sglang-vllm-inference.git",
        "  https://github.com/MANUJMEHROTRA/sglang-vllm-inference.git\n",
    ]:
        assert slug_from_url(url) == expected, f"{url!r} -> {slug_from_url(url)!r}"

    # A two-segment result only: never "github.com/owner/name".
    assert slug_from_url("https://github.com/a/b").count("/") == 1


def test_sysinfo_runs_without_a_gpu():
    hw = capture()
    caps = capability_report(hw)
    assert set(caps) >= {"has_gpu", "bf16", "fp8", "flash_attention_2"}
    assert gpu_specs_for("Tesla T4")["sm"] == 75
    assert gpu_specs_for("NVIDIA A100-SXM4-80GB")["mem_gb"] == 80.0   # longest match
    assert gpu_specs_for(None) == {}


def test_aggregate_handles_an_empty_results_dir():
    """The very first run, before any notebook has been executed.

    The site must get a valid empty payload so it can show its own
    "run the notebooks" state instead of a fetch error.
    """
    with tempfile.TemporaryDirectory() as td:
        empty = Path(td) / "results"
        empty.mkdir()
        out = Path(td) / "site.json"
        payload = build(empty, out)
        written = json.loads(out.read_text())

    assert payload["counts"]["total"] == 0
    assert payload["runs"] == []
    assert payload["facets"] == {}
    assert written["schema_version"]            # still a valid document


def test_aggregate_builds_site_payload():
    records = _fake_run()
    with tempfile.TemporaryDirectory() as td:
        results_dir = Path(td) / "results"
        for i, engine in enumerate(["vllm", "sglang"]):
            RunResult(
                engine=EngineSpec(engine=engine, version="1.0", model_id="Qwen/Qwen2.5-0.5B"),
                workload=WorkloadSpec(name="chat", concurrency=2**i, n_requests=50),
                hardware=Hardware(gpu_name="Tesla T4", usd_per_hour=0.35),
                metrics=summarize(records, 10.0), requests=records, duration_s=10.0,
            ).save(results_dir)
        out = Path(td) / "site.json"
        payload = build(results_dir, out)
        written = json.loads(out.read_text())

    assert payload["counts"]["total"] == 2
    assert payload["facets"]["engine"] == {"vllm": 1, "sglang": 1}
    # Raw per-request records are dropped from the published file; the full
    # records stay in results/ so any percentile stays recomputable.
    assert "requests" not in payload["runs"][0]
    assert written["schema_version"] == payload["schema_version"]
    assert len(written["runs"]) == 2


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ok  {name}")
            passed += 1
    print(f"\n{passed} tests passed")
