"""Turning a list of RequestRecords into the numbers people argue about.

Read this file once and the rest of the repo stops being mysterious. Every
metric below is defined here exactly once, and every notebook uses these
functions rather than recomputing a mean somewhere. Most "benchmark
disagreements" on the internet are definition disagreements, not measurement
disagreements.

The definitions
---------------

**TTFT** (Time To First Token): request sent -> first token byte received.
Dominated by *prefill*, the one forward pass over the whole prompt. Scales with
prompt length and with how long you waited in the queue behind other requests.
This is the "did it hang?" metric.

**ITL** (Inter-Token Latency): the gap between two consecutive tokens, after
the first. One sample per token, so a 128-token response gives 127 samples.
This is the "is it stuttering?" metric, and only its *tail* matters -- a p99 ITL
of 400ms is visible to a human even if the mean is 20ms.

**TPOT** (Time Per Output Token): ``(e2e - ttft) / (output_tokens - 1)``. The
mean ITL for one request, computed the cheap way. People use TPOT and "mean
ITL" interchangeably; they are the same number up to how you handle the first
token, which is why both are reported here.

**E2E latency**: sent -> last token. ``ttft + (output_tokens - 1) * tpot``.

**Throughput** comes in three flavours and mixing them up is the single most
common benchmarking error:
  - *output token throughput*: generated tokens/s across the whole run. The
    number that matters for cost.
  - *total token throughput*: (prompt + generated)/s. Flatters prefill-heavy
    workloads enormously -- a summarisation benchmark can show 10x the "tokens/s"
    of a chat benchmark while generating fewer tokens.
  - *request throughput*: completed requests/s. The number that matters for
    capacity planning.

**Goodput**: requests/s that met an SLO, e.g. "TTFT <= 1s AND p99 ITL <= 50ms".
This is the metric that actually tracks user experience, because a server can
post a magnificent throughput number by making everyone wait. Raw records are
kept in the schema specifically so goodput can be recomputed under a different
SLO later without re-running anything.

A note on percentiles
---------------------
Implemented here with linear interpolation (the same convention as numpy's
default ``linear`` method) and no numpy import, so results stay inspectable in
a bare environment. With n < ~100 samples a p99 is mostly noise; the summary
reports the sample count beside it so you can tell.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .schema import RequestRecord

# Default SLO used for goodput when a notebook does not override it. These are
# roughly "a chat UI that feels responsive": first token within a second,
# and no visible stall between tokens.
DEFAULT_SLO = {"ttft_s": 1.0, "p99_itl_s": 0.05}


def percentile(values: Sequence[float], q: float) -> float:
    """``q`` in [0, 100]. Linear interpolation between order statistics."""
    if not values:
        return float("nan")
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    pos = (len(s) - 1) * (q / 100.0)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return float(s[lo] * (1.0 - frac) + s[hi] * frac)


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def _dist(values: Sequence[float], prefix: str) -> Dict[str, float]:
    """mean/p50/p90/p99/max for one distribution, flattened into a dict."""
    return {
        f"{prefix}_mean": _mean(values),
        f"{prefix}_p50": percentile(values, 50),
        f"{prefix}_p90": percentile(values, 90),
        f"{prefix}_p99": percentile(values, 99),
        f"{prefix}_max": float(max(values)) if values else float("nan"),
        f"{prefix}_n": len(values),
    }


def tpot_of(record: RequestRecord) -> Optional[float]:
    """Time per output token for one request, excluding the first token.

    Returns None for a 1-token response: there is no *inter*-token gap to
    measure, and folding a 0 into the average would silently drag every
    aggregate down.
    """
    if not record.success or record.e2e_s is None or record.ttft_s is None:
        return None
    n = record.output_tokens or 0
    if n < 2:
        return None
    return (record.e2e_s - record.ttft_s) / (n - 1)


def summarize(
    records: Iterable[RequestRecord],
    duration_s: float,
    slo: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Collapse a run into the metric dict stored on ``RunResult.metrics``.

    ``duration_s`` is wall-clock for the whole run -- from the first request
    being sent to the last one completing. Throughput divides by this, so it
    must *not* include model load or warmup time.
    """
    records = list(records)
    ok = [r for r in records if r.success and r.ttft_s is not None]
    failed = len(records) - len(ok)

    ttfts = [r.ttft_s for r in ok if r.ttft_s is not None]
    e2es = [r.e2e_s for r in ok if r.e2e_s is not None]
    tpots = [t for t in (tpot_of(r) for r in ok) if t is not None]
    # Pooled across requests: this is the distribution a user experiences,
    # not the distribution of per-request averages.
    all_itls = [itl for r in ok for itl in r.itls_s]

    out_tokens = sum(r.output_tokens or 0 for r in ok)
    in_tokens = sum(r.prompt_tokens or 0 for r in ok)

    metrics: Dict[str, Any] = {
        "n_requests": len(records),
        "n_success": len(ok),
        "n_failed": failed,
        "duration_s": duration_s,
        "total_input_tokens": in_tokens,
        "total_output_tokens": out_tokens,
        # The three throughputs, never collapsed into one "tokens/s".
        "output_throughput_tok_s": out_tokens / duration_s if duration_s > 0 else float("nan"),
        "total_throughput_tok_s": (in_tokens + out_tokens) / duration_s
        if duration_s > 0
        else float("nan"),
        "request_throughput_req_s": len(ok) / duration_s if duration_s > 0 else float("nan"),
    }
    metrics.update(_dist(ttfts, "ttft_s"))
    metrics.update(_dist(e2es, "e2e_s"))
    metrics.update(_dist(tpots, "tpot_s"))
    metrics.update(_dist(all_itls, "itl_s"))

    slo = slo or DEFAULT_SLO
    metrics["slo"] = dict(slo)
    metrics.update(goodput(ok, duration_s, slo))
    return metrics


def goodput(
    records: Sequence[RequestRecord], duration_s: float, slo: Dict[str, float]
) -> Dict[str, Any]:
    """Requests/s that met every SLO key.

    Supported keys: ``ttft_s``, ``p99_itl_s``, ``e2e_s``, ``tpot_s`` -- each an
    upper bound. A request must satisfy *all* present keys to count.

    ``p99_itl_s`` is evaluated per request (the p99 of that one request's own
    ITL samples), which is the right question: "did this user see a stall?"
    """
    good = 0
    for r in records:
        if not r.success:
            continue
        if "ttft_s" in slo and (r.ttft_s is None or r.ttft_s > slo["ttft_s"]):
            continue
        if "e2e_s" in slo and (r.e2e_s is None or r.e2e_s > slo["e2e_s"]):
            continue
        if "tpot_s" in slo:
            t = tpot_of(r)
            if t is None or t > slo["tpot_s"]:
                continue
        if "p99_itl_s" in slo:
            if not r.itls_s or percentile(r.itls_s, 99) > slo["p99_itl_s"]:
                continue
        good += 1
    return {
        "goodput_req_s": good / duration_s if duration_s > 0 else float("nan"),
        "goodput_fraction": good / len(records) if records else float("nan"),
        "n_good": good,
    }


# ---------------------------------------------------------------------------
# First-principles models. These are not measurements -- they are the napkin
# math each notebook checks its measurements against. When a measurement and
# the model disagree by more than ~2x, one of them is wrong and finding out
# which is where the learning happens.
# ---------------------------------------------------------------------------


@dataclass
class ModelShape:
    """The handful of numbers that determine inference cost for a transformer.

    Pull these from ``config.json`` on the Hub -- every field maps to a key
    there (``num_hidden_layers``, ``hidden_size``, ``num_attention_heads``,
    ``num_key_value_heads``, ``intermediate_size``, ``vocab_size``).
    """

    n_layers: int
    hidden_size: int
    n_heads: int
    n_kv_heads: int
    intermediate_size: int
    vocab_size: int
    head_dim: Optional[int] = None

    # Small models very often tie the input embedding and the output head --
    # Qwen2.5-0.5B/1.5B and Llama-3.2-1B all do. Getting this wrong inflates
    # the parameter count by 2 * vocab * hidden, which for Qwen2.5-0.5B is
    # 136M on a 494M model: a 27% error that propagates straight into every
    # roofline estimate below. Check ``tie_word_embeddings`` in config.json.
    tie_word_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.n_heads

    @classmethod
    def from_hf_config(cls, model_id: str, **overrides) -> "ModelShape":
        """Build from a model's ``config.json`` on the Hub or on disk.

        Always prefer this to typing numbers in by hand -- the fields are the
        same ones the engine itself reads, so the napkin math and the engine
        cannot drift apart.
        """
        from transformers import AutoConfig  # noqa: PLC0415

        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        n_heads = getattr(cfg, "num_attention_heads")
        shape = cls(
            n_layers=getattr(cfg, "num_hidden_layers"),
            hidden_size=getattr(cfg, "hidden_size"),
            n_heads=n_heads,
            n_kv_heads=getattr(cfg, "num_key_value_heads", n_heads),
            intermediate_size=getattr(cfg, "intermediate_size"),
            vocab_size=getattr(cfg, "vocab_size"),
            head_dim=getattr(cfg, "head_dim", None),
            tie_word_embeddings=bool(getattr(cfg, "tie_word_embeddings", False)),
        )
        for key, value in overrides.items():
            setattr(shape, key, value)
        return shape

    def params(self) -> int:
        """Approximate parameter count (embeddings + blocks + lm_head).

        Good to ~1% for Llama/Qwen-style models, which is all the precision the
        estimates below deserve. Verified against Qwen2.5-0.5B (494M) and
        Qwen2.5-1.5B (1.54B) in ``tests/test_bench.py``.
        """
        d, h = self.hidden_size, self.head_dim or 1
        attn = d * (self.n_heads * h) + 2 * d * (self.n_kv_heads * h) + (self.n_heads * h) * d
        mlp = 3 * d * self.intermediate_size          # gate, up, down (SwiGLU)
        per_layer = attn + mlp + 2 * d                 # + 2 RMSNorm weights
        embeddings = self.vocab_size * d * (1 if self.tie_word_embeddings else 2)
        return self.n_layers * per_layer + embeddings

    def kv_cache_bytes(self, seq_len: int, batch: int = 1, dtype_bytes: int = 2) -> int:
        """Bytes of KV cache for ``batch`` sequences of ``seq_len`` tokens.

        2 (K and V) x layers x kv_heads x head_dim x tokens x bytes.

        The ``n_kv_heads`` term is why grouped-query attention exists: Qwen2.5
        and Llama-3 share each KV head across several query heads, cutting this
        number (and therefore the number of concurrent users that fit in HBM)
        by the GQA ratio, typically 4-8x.
        """
        return (
            2
            * self.n_layers
            * self.n_kv_heads
            * (self.head_dim or 1)
            * seq_len
            * batch
            * dtype_bytes
        )

    def prefill_flops(self, seq_len: int) -> float:
        """~2 * params * tokens, plus the quadratic attention term.

        The 2 is one multiply + one add per parameter per token. The attention
        term is 4 * layers * seq_len^2 * hidden (scores + the weighted sum) and
        only starts to dominate for long prompts -- which is exactly why long
        context is a different engineering problem from long generation.
        """
        dense = 2.0 * self.params() * seq_len
        attn = 4.0 * self.n_layers * (seq_len**2) * self.hidden_size
        return dense + attn

    def decode_bytes_moved(self, dtype_bytes: int = 2, kv_bytes: int = 0) -> float:
        """Bytes read from HBM to generate ONE token for ONE sequence.

        Every weight must be read once, plus that sequence's KV cache. This is
        the whole reason decode is memory-bound: the arithmetic intensity is
        about 2 FLOPs per byte, while an A100 needs ~160 FLOPs/byte to saturate
        its tensor cores. The hardware is idle ~99% of the time at batch 1.
        """
        return self.params() * dtype_bytes + kv_bytes


def roofline_decode_tok_s(
    shape: ModelShape,
    hbm_bandwidth_gb_s: float,
    dtype_bytes: int = 2,
    batch: int = 1,
    seq_len: int = 0,
    efficiency: float = 1.0,
) -> float:
    """Upper bound on decode tokens/s from memory bandwidth alone.

    Weights are read *once per step* no matter the batch size -- that is the
    entire economic argument for batching. KV cache, by contrast, is per
    sequence, so it grows with the batch and eventually becomes the dominant
    term for long contexts.

    ``efficiency`` is the fraction of peak bandwidth you actually achieve;
    0.7-0.85 is typical for a well-written kernel. Leave it at 1.0 to get the
    hard physical ceiling, which is the useful number for "how much headroom
    is left in this kernel?".
    """
    kv = shape.kv_cache_bytes(seq_len, batch=batch, dtype_bytes=dtype_bytes) if seq_len else 0
    bytes_per_step = shape.params() * dtype_bytes + kv
    steps_per_s = (hbm_bandwidth_gb_s * 1e9 * efficiency) / bytes_per_step
    return steps_per_s * batch


def arithmetic_intensity(flops: float, bytes_moved: float) -> float:
    """FLOPs per byte. Compare against the GPU's ridge point (peak FLOPS /
    peak bandwidth) to classify a kernel as memory- or compute-bound.

    Ridge points (peak dense fp16 TFLOP/s divided by peak HBM GB/s):

        T4            65 / 0.320  ~ 203 FLOP/byte
        L4           121 / 0.300  ~ 403
        A100-40GB    312 / 1.555  ~ 201
        A100-80GB    312 / 2.039  ~ 153
        H100 SXM     989 / 3.350  ~ 295

    Note how little the ridge moves across four hardware generations while
    absolute performance grows ~15x: memory bandwidth has not kept pace with
    arithmetic. Anything below the ridge is bandwidth-starved and no amount of
    faster math will help it.
    """
    return flops / bytes_moved if bytes_moved else float("inf")


def littles_law_concurrency(throughput_req_s: float, latency_s: float) -> float:
    """L = lambda * W. The in-flight request count implied by a (throughput,
    latency) pair.

    Use it as a consistency check on any benchmark: if a server reports 20
    req/s at 2s mean latency, there were ~40 requests in flight. If you only
    sent 8, one of the two numbers is wrong -- usually the client was the
    bottleneck, which is the most common way a serving benchmark lies.
    """
    return throughput_req_s * latency_s
