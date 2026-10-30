"""Workloads: the traffic shapes we measure against.

A serving engine has no single speed. It has a speed *per workload*, and the
workloads below are chosen to separate effects that get muddled together in a
single "tokens/s" headline:

``chat``          short prompt, medium output -> decode-dominated. The classic
                  memory-bandwidth-bound regime. Batching helps enormously.
``summarize``     long prompt, short output -> prefill-dominated, compute-bound.
                  Chunked prefill and FlashAttention matter; KV cache size is
                  the binding constraint.
``rag``           long *shared* prefix + short unique tail. The workload where
                  prefix caching / RadixAttention stops being a footnote and
                  becomes a 3-10x effect.
``agent``         many short turns reusing a growing prefix. Same mechanism as
                  rag, harsher on the scheduler.
``longgen``       short prompt, long output -> pure decode. Isolates the decode
                  kernel and KV growth from everything else.

Prompts are synthetic by design
-------------------------------
We build prompts from a fixed pseudo-random token soup rather than ShareGPT or
a real dataset. Reasons:

1. **Exact, reproducible lengths.** The single biggest source of bogus
   comparisons is different engines being fed different numbers of tokens. Here
   the input length is an input parameter, not a property of a dataset.
2. **No tokenizer drift.** Lengths are measured with the *target model's* own
   tokenizer, so "512 tokens" means the same thing to every engine in a run.
3. **No download, no licence, no network flake** on a fresh Colab runtime.

The cost is realism: real text has structure that affects nothing in the
prefill/decode cost model but does affect output *length* when the model is
allowed to stop. We sidestep that with ``ignore_eos=True`` so every response is
exactly ``output_len`` tokens. Without that flag you are benchmarking the
model's verbosity as much as the engine's speed.

If you want realistic text instead, ``natural_prompt()`` builds prompts from
real English sentences at a controlled token length -- useful for the quality
and perplexity work in notebook 03, where token *identity* matters.
"""

from __future__ import annotations

import random
from dataclasses import replace
from typing import List, Optional, Sequence, Tuple

from .schema import WorkloadSpec
from .tokenizer import as_tokenizer

# A pool of common English words. Real-ish subwords keep the tokenizer in a
# normal regime (no weird byte-fallback explosions) while staying meaningless.
_WORDS = (
    "system model token memory latency throughput kernel batch cache attention "
    "layer tensor matrix vector weight gradient compute bandwidth schedule queue "
    "request response server client stream decode prefill quantize precision "
    "parallel pipeline fusion register shared global warp thread block grid "
    "optimize profile measure baseline variance percentile distribution sample "
    "context window sequence position embedding projection normalization residual "
    "inference training evaluation benchmark hardware software framework runtime"
).split()

_SENTENCES = (
    "The scheduler decides which requests run in the next forward pass. "
    "Memory bandwidth, not arithmetic, is the binding constraint during decode. "
    "A paged key-value cache removes the need for contiguous allocation. "
    "Continuous batching lets a finished sequence leave the batch immediately. "
    "Quantization trades a small amount of accuracy for a large amount of speed. "
    "Prefill is compute bound because every prompt token is processed at once. "
    "The first token is slow; every token after it is cheap by comparison. "
    "Tail latency, not average latency, is what a user actually notices. "
)


PRESETS = {
    "chat": WorkloadSpec(
        name="chat", input_len=256, output_len=256, n_requests=64, shared_prefix_tokens=0
    ),
    "summarize": WorkloadSpec(
        name="summarize", input_len=4096, output_len=64, n_requests=32, shared_prefix_tokens=0
    ),
    "rag": WorkloadSpec(
        name="rag", input_len=2048, output_len=128, n_requests=64, shared_prefix_tokens=1792
    ),
    "agent": WorkloadSpec(
        name="agent", input_len=1024, output_len=64, n_requests=96, shared_prefix_tokens=896
    ),
    "longgen": WorkloadSpec(
        name="longgen", input_len=128, output_len=1024, n_requests=24, shared_prefix_tokens=0
    ),
}


def preset(name: str, **overrides) -> WorkloadSpec:
    """Fetch a named workload, optionally overriding fields.

    ``preset("chat", concurrency=16, n_requests=128)``
    """
    if name not in PRESETS:
        raise KeyError(f"unknown workload {name!r}; have {sorted(PRESETS)}")
    return replace(PRESETS[name], **overrides)


class PromptBuilder:
    """Builds prompts of an exact token length for a specific tokenizer.

    Why a class: the token-length calibration needs the tokenizer, and we want
    to build the shared prefix exactly once so that every request in a
    prefix-caching workload really does share a byte-identical prefix. Even a
    one-character difference at the front defeats a radix cache completely,
    which is a mistake worth making once in a notebook and never again in
    production.
    """

    def __init__(self, tokenizer, seed: int = 0):
        # Accepts a `tokenizers.Tokenizer`, a transformers tokenizer, or the
        # adapter returned by bench.load_tokenizer -- see bench/tokenizer.py
        # for why the fast backend is worth preferring on Colab.
        self.tok = as_tokenizer(tokenizer)
        self.seed = seed
        self._prefix_cache: dict[int, str] = {}

    def _encode_len(self, text: str) -> int:
        return len(self.tok.encode(text))

    def text_of_tokens(self, n_tokens: int, rng: random.Random) -> str:
        """Generate text that encodes to *exactly* ``n_tokens`` tokens.

        Grow by words until we overshoot, then trim token-wise and decode back.
        The decode/re-encode round trip is the only way to be exact, because
        word-to-token ratios vary by tokenizer (Qwen2.5 sits around 1.3
        tokens/word on text like this; a BERT WordPiece tokenizer is different
        again).
        """
        if n_tokens <= 0:
            return ""
        words: List[str] = []
        # ~1.4 tokens per word is a safe overshoot for the vocabularies here.
        target_words = max(1, int(n_tokens / 1.4))
        while True:
            while len(words) < target_words:
                words.append(rng.choice(_WORDS))
            text = " ".join(words)
            ids = self.tok.encode(text)
            if len(ids) >= n_tokens:
                return self.tok.decode(ids[:n_tokens])
            target_words = max(target_words + 8, int(target_words * 1.2))

    def shared_prefix(self, n_tokens: int) -> str:
        """A deterministic prefix, identical across every call and every engine.

        Cached so that repeated calls in one session return the same string --
        a fresh random prefix per request would quietly turn a cache-hit
        experiment into a cache-miss experiment.
        """
        if n_tokens not in self._prefix_cache:
            rng = random.Random(1_000_000 + n_tokens)  # independent of self.seed
            header = "You are a helpful assistant. Use the following reference material.\n\n"
            body = self.text_of_tokens(max(0, n_tokens - self._encode_len(header)), rng)
            self._prefix_cache[n_tokens] = header + body
        return self._prefix_cache[n_tokens]

    def build(self, spec: WorkloadSpec) -> List[str]:
        """Materialise every prompt for a workload.

        Returns ``spec.n_requests`` prompts, each ~``spec.input_len`` tokens,
        all beginning with the same ``shared_prefix_tokens``-token prefix.
        """
        rng = random.Random(spec.seed)
        prefix = self.shared_prefix(spec.shared_prefix_tokens) if spec.shared_prefix_tokens else ""
        prefix_len = self._encode_len(prefix) if prefix else 0

        prompts: List[str] = []
        for _ in range(spec.n_requests):
            total = spec.input_len
            if spec.input_len_std:
                total = max(prefix_len + 8, int(rng.gauss(spec.input_len, spec.input_len_std)))
            unique_len = max(1, total - prefix_len)
            tail = self.text_of_tokens(unique_len, rng)
            prompts.append(prefix + tail if prefix else tail)
        return prompts

    def verify(self, prompts: Sequence[str], spec: WorkloadSpec) -> Tuple[float, int, int]:
        """(mean, min, max) actual token length. Print this. Always.

        If the mean is not within a token or two of ``spec.input_len``, the
        comparison is already broken and no amount of careful timing will fix
        it.
        """
        lens = [self._encode_len(p) for p in prompts]
        return (sum(lens) / len(lens), min(lens), max(lens))


def natural_prompt(n_repeats: int = 1) -> str:
    """Real English, for quality/perplexity work where token identity matters."""
    return (_SENTENCES * n_repeats).strip()


def poisson_arrivals(n: int, rate: float, seed: int = 0) -> List[float]:
    """Arrival offsets (seconds from run start) for an open-loop run.

    Inter-arrival times are Exponential(rate), which is what "Poisson process
    at ``rate`` req/s" means. Using evenly-spaced arrivals instead -- a common
    shortcut -- removes the bursts, and bursts are precisely what makes
    queueing interesting: a server that handles 10 req/s uniformly can fall
    over at 10 req/s Poisson.
    """
    rng = random.Random(seed)
    out, t = [], 0.0
    for _ in range(n):
        out.append(t)
        t += rng.expovariate(rate)
    return out


def describe(spec: WorkloadSpec) -> str:
    """Human-readable one-liner for plot titles and notebook output."""
    bits = [
        f"{spec.name}",
        f"in={spec.input_len}",
        f"out={spec.output_len}",
        f"n={spec.n_requests}",
    ]
    if spec.shared_prefix_tokens:
        pct = 100.0 * spec.shared_prefix_tokens / max(1, spec.input_len)
        bits.append(f"shared_prefix={spec.shared_prefix_tokens} ({pct:.0f}%)")
    if spec.mode == "closed":
        bits.append(f"concurrency={spec.concurrency}")
    else:
        bits.append(f"rate={spec.request_rate} req/s")
    return "  ".join(bits)
