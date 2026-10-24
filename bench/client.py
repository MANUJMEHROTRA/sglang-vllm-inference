"""One async HTTP client that drives every generative engine identically.

The central decision of this repo
---------------------------------
Every generative engine here -- vLLM, SGLang, llama.cpp's ``llama-server``,
TensorRT-LLM's ``trtllm-serve``, and Triton's OpenAI frontend -- is benchmarked
through the **same OpenAI-compatible ``/v1/completions`` endpoint over HTTP
with streaming on**. Not through each engine's native Python API.

That costs a few hundred microseconds of HTTP and SSE overhead per request.
It buys the only thing that makes the comparison mean anything:

* **TTFT is measured the same way everywhere** -- wall clock from "bytes on the
  socket" to "first token byte back". Native Python APIs each define their
  first-token moment slightly differently, and some do not expose it at all.
* **The same client code path, the same event loop, the same parsing.** Any
  client-side overhead is a constant across engines and cancels out of the
  comparison.
* **It is how you will actually deploy.** A number from an offline
  ``LLM.generate()`` call includes none of the scheduling, queueing or
  tokenisation work that a real server does, and is routinely 20-40% optimistic.

``/v1/completions`` rather than ``/v1/chat/completions`` is deliberate too:
chat endpoints apply a chat template, and templates differ between engines and
versions. That silently changes the prompt token count, which changes prefill
cost, which changes the thing we are trying to measure. Raw completions keep
the token count under our control.

Token counting
--------------
We prefer the server's own ``usage`` block (requested via
``stream_options={"include_usage": true}``) and fall back to counting stream
chunks. Which method was used is recorded on every record, because the two can
disagree by a few percent and that must never be mistaken for a speed
difference.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .schema import RequestRecord

try:
    import aiohttp
except ImportError:  # pragma: no cover - surfaced with a clear message at use
    aiohttp = None  # type: ignore[assignment]


# Per-engine extras that mean "generate exactly N tokens, do not stop early".
# Without this the engines stop at different points on the same prompt and you
# end up comparing the models' verbosity rather than the engines' speed.
IGNORE_EOS_EXTRA: Dict[str, Dict[str, Any]] = {
    "vllm": {"ignore_eos": True},
    "sglang": {"ignore_eos": True},
    # llama-server honours ignore_eos on its OpenAI-compatible route too, and
    # needs min_tokens to stop the sampler bailing out early.
    "llamacpp": {"ignore_eos": True},
    "tensorrt_llm": {"ignore_eos": True},
    "triton": {"ignore_eos": True},
    "hf": {},
}


@dataclass
class EndpointConfig:
    """Where to send requests and what dialect to speak."""

    base_url: str = "http://127.0.0.1:8000"
    model: str = ""                    # the id the server advertises at /v1/models
    engine: str = "vllm"               # selects the extras above
    api_key: str = "EMPTY"
    timeout_s: float = 600.0
    endpoint: str = "/v1/completions"
    extra_body: Dict[str, Any] = field(default_factory=dict)

    def url(self) -> str:
        return self.base_url.rstrip("/") + self.endpoint

    def headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }


def build_payload(
    cfg: EndpointConfig,
    prompt: str,
    max_tokens: int,
    ignore_eos: bool = True,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """Assemble the request body.

    ``temperature=0`` is the default so that output is deterministic and two
    runs of the same workload differ only in timing. (Sampling cost itself is
    negligible next to the forward pass, so this does not flatter anyone.)
    """
    payload: Dict[str, Any] = {
        "model": cfg.model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        # Ask for the usage block in the final chunk. Servers that do not
        # understand this key ignore it, and we fall back to chunk counting.
        "stream_options": {"include_usage": True},
    }
    if ignore_eos:
        payload.update(IGNORE_EOS_EXTRA.get(cfg.engine, {}))
    payload.update(cfg.extra_body)
    return payload


async def stream_one(
    session: "aiohttp.ClientSession",
    cfg: EndpointConfig,
    prompt: str,
    max_tokens: int,
    start_offset_s: float,
    ignore_eos: bool = True,
) -> RequestRecord:
    """Issue one streaming completion and time every token arrival.

    Timing notes that matter:

    * ``perf_counter`` is taken immediately *before* the send and immediately
      *on arrival* of each chunk. It is monotonic; ``time.time()`` is not and
      can jump backwards mid-run.
    * The first token's timestamp is taken on the first chunk carrying
      **non-empty text**. Several servers emit a priming chunk with an empty
      string, and counting that as "first token" understates TTFT -- sometimes
      dramatically, since the priming chunk can be sent before prefill starts.
    * No work happens between the timestamp and appending to the list. Parsing
      the JSON *after* stamping keeps client-side CPU out of the measurement.
    """
    record = RequestRecord(start_offset_s=start_offset_s)
    payload = build_payload(cfg, prompt, max_tokens, ignore_eos=ignore_eos)

    chunk_tokens = 0
    usage: Optional[Dict[str, Any]] = None
    t_prev: Optional[float] = None

    t0 = time.perf_counter()
    try:
        async with session.post(
            cfg.url(),
            json=payload,
            headers=cfg.headers(),
            timeout=aiohttp.ClientTimeout(total=cfg.timeout_s),
        ) as resp:
            if resp.status != 200:
                body = (await resp.text())[:400]
                record.success = False
                record.error = f"HTTP {resp.status}: {body}"
                return record

            async for raw in resp.content:
                now = time.perf_counter()           # stamp first, parse second
                line = raw.decode("utf-8", "replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue

                if obj.get("usage"):
                    usage = obj["usage"]

                text = _extract_text(obj)
                if not text:
                    continue                        # priming / usage-only chunk

                if record.ttft_s is None:
                    record.ttft_s = now - t0
                else:
                    record.itls_s.append(now - (t_prev or now))
                t_prev = now
                chunk_tokens += 1

        record.e2e_s = time.perf_counter() - t0

    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - record it, never kill the sweep
        record.success = False
        record.error = f"{type(exc).__name__}: {exc}"
        return record

    if record.ttft_s is None:
        record.success = False
        record.error = "stream produced no tokens"
        return record

    if usage and usage.get("completion_tokens"):
        record.prompt_tokens = usage.get("prompt_tokens")
        record.output_tokens = usage.get("completion_tokens")
        record.token_count_method = "usage"
    else:
        record.output_tokens = chunk_tokens
        record.token_count_method = "chunks"

    return record


def _extract_text(obj: Dict[str, Any]) -> str:
    """Pull generated text out of a completions *or* chat-completions chunk.

    Supporting both means the same client works if you point it at a chat
    endpoint for a one-off sanity check, without a second code path to keep
    honest.
    """
    choices = obj.get("choices") or []
    if not choices:
        return ""
    choice = choices[0]
    if "text" in choice and choice["text"]:
        return choice["text"]
    delta = choice.get("delta") or {}
    return delta.get("content") or ""


async def wait_until_ready(
    base_url: str, timeout_s: float = 600.0, poll_s: float = 2.0
) -> Dict[str, Any]:
    """Block until ``/v1/models`` answers, then return what it advertises.

    Model load is slow and highly variable (weight download, CUDA graph
    capture, TensorRT engine deserialisation). Polling the real endpoint is the
    only reliable readiness signal -- a fixed ``sleep(60)`` either wastes a
    minute or starts benchmarking a server that is still capturing graphs,
    which poisons the first few requests of the run.
    """
    if aiohttp is None:
        raise RuntimeError("aiohttp is required: pip install aiohttp")
    deadline = time.time() + timeout_s
    last_err = "never attempted"
    async with aiohttp.ClientSession() as session:
        while time.time() < deadline:
            try:
                async with session.get(
                    base_url.rstrip("/") + "/v1/models",
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    last_err = f"HTTP {resp.status}"
            except Exception as exc:  # noqa: BLE001
                last_err = f"{type(exc).__name__}: {exc}"
            await asyncio.sleep(poll_s)
    raise TimeoutError(f"server at {base_url} not ready after {timeout_s}s (last: {last_err})")


async def discover_model_id(base_url: str, timeout_s: float = 600.0) -> str:
    """Return the first model id the server advertises.

    Engines disagree about what to call a model: vLLM echoes the HF path,
    llama-server returns the GGUF filename, TensorRT-LLM returns the engine
    directory name. Asking removes a whole class of "model not found" errors.
    """
    info = await wait_until_ready(base_url, timeout_s=timeout_s)
    data = info.get("data") or []
    if not data:
        raise RuntimeError(f"{base_url}/v1/models returned no models: {info}")
    return data[0]["id"]


async def warmup(cfg: EndpointConfig, n: int = 3, max_tokens: int = 16) -> List[RequestRecord]:
    """Fire a few throwaway requests and discard their timings.

    Always do this. The first request through a fresh server pays for lazy CUDA
    context creation, cuBLAS/cuDNN autotuning, memory-pool growth, JIT of
    Triton kernels, and sometimes CUDA graph capture. On a T4 the first request
    can be 10x slower than the steady state. Including it in a 32-request run
    moves the mean by a third and the p99 by far more.
    """
    if aiohttp is None:
        raise RuntimeError("aiohttp is required: pip install aiohttp")
    async with aiohttp.ClientSession() as session:
        return [
            await stream_one(session, cfg, "Warm up the engine.", max_tokens, 0.0)
            for _ in range(n)
        ]


async def probe(cfg: EndpointConfig, prompt: str = "Hello", max_tokens: int = 8) -> RequestRecord:
    """Single request, for checking a server is alive and answering sanely."""
    if aiohttp is None:
        raise RuntimeError("aiohttp is required: pip install aiohttp")
    async with aiohttp.ClientSession() as session:
        return await stream_one(session, cfg, prompt, max_tokens, 0.0)
