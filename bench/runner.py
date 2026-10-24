"""Drives a workload against an endpoint and returns a RunResult.

Two load models, because they answer different questions
--------------------------------------------------------

**Closed loop** (``run_closed``): N workers, each sending its next request the
instant the previous one returns. In-flight count is pinned at N forever. This
measures *capacity*: sweep N and you trace the engine's latency/throughput
curve all the way to saturation. It is also what an offline batch job looks
like. What it cannot show you is queueing collapse, because the load
automatically backs off whenever the server slows down.

**Open loop** (``run_open``): requests arrive on a Poisson process at a fixed
rate, regardless of whether the server is keeping up. This measures *queueing*.
Push the rate past capacity and latency diverges -- the cliff is the point of
the experiment, and it is invisible in a closed-loop sweep.

Most published "benchmarks" are closed-loop only, which is why they tend to
report throughput numbers that no production system ever sees at an acceptable
latency. Notebook 07 plots both.

The client must not be the bottleneck
-------------------------------------
A single Python event loop parsing SSE can itself cap out around a few thousand
chunks/s. ``validate_not_client_bound`` applies Little's Law after every run
and shouts if the observed concurrency is far below what was requested, which
is the signature of a client-side limit. Believing a client-bound number is the
most common way a serving benchmark goes wrong.
"""

from __future__ import annotations

import asyncio
import time
from typing import List, Optional, Sequence

from .client import EndpointConfig, stream_one, warmup
from .metrics import DEFAULT_SLO, littles_law_concurrency, summarize
from .schema import EngineSpec, Hardware, RequestRecord, RunResult, WorkloadSpec
from .sysinfo import capture
from .workloads import poisson_arrivals

try:
    import aiohttp
except ImportError:  # pragma: no cover
    aiohttp = None  # type: ignore[assignment]


def _session():
    # The connector limit must exceed the concurrency we intend to drive, or
    # aiohttp silently queues connections and we end up measuring our own
    # client's connection pool instead of the server. Default is 100.
    connector = aiohttp.TCPConnector(limit=0, force_close=False, ttl_dns_cache=300)
    return aiohttp.ClientSession(connector=connector)


async def run_closed(
    cfg: EndpointConfig,
    spec: WorkloadSpec,
    prompts: Sequence[str],
    progress: bool = True,
) -> tuple[List[RequestRecord], float]:
    """Fixed in-flight count. Returns (records, wall_duration_s)."""
    concurrency = spec.concurrency or 1
    queue: asyncio.Queue = asyncio.Queue()
    for i, p in enumerate(prompts):
        queue.put_nowait((i, p))

    records: List[Optional[RequestRecord]] = [None] * len(prompts)
    done = 0

    async with _session() as session:
        t_start = time.perf_counter()

        async def worker() -> None:
            nonlocal done
            while True:
                try:
                    idx, prompt = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                rec = await stream_one(
                    session,
                    cfg,
                    prompt,
                    spec.output_len,
                    start_offset_s=time.perf_counter() - t_start,
                    ignore_eos=spec.ignore_eos,
                )
                records[idx] = rec
                done += 1
                if progress and done % max(1, len(prompts) // 10) == 0:
                    print(f"  {done}/{len(prompts)} done", flush=True)

        await asyncio.gather(*[worker() for _ in range(concurrency)])
        duration = time.perf_counter() - t_start

    return [r for r in records if r is not None], duration


async def run_open(
    cfg: EndpointConfig,
    spec: WorkloadSpec,
    prompts: Sequence[str],
    progress: bool = True,
) -> tuple[List[RequestRecord], float]:
    """Poisson arrivals at ``spec.request_rate``, independent of server speed.

    Note the duration here is measured to the completion of the last request,
    which includes any queueing backlog. If the server cannot keep up, duration
    exceeds the arrival window and throughput lands below the offered rate --
    that gap *is* the overload signal.
    """
    rate = spec.request_rate or 1.0
    offsets = poisson_arrivals(len(prompts), rate, seed=spec.seed)
    records: List[Optional[RequestRecord]] = [None] * len(prompts)
    done = 0

    async with _session() as session:
        t_start = time.perf_counter()

        async def fire(idx: int, prompt: str, at: float) -> None:
            nonlocal done
            delay = at - (time.perf_counter() - t_start)
            if delay > 0:
                await asyncio.sleep(delay)
            records[idx] = await stream_one(
                session,
                cfg,
                prompt,
                spec.output_len,
                start_offset_s=time.perf_counter() - t_start,
                ignore_eos=spec.ignore_eos,
            )
            done += 1
            if progress and done % max(1, len(prompts) // 10) == 0:
                print(f"  {done}/{len(prompts)} done", flush=True)

        await asyncio.gather(*[fire(i, p, t) for i, (p, t) in enumerate(zip(prompts, offsets))])
        duration = time.perf_counter() - t_start

    return [r for r in records if r is not None], duration


async def benchmark(
    cfg: EndpointConfig,
    engine: EngineSpec,
    spec: WorkloadSpec,
    prompts: Sequence[str],
    hardware: Optional[Hardware] = None,
    slo: Optional[dict] = None,
    do_warmup: bool = True,
    notes: str = "",
    progress: bool = True,
) -> RunResult:
    """Warm up, run the workload, summarise, and package a RunResult."""
    if aiohttp is None:
        raise RuntimeError("aiohttp is required: pip install aiohttp")
    if len(prompts) != spec.n_requests:
        raise ValueError(
            f"{len(prompts)} prompts for a spec declaring n_requests={spec.n_requests}; "
            "regenerate prompts from the spec so the result's identity is accurate."
        )

    if do_warmup:
        if progress:
            print("warming up (timings discarded)...", flush=True)
        await warmup(cfg)

    if progress:
        print(f"running {spec.name}: {spec.mode} mode, {spec.n_requests} requests", flush=True)

    runner = run_closed if spec.mode == "closed" else run_open
    records, duration = await runner(cfg, spec, prompts, progress=progress)

    result = RunResult(
        engine=engine,
        workload=spec,
        hardware=hardware or capture(),
        metrics=summarize(records, duration, slo or DEFAULT_SLO),
        requests=records,
        duration_s=duration,
        notes=notes,
    )
    if progress:
        print(report(result), flush=True)
        for warning in validate(result):
            print(f"  !! {warning}", flush=True)
    return result


def validate(result: RunResult) -> List[str]:
    """Sanity checks that catch the usual lies. Run after every single run.

    None of these are style nits -- each one corresponds to a specific way a
    serving benchmark silently produces a meaningless number.
    """
    warnings: List[str] = []
    m = result.metrics
    spec = result.workload

    if m.get("n_failed"):
        warnings.append(f"{m['n_failed']} requests failed; metrics cover only the successes")

    if m.get("n_success", 0) < 20:
        warnings.append(
            f"only {m.get('n_success')} successful requests -- p99 figures are noise at this n"
        )

    # Did we actually generate the number of tokens we asked for? If not,
    # ignore_eos was not honoured and the comparison is length-confounded.
    n_ok = max(1, m.get("n_success", 1))
    mean_out = m.get("total_output_tokens", 0) / n_ok
    if spec.ignore_eos and abs(mean_out - spec.output_len) > max(2, 0.05 * spec.output_len):
        warnings.append(
            f"mean output {mean_out:.1f} tokens vs requested {spec.output_len}: "
            "ignore_eos may not be supported by this server -- lengths differ between "
            "engines and the comparison is confounded"
        )

    if spec.mode == "closed" and spec.concurrency:
        observed = littles_law_concurrency(
            m.get("request_throughput_req_s", 0.0), m.get("e2e_s_mean", 0.0)
        )
        if observed < 0.7 * spec.concurrency:
            warnings.append(
                f"Little's Law says ~{observed:.1f} requests were in flight but "
                f"{spec.concurrency} were requested -- the client, not the server, may be "
                "the bottleneck"
            )

    if spec.mode == "open" and spec.request_rate:
        achieved = m.get("request_throughput_req_s", 0.0)
        if achieved < 0.9 * spec.request_rate:
            warnings.append(
                f"achieved {achieved:.2f} req/s against an offered {spec.request_rate} req/s -- "
                "the server is past capacity (this is a real finding, not an error)"
            )

    methods = {r.token_count_method for r in result.requests if r.success}
    if methods == {"chunks"}:
        warnings.append(
            "token counts came from chunk counting, not the server's usage block -- "
            "fine, but do not compare these counts against a usage-based run to the token"
        )
    return warnings


def report(result: RunResult) -> str:
    """The standard text block printed after every run."""
    m = result.metrics
    ms = lambda key: m.get(key, float("nan")) * 1000  # noqa: E731

    lines = [
        "",
        f"  engine        {result.engine.engine} {result.engine.version} "
        f"({result.engine.dtype})",
        f"  model         {result.engine.model_id}",
        f"  workload      {result.workload.name}  in={result.workload.input_len} "
        f"out={result.workload.output_len} "
        + (
            f"concurrency={result.workload.concurrency}"
            if result.workload.mode == "closed"
            else f"rate={result.workload.request_rate} req/s"
        ),
        f"  requests      {m.get('n_success')}/{m.get('n_requests')} ok "
        f"in {m.get('duration_s', 0):.1f}s",
        "",
        f"  TTFT    mean {ms('ttft_s_mean'):8.1f} ms   p50 {ms('ttft_s_p50'):8.1f}   "
        f"p99 {ms('ttft_s_p99'):8.1f}",
        f"  TPOT    mean {ms('tpot_s_mean'):8.2f} ms   p50 {ms('tpot_s_p50'):8.2f}   "
        f"p99 {ms('tpot_s_p99'):8.2f}",
        f"  ITL     mean {ms('itl_s_mean'):8.2f} ms   p50 {ms('itl_s_p50'):8.2f}   "
        f"p99 {ms('itl_s_p99'):8.2f}",
        f"  E2E     mean {m.get('e2e_s_mean', 0):8.2f} s    p99 {m.get('e2e_s_p99', 0):8.2f} s",
        "",
        f"  output throughput  {m.get('output_throughput_tok_s', 0):9.1f} tok/s",
        f"  total  throughput  {m.get('total_throughput_tok_s', 0):9.1f} tok/s "
        "(incl. prompt -- flatters prefill-heavy runs)",
        f"  request throughput {m.get('request_throughput_req_s', 0):9.2f} req/s",
        f"  goodput            {m.get('goodput_req_s', 0):9.2f} req/s "
        f"({100 * m.get('goodput_fraction', 0):.0f}% met SLO {m.get('slo')})",
    ]
    return "\n".join(lines)


async def sweep_concurrency(
    cfg: EndpointConfig,
    engine: EngineSpec,
    base_spec: WorkloadSpec,
    builder,
    levels: Sequence[int] = (1, 2, 4, 8, 16, 32, 64),
    hardware: Optional[Hardware] = None,
    notes: str = "",
) -> List[RunResult]:
    """Walk the latency/throughput curve.

    This is *the* experiment of the whole repo. A single data point ("engine X
    does N tok/s") is close to meaningless; the shape of this curve is the
    engine's actual character:

    * where throughput stops rising -> the saturation point, i.e. real capacity
    * how steeply TTFT climbs after it -> how the scheduler handles overload
    * the gap between concurrency 1 and the knee -> how much the engine gains
      from batching, which is the single biggest differentiator between a
      naive loop and a production server

    ``builder`` is a :class:`bench.workloads.PromptBuilder`. Prompts are rebuilt
    per level because ``n_requests`` scales with concurrency -- otherwise high
    concurrency levels finish in a handful of seconds and the measurement is
    dominated by ramp-up.
    """
    from dataclasses import replace  # noqa: PLC0415

    results: List[RunResult] = []
    for level in levels:
        # Keep each level running long enough to reach steady state: at least
        # 8 requests per worker, so the batch composition settles.
        n = max(base_spec.n_requests, level * 8)
        spec = replace(base_spec, mode="closed", concurrency=level, n_requests=n)
        prompts = builder.build(spec)
        print(f"\n=== concurrency {level} ===")
        results.append(
            await benchmark(
                cfg, engine, spec, prompts, hardware=hardware, notes=notes, progress=False
            )
        )
    return results


def run(coro):
    """Run an async benchmark from a notebook cell.

    Jupyter already owns a running event loop, so ``asyncio.run`` raises
    there. ``nest_asyncio`` patches the loop to allow re-entry; it is installed
    by the setup cell in notebook 00.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    try:
        import nest_asyncio  # noqa: PLC0415

        nest_asyncio.apply()
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Running inside a notebook event loop: pip install nest_asyncio"
        ) from exc
    return loop.run_until_complete(coro)
