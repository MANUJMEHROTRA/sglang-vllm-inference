"""Launching and babysitting inference servers from inside a Colab notebook.

Colab makes this awkward: a server is a long-lived foreground process, a
notebook cell is not, and when the server dies at import time you get a dead
port and no traceback. Everything here exists to make the failure *visible*.

The pattern used by every notebook:

    with serve(VLLM_CMD, port=8000) as url:
        model = await discover_model_id(url)
        ...

On exit the process is terminated and the GPU is freed. That last part matters
more than it sounds: a leaked vLLM process holds ~90% of VRAM (it preallocates
the KV cache pool), so the *next* cell fails with a confusing OOM that has
nothing to do with the code you just wrote. ``free_gpu()`` and ``kill_stale()``
are the escape hatches.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence

import urllib.error
import urllib.request

DEFAULT_LOG_DIR = Path("/content/logs") if Path("/content").exists() else Path("./logs")


def _http_ok(url: str, timeout: float = 5.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
            return 200 <= resp.status < 300
    except (urllib.error.URLError, OSError, ValueError):
        return False


def wait_for_http(
    url: str,
    proc: Optional[subprocess.Popen] = None,
    timeout_s: float = 900.0,
    poll_s: float = 3.0,
    log_path: Optional[Path] = None,
) -> None:
    """Poll ``url`` until it answers, or the process dies, or we time out.

    Checking ``proc.poll()`` each round is what turns "hung for 15 minutes" into
    "crashed after 20 seconds, here is the log". Model servers fail fast and
    loudly far more often than they hang.
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            tail = _tail(log_path) if log_path else "(no log captured)"
            raise RuntimeError(
                f"server exited with code {proc.returncode} before becoming ready.\n"
                f"--- last log lines ---\n{tail}"
            )
        if _http_ok(url):
            return
        time.sleep(poll_s)
    tail = _tail(log_path) if log_path else "(no log captured)"
    raise TimeoutError(f"{url} not ready after {timeout_s}s\n--- last log lines ---\n{tail}")


def _tail(path: Optional[Path], n: int = 40) -> str:
    if not path or not Path(path).exists():
        return "(no log file)"
    lines = Path(path).read_text(errors="replace").splitlines()
    return "\n".join(lines[-n:])


@contextlib.contextmanager
def serve(
    cmd: Sequence[str],
    port: int = 8000,
    host: str = "127.0.0.1",
    health_path: str = "/v1/models",
    timeout_s: float = 900.0,
    log_name: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    log_dir: Optional[Path] = None,
) -> Iterator[str]:
    """Start a server, yield its base URL, and always tear it down.

    stdout and stderr go to a log file rather than the notebook, because these
    servers emit thousands of lines and a notebook cell that captures them all
    becomes unusable (and the saved .ipynb becomes enormous). ``print_log()``
    shows it on demand; a crash dumps the tail automatically.
    """
    log_dir = Path(log_dir or DEFAULT_LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / (log_name or f"server-{port}.log")

    full_env = {**os.environ, **(env or {})}
    base_url = f"http://{host}:{port}"
    print(f"launching: {' '.join(cmd)}")
    print(f"logging to: {log_path}")

    with log_path.open("w") as log_file:
        proc = subprocess.Popen(
            list(cmd),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=full_env,
            # New process group so we can signal the whole tree. vLLM and
            # SGLang both fork worker processes; killing only the parent
            # orphans the workers and they keep the GPU pinned.
            start_new_session=True,
        )
    try:
        t0 = time.time()
        wait_for_http(base_url + health_path, proc, timeout_s, log_path=log_path)
        print(f"ready in {time.time() - t0:.0f}s -> {base_url}")
        yield base_url
    finally:
        stop(proc)
        print(f"server stopped (log kept at {log_path})")


def stop(proc: subprocess.Popen, grace_s: float = 20.0) -> None:
    """SIGTERM the whole process group, then SIGKILL what survives."""
    if proc.poll() is not None:
        return
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    deadline = time.time() + grace_s
    while time.time() < deadline and proc.poll() is None:
        time.sleep(0.5)
    if proc.poll() is None:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait(timeout=10)


def restart_runtime(reason: str = "") -> None:
    """Restart the Colab/Jupyter kernel.

    Mandatory after any pip install that changes torch. Two separate things go
    wrong otherwise, and they compound:

    1. The old torch module objects are already loaded in memory, so the
       process keeps using them regardless of what is now on disk.
    2. If the swap was partial, the on-disk torch is a mix of two versions and
       imports of its own internals fail.

    A restart fixes (1) outright and makes (2) visible immediately instead of
    three cells later. Re-run the notebook from the top afterwards -- every
    install cell is idempotent, so the second pass is fast.
    """
    if reason:
        print(reason)
    print("Restarting the runtime. Re-run from the top when it comes back "
          "(installs are already done, so it will be quick).")
    try:
        import IPython  # noqa: PLC0415

        IPython.Application.instance().kernel.do_shutdown(True)
    except Exception:  # noqa: BLE001 - not in a kernel; fall back to a hard exit
        os.kill(os.getpid(), signal.SIGKILL)


def install_engine(pip_spec: str, engine_name: Optional[str] = None) -> bool:
    """``pip install`` an engine and report whether torch moved underneath us.

    Returns True when the runtime needs restarting. Engine notebooks use this
    instead of a bare ``%pip install`` so that a torch swap is detected at the
    moment it happens rather than diagnosed later from a traceback inside
    whichever library imported torch first.
    """
    import importlib.metadata as md  # noqa: PLC0415
    import sys  # noqa: PLC0415

    def version_of(pkg: str) -> Optional[str]:
        try:
            return md.version(pkg)
        except md.PackageNotFoundError:
            return None

    before = version_of("torch")
    print(f"torch before install: {before or 'not installed'}")

    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pip_spec.split()],
                   check=True)

    # The import system caches directory listings; without this the freshly
    # written dist-info is invisible and we would read the pre-install version.
    import importlib  # noqa: PLC0415

    importlib.invalidate_caches()

    after = version_of("torch")
    print(f"torch after install:  {after or 'not installed'}")

    if before and after and before != after:
        print(f"\ntorch changed {before} -> {after} under a running kernel.")
        return True
    return False


def print_log(port: int = 8000, n: int = 60, log_dir: Optional[Path] = None) -> None:
    print(_tail(Path(log_dir or DEFAULT_LOG_DIR) / f"server-{port}.log", n))


def kill_stale(patterns: Sequence[str] = ("vllm", "sglang", "llama-server", "trtllm")) -> None:
    """Kill leftover server processes from a previous cell or a crashed run.

    Run this when a launch fails with "address already in use" or an OOM that
    makes no sense given the model size.
    """
    for pat in patterns:
        subprocess.run(["pkill", "-9", "-f", pat], check=False)
    time.sleep(2)


def free_gpu() -> None:
    """Drop cached allocations and report what is still held.

    If memory stays high after this, a *separate process* holds it -- call
    ``kill_stale()``. ``torch.cuda.empty_cache()`` can only release memory
    owned by this process's allocator, which is the usual source of confusion.
    """
    import gc  # noqa: PLC0415

    gc.collect()
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            free_b, total_b = torch.cuda.mem_get_info()
            print(
                f"GPU memory: {(total_b - free_b) / 1024**3:.1f} GB used / "
                f"{total_b / 1024**3:.1f} GB total"
            )
    except ImportError:
        pass
    subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
         "--format=csv"],
        check=False,
    )


# ---------------------------------------------------------------------------
# Command builders. Each returns an argv list; the notebooks print it before
# launching so the exact configuration is visible in the saved output.
# ---------------------------------------------------------------------------


def vllm_cmd(
    model: str,
    port: int = 8000,
    max_model_len: int = 4096,
    gpu_memory_utilization: float = 0.85,
    dtype: str = "auto",
    enable_prefix_caching: Optional[bool] = None,
    max_num_seqs: int = 256,
    max_num_batched_tokens: Optional[int] = None,
    quantization: Optional[str] = None,
    extra: Sequence[str] = (),
) -> List[str]:
    """``vllm serve`` argv.

    The flags worth understanding before you touch anything else:

    ``gpu_memory_utilization``  vLLM grabs this fraction of VRAM *up front* and
        carves the leftover (after weights and activations) into KV cache
        blocks. Raising it directly raises how many sequences fit concurrently.
        Too high and CUDA graph capture OOMs; 0.85 is a safe Colab default,
        0.90-0.95 works once nothing else shares the GPU.
    ``max_model_len``  caps context. It also caps the KV cache *per sequence*,
        so lowering it is the fastest way to fit more concurrent requests on a
        16GB T4. Leaving it at the model's full 32k on a small GPU is the most
        common cause of "only 2 sequences fit".
    ``max_num_seqs``  ceiling on sequences in one decode batch. The real knob
        for the latency/throughput trade-off: lower = better tail latency,
        higher = better throughput, until the KV cache runs out and vLLM starts
        preempting.
    ``max_num_batched_tokens``  tokens per scheduler step. With chunked prefill
        on, this is what stops one 8k-token prefill from stalling every decode
        in flight -- the direct cause of TTFT/ITL spikes under mixed load.
    """
    cmd = [
        "vllm", "serve", model,
        "--port", str(port),
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--dtype", dtype,
        "--max-num-seqs", str(max_num_seqs),
        "--disable-log-requests",
    ]
    if enable_prefix_caching is True:
        cmd.append("--enable-prefix-caching")
    elif enable_prefix_caching is False:
        cmd.append("--no-enable-prefix-caching")
    if max_num_batched_tokens:
        cmd += ["--max-num-batched-tokens", str(max_num_batched_tokens)]
    if quantization:
        cmd += ["--quantization", quantization]
    return cmd + list(extra)


def sglang_cmd(
    model: str,
    port: int = 8000,
    context_length: int = 4096,
    mem_fraction_static: float = 0.80,
    dtype: str = "auto",
    disable_radix_cache: bool = False,
    max_running_requests: int = 256,
    chunked_prefill_size: Optional[int] = None,
    quantization: Optional[str] = None,
    extra: Sequence[str] = (),
) -> List[str]:
    """``python -m sglang.launch_server`` argv.

    Name mapping against vLLM, since the concepts are the same and only the
    spelling differs:

    ======================  ==========================
    vLLM                    SGLang
    ======================  ==========================
    gpu_memory_utilization  mem_fraction_static
    max_model_len           context_length
    max_num_seqs            max_running_requests
    enable_prefix_caching   (radix cache, on by default)
    max_num_batched_tokens  chunked_prefill_size
    ======================  ==========================

    The important asymmetry: vLLM's prefix caching is opt-in in some versions,
    while SGLang's RadixAttention is **on by default**. Benchmarking stock
    settings therefore compares "cache on" against "cache off" and produces the
    lopsided numbers you see in a lot of blog posts. ``disable_radix_cache``
    exists so notebook 02 can run the controlled version of that experiment.
    """
    cmd = [
        "python", "-m", "sglang.launch_server",
        "--model-path", model,
        "--port", str(port),
        "--host", "127.0.0.1",
        "--context-length", str(context_length),
        "--mem-fraction-static", str(mem_fraction_static),
        "--dtype", dtype,
        "--max-running-requests", str(max_running_requests),
    ]
    if disable_radix_cache:
        cmd.append("--disable-radix-cache")
    if chunked_prefill_size:
        cmd += ["--chunked-prefill-size", str(chunked_prefill_size)]
    if quantization:
        cmd += ["--quantization", quantization]
    return cmd + list(extra)


def llamacpp_cmd(
    gguf_path: str,
    port: int = 8000,
    n_gpu_layers: int = 99,
    ctx_size: int = 4096,
    n_parallel: int = 8,
    threads: Optional[int] = None,
    flash_attn: bool = True,
    cont_batching: bool = True,
    binary: str = "llama-server",
    extra: Sequence[str] = (),
) -> List[str]:
    """``llama-server`` argv.

    ``n_gpu_layers`` (``-ngl``) is the flag that defines llama.cpp's character.
    It offloads the first N transformer layers to the GPU and runs the rest on
    CPU. 0 = pure CPU; 99 = "all of them, if they fit". The in-between is the
    interesting part and the reason llama.cpp exists: it is the only engine
    here that will run a model which does not fit in VRAM, at a speed set by
    whichever side holds more layers. Partial offload is *not* linear -- every
    token crosses PCIe once per boundary, so a 50/50 split is far worse than
    half speed. Notebook 03 measures that curve.

    ``ctx_size`` is the KV budget shared across ``n_parallel`` slots, so each
    concurrent request really gets ``ctx_size / n_parallel`` tokens of context.
    Forgetting this division is the usual cause of truncated outputs at
    concurrency.
    """
    cmd = [
        binary,
        "-m", gguf_path,
        "--port", str(port),
        "--host", "127.0.0.1",
        "-ngl", str(n_gpu_layers),
        "-c", str(ctx_size),
        "-np", str(n_parallel),
    ]
    if threads:
        cmd += ["-t", str(threads)]
    if flash_attn:
        cmd += ["-fa", "on"]
    if cont_batching:
        cmd.append("--cont-batching")
    return cmd + list(extra)


def trtllm_cmd(
    model: str,
    port: int = 8000,
    max_batch_size: int = 64,
    max_num_tokens: int = 8192,
    max_seq_len: int = 4096,
    kv_cache_fraction: float = 0.80,
    backend: str = "pytorch",
    extra: Sequence[str] = (),
) -> List[str]:
    """``trtllm-serve`` argv.

    TensorRT-LLM now ships two backends. The ``pytorch`` one needs no ahead-of-
    time engine build and behaves like vLLM/SGLang; the classic ``tensorrt``
    one compiles a fixed engine first (see notebook 04) and is faster but
    rebuilt whenever a shape bound changes. We default to ``pytorch`` because
    a Colab session rarely has the ~15 minutes an engine build costs.
    """
    cmd = [
        "trtllm-serve", model,
        "--port", str(port),
        "--host", "127.0.0.1",
        "--backend", backend,
        "--max_batch_size", str(max_batch_size),
        "--max_num_tokens", str(max_num_tokens),
        "--max_seq_len", str(max_seq_len),
        "--kv_cache_free_gpu_memory_fraction", str(kv_cache_fraction),
    ]
    return cmd + list(extra)
