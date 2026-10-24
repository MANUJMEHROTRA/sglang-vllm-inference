"""The result schema that every notebook in this repo writes.

Everything benchmarked here -- vLLM, SGLang, llama.cpp, TensorRT-LLM, Triton,
plain HuggingFace, ONNX Runtime -- emits the *same* JSON document. That is the
only reason the final comparison notebook can put them on one axis without
hand-waving.

Four rules keep the comparison honest:

1. **One schema version.** If a field changes meaning, bump SCHEMA_VERSION so
   stale result files are rejected loudly instead of silently plotted beside
   new ones.
2. **Raw per-request records are kept, not just summaries.** You cannot
   recompute a p99 from someone else's p50/p99 pair, and you cannot re-derive
   goodput under a *different* SLO from a summary. You can from raw records.
3. **Provenance is mandatory.** "vLLM is 2x faster than X" is meaningless
   without the GPU, the driver, the library version and the exact engine
   flags. Prefix caching alone can swing a number by 10x, so the flags are
   part of the measurement, not a footnote.
4. **The workload is part of the result's identity.** Same engine + same model
   + different input/output lengths is a *different point*, never something to
   average with the first one. Prefill-heavy and decode-heavy workloads stress
   opposite hardware limits (see notebook 00).

Nothing in this module imports torch, numpy or any engine. It stays importable
in a bare Python 3.9+ environment so results can be inspected and re-plotted
anywhere, including on a laptop with no GPU.
"""

from __future__ import annotations

import json
import platform
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# Bump on any breaking change to the field meanings below.
SCHEMA_VERSION = "1.0"

# Engines we know how to drive. Kept as a tuple (not an Enum) so result files
# stay plain JSON and a new engine is a one-line change.
KNOWN_ENGINES = (
    "hf",             # plain transformers .generate() -- the honest baseline
    "vllm",
    "sglang",
    "llamacpp",       # llama.cpp / llama-server, GGUF weights
    "tensorrt_llm",
    "triton",         # Triton Inference Server fronting some backend
    "onnxruntime",
    "torch_compile",
)

# Decode-only (autoregressive) vs encode-only (single forward pass). The
# metric set differs: TTFT/TPOT are meaningless for an encoder, where the whole
# answer arrives in one shot. See notebook 06.
TASK_GENERATIVE = "generative"
TASK_ENCODE = "encode"


@dataclass
class RequestRecord:
    """One request's raw timeline. All times are seconds, relative to the
    moment the request was *sent* (not to the start of the run) except
    ``start_offset_s`` which is relative to run start.

    ``itls_s`` holds inter-token latencies: the gaps between successive token
    arrivals *after* the first. len(itls_s) == output_tokens - 1 for a clean
    stream. These are kept raw because the tail of this distribution -- not its
    mean -- is what users actually feel as "stuttering".
    """

    start_offset_s: float
    ttft_s: Optional[float] = None
    e2e_s: Optional[float] = None
    itls_s: List[float] = field(default_factory=list)
    prompt_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    success: bool = True
    error: Optional[str] = None

    # "usage" when the server reported token counts, "chunks" when we fell back
    # to counting stream chunks. Recorded because the two can disagree by a few
    # percent and that difference should never be mistaken for a speed change.
    token_count_method: Optional[str] = None


@dataclass
class EngineSpec:
    """What was actually running. Reproducibility lives or dies here."""

    engine: str
    version: str                      # e.g. "0.32.0"
    model_id: str                     # HF id or GGUF filename
    task: str = TASK_GENERATIVE
    model_family: Optional[str] = None     # "qwen", "llama", "bert", ...
    params_b: Optional[float] = None       # billions, for tokens/s/param math
    dtype: str = "bf16"                    # bf16 | fp16 | fp8 | int4 | Q4_K_M
    quantization: Optional[str] = None     # awq | gptq | fp8 | gguf-Q4_K_M
    max_model_len: Optional[int] = None
    tensor_parallel_size: int = 1

    # The flags that move the number. Put anything here that you would have to
    # mention when defending the result in review: enable_prefix_caching,
    # chunked_prefill, max_num_seqs, gpu_memory_utilization, n_gpu_layers,
    # cuda_graph, attention_backend, ...
    flags: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkloadSpec:
    """The traffic shape. Two different shapes are two different experiments.

    ``mode``:
      - "closed": a fixed number of in-flight requests; each worker fires the
        next one the instant its previous finishes. Measures *capacity* and is
        what a batch job looks like.
      - "open": Poisson arrivals at ``request_rate`` req/s, independent of how
        fast the server is. Measures *queueing* and is what real traffic looks
        like; this is the only mode where a server can fall behind and where
        goodput becomes the interesting metric.
    """

    name: str
    mode: str = "closed"
    n_requests: int = 64
    concurrency: Optional[int] = None       # closed mode
    request_rate: Optional[float] = None    # open mode, req/s
    input_len: int = 512
    input_len_std: int = 0
    output_len: int = 128
    shared_prefix_tokens: int = 0           # drives prefix-cache / RadixAttention
    seed: int = 0
    ignore_eos: bool = True                 # fixed output length == comparable


@dataclass
class Hardware:
    """Captured, never typed by hand -- see bench.sysinfo.capture()."""

    gpu_name: Optional[str] = None
    gpu_count: int = 0
    gpu_memory_gb: Optional[float] = None
    gpu_compute_capability: Optional[str] = None   # "7.5" (T4), "8.9" (L4)...
    driver_version: Optional[str] = None
    cuda_version: Optional[str] = None
    torch_version: Optional[str] = None
    cpu: Optional[str] = None
    cpu_count: Optional[int] = None
    ram_gb: Optional[float] = None
    platform: str = field(default_factory=platform.platform)

    # Used by the cost chart in notebook 07. Edit to match what you actually
    # pay; the default is a rough on-demand cloud rate, not a Colab price.
    usd_per_hour: Optional[float] = None


@dataclass
class RunResult:
    """One (engine, model, workload, hardware) point."""

    engine: EngineSpec
    workload: WorkloadSpec
    hardware: Hardware
    metrics: Dict[str, Any] = field(default_factory=dict)
    requests: List[RequestRecord] = field(default_factory=list)

    schema_version: str = SCHEMA_VERSION
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: float = field(default_factory=time.time)
    duration_s: Optional[float] = None
    notes: str = ""

    # Set True for numbers that did NOT come from a real run on this machine
    # (illustrative data, figures quoted from a paper). The comparison notebook
    # and the website both render these differently. Never let a synthetic
    # number pass as measured.
    synthetic: bool = False

    # ---- serialisation -------------------------------------------------

    def to_dict(self, include_requests: bool = True) -> Dict[str, Any]:
        d = asdict(self)
        if not include_requests:
            d["requests"] = []
        return d

    def save(self, directory: str | Path, include_requests: bool = True) -> Path:
        """Write to ``<dir>/<engine>__<model>__<workload>__<gpu>__<run_id>.json``.

        The filename encodes the identity of the point so a directory listing
        is already a readable index, and so two runs that differ only in a flag
        cannot overwrite each other.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.slug()}.json"
        path.write_text(json.dumps(self.to_dict(include_requests), indent=2))
        return path

    def slug(self) -> str:
        model = _safe(self.engine.model_id.split("/")[-1])
        gpu = _safe(self.hardware.gpu_name or "cpu")
        return "__".join(
            [_safe(self.engine.engine), model, _safe(self.workload.name), gpu, self.run_id]
        )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RunResult":
        version = d.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"result schema {version!r} != expected {SCHEMA_VERSION!r}. "
                "Re-run the benchmark rather than plotting incomparable fields."
            )
        return cls(
            engine=EngineSpec(**d["engine"]),
            workload=WorkloadSpec(**d["workload"]),
            hardware=Hardware(**d["hardware"]),
            metrics=d.get("metrics", {}),
            requests=[RequestRecord(**r) for r in d.get("requests", [])],
            schema_version=version,
            run_id=d["run_id"],
            timestamp=d["timestamp"],
            duration_s=d.get("duration_s"),
            notes=d.get("notes", ""),
            synthetic=d.get("synthetic", False),
        )

    @classmethod
    def load(cls, path: str | Path) -> "RunResult":
        return cls.from_dict(json.loads(Path(path).read_text()))


def load_dir(directory: str | Path, strict: bool = False) -> List[RunResult]:
    """Load every ``*.json`` result under ``directory`` (recursively).

    With ``strict=False`` a file that fails to parse is skipped with a warning
    rather than killing a long comparison notebook; the warning names the file
    so you can go fix it.
    """
    out: List[RunResult] = []
    for path in sorted(Path(directory).rglob("*.json")):
        if path.name in {"index.json", "results.json", "versions.lock.json"}:
            continue
        try:
            out.append(RunResult.load(path))
        except Exception as exc:  # noqa: BLE001 - we want the filename in the msg
            if strict:
                raise
            print(f"[load_dir] skipping {path.name}: {exc}")
    return out


def _safe(s: str) -> str:
    keep = "-."
    return "".join(c if (c.isalnum() or c in keep) else "-" for c in s).strip("-")
