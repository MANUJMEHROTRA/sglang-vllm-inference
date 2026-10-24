"""Shared benchmarking harness for the inference-engine notebooks.

Import surface kept small on purpose -- a notebook should start with

    from bench import *   # noqa: F403

and have everything it needs to describe, run, summarise and plot an
experiment, without importing torch or any engine until it actually needs one.
"""

from .client import EndpointConfig, discover_model_id, probe, stream_one, wait_until_ready
from .metrics import (
    DEFAULT_SLO,
    ModelShape,
    arithmetic_intensity,
    goodput,
    littles_law_concurrency,
    percentile,
    roofline_decode_tok_s,
    summarize,
)
from .runner import benchmark, report, run, sweep_concurrency, validate
from .schema import (
    SCHEMA_VERSION,
    EngineSpec,
    Hardware,
    RequestRecord,
    RunResult,
    WorkloadSpec,
    load_dir,
)
from .servers import (
    free_gpu,
    kill_stale,
    llamacpp_cmd,
    print_log,
    serve,
    sglang_cmd,
    trtllm_cmd,
    vllm_cmd,
)
from .sysinfo import (
    GPU_SPECS,
    capability_report,
    capture,
    gpu_specs_for,
    package_versions,
    summary_line,
)
from .workloads import PromptBuilder, describe, preset

__all__ = [
    "DEFAULT_SLO",
    "GPU_SPECS",
    "SCHEMA_VERSION",
    "EndpointConfig",
    "EngineSpec",
    "Hardware",
    "ModelShape",
    "PromptBuilder",
    "RequestRecord",
    "RunResult",
    "WorkloadSpec",
    "arithmetic_intensity",
    "benchmark",
    "capability_report",
    "capture",
    "describe",
    "discover_model_id",
    "free_gpu",
    "goodput",
    "gpu_specs_for",
    "kill_stale",
    "littles_law_concurrency",
    "llamacpp_cmd",
    "load_dir",
    "package_versions",
    "percentile",
    "preset",
    "print_log",
    "probe",
    "report",
    "roofline_decode_tok_s",
    "run",
    "serve",
    "sglang_cmd",
    "stream_one",
    "summarize",
    "summary_line",
    "sweep_concurrency",
    "trtllm_cmd",
    "validate",
    "vllm_cmd",
    "wait_until_ready",
]

__version__ = "1.0.0"
