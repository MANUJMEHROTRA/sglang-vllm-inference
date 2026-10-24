"""Capture what the benchmark actually ran on.

Never type hardware details by hand into a result file. On Colab you do not
choose your GPU -- you are assigned one, it can change between sessions, and a
T4 (sm_75, 16GB, 320 GB/s) and an L4 (sm_89, 24GB, 300 GB/s but with FP8 and
much stronger fp16 tensor cores) produce numbers that are not comparable. Half
the irreproducible benchmarks on the internet are really "I got a different GPU
that day".
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
from typing import Dict, Optional

from .schema import Hardware

# Peak HBM bandwidth (GB/s) and dense fp16 tensor-core TFLOPS, for roofline
# math in the notebooks. Vendor peak numbers: treat them as the ceiling you
# will never reach, not as a target. ~70-85% of peak bandwidth is a good
# kernel; ~50-70% of peak FLOPS is a good GEMM.
GPU_SPECS: Dict[str, Dict[str, float]] = {
    # name substring -> specs
    "T4":        {"hbm_gb_s": 320.0,  "fp16_tflops": 65.0,   "mem_gb": 16.0,  "sm": 75},
    "L4":        {"hbm_gb_s": 300.0,  "fp16_tflops": 121.0,  "mem_gb": 24.0,  "sm": 89},
    "V100":      {"hbm_gb_s": 900.0,  "fp16_tflops": 125.0,  "mem_gb": 16.0,  "sm": 70},
    "A100-SXM4-80": {"hbm_gb_s": 2039.0, "fp16_tflops": 312.0, "mem_gb": 80.0, "sm": 80},
    "A100":      {"hbm_gb_s": 1555.0, "fp16_tflops": 312.0,  "mem_gb": 40.0,  "sm": 80},
    "H100":      {"hbm_gb_s": 3350.0, "fp16_tflops": 989.0,  "mem_gb": 80.0,  "sm": 90},
    "L40S":      {"hbm_gb_s": 864.0,  "fp16_tflops": 362.0,  "mem_gb": 48.0,  "sm": 89},
    "RTX 4090":  {"hbm_gb_s": 1008.0, "fp16_tflops": 165.0,  "mem_gb": 24.0,  "sm": 89},
}

# Rough on-demand USD/hour, used only for the cost-per-token chart in notebook
# 07. These move constantly -- override them for your own accounting.
GPU_HOURLY_USD: Dict[str, float] = {
    "T4": 0.35,
    "L4": 0.80,
    "V100": 0.90,
    "A100": 1.80,
    "A100-SXM4-80": 2.50,
    "H100": 3.50,
    "L40S": 1.20,
}


def gpu_specs_for(gpu_name: Optional[str]) -> Dict[str, float]:
    """Longest-substring match so 'NVIDIA A100-SXM4-80GB' beats the 'A100' key."""
    if not gpu_name:
        return {}
    matches = [k for k in GPU_SPECS if k.lower() in gpu_name.lower()]
    if not matches:
        return {}
    return GPU_SPECS[max(matches, key=len)]


def hourly_usd_for(gpu_name: Optional[str]) -> Optional[float]:
    if not gpu_name:
        return None
    matches = [k for k in GPU_HOURLY_USD if k.lower() in gpu_name.lower()]
    return GPU_HOURLY_USD[max(matches, key=len)] if matches else None


def _nvidia_smi() -> Optional[Dict[str, str]]:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        q = "name,memory.total,driver_version,compute_cap"
        out = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"],
            text=True,
            timeout=20,
        ).strip()
    except Exception:
        return None
    if not out:
        return None
    lines = [ln for ln in out.splitlines() if ln.strip()]
    first = [f.strip() for f in lines[0].split(",")]
    info = {"count": str(len(lines)), "name": first[0]}
    if len(first) > 1:
        m = re.search(r"(\d+)", first[1])
        if m:
            info["memory_gb"] = str(round(int(m.group(1)) / 1024, 1))
    if len(first) > 2:
        info["driver"] = first[2]
    if len(first) > 3:
        info["compute_cap"] = first[3]
    return info


def capture(usd_per_hour: Optional[float] = None) -> Hardware:
    """Best-effort environment capture. Works with or without torch/CUDA."""
    hw = Hardware()

    smi = _nvidia_smi()
    if smi:
        hw.gpu_name = smi.get("name")
        hw.gpu_count = int(smi.get("count", 1))
        hw.gpu_memory_gb = float(smi["memory_gb"]) if "memory_gb" in smi else None
        hw.driver_version = smi.get("driver")
        hw.gpu_compute_capability = smi.get("compute_cap")

    try:
        import torch  # noqa: PLC0415 - optional dependency by design

        hw.torch_version = torch.__version__
        hw.cuda_version = torch.version.cuda
        if torch.cuda.is_available():
            hw.gpu_count = hw.gpu_count or torch.cuda.device_count()
            if not hw.gpu_name:
                hw.gpu_name = torch.cuda.get_device_name(0)
            if not hw.gpu_compute_capability:
                major, minor = torch.cuda.get_device_capability(0)
                hw.gpu_compute_capability = f"{major}.{minor}"
            if hw.gpu_memory_gb is None:
                total = torch.cuda.get_device_properties(0).total_memory
                hw.gpu_memory_gb = round(total / 1024**3, 1)
    except Exception:
        pass

    hw.cpu = platform.processor() or platform.machine()
    hw.cpu_count = os.cpu_count()
    try:
        # Linux only; harmless elsewhere.
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        hw.ram_gb = round(pages * page_size / 1024**3, 1)
    except (ValueError, OSError, AttributeError):
        pass

    hw.usd_per_hour = usd_per_hour if usd_per_hour is not None else hourly_usd_for(hw.gpu_name)
    return hw


def package_versions(*names: str) -> Dict[str, str]:
    """Record the versions of the packages that produced a result.

    These libraries ship performance changes in patch releases. A result
    without its engine version is a rumour, so every notebook writes this into
    ``EngineSpec.version`` / ``flags``.
    """
    from importlib.metadata import PackageNotFoundError, version  # noqa: PLC0415

    out: Dict[str, str] = {}
    for name in names:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = "not-installed"
    return out


def summary_line(hw: Optional[Hardware] = None) -> str:
    """One-line banner for the top of each notebook."""
    hw = hw or capture()
    specs = gpu_specs_for(hw.gpu_name)
    bits = [hw.gpu_name or "no GPU"]
    if hw.gpu_memory_gb:
        bits.append(f"{hw.gpu_memory_gb:g} GB")
    if hw.gpu_compute_capability:
        bits.append(f"sm_{hw.gpu_compute_capability.replace('.', '')}")
    if specs:
        bits.append(f"{specs['hbm_gb_s']:.0f} GB/s")
        bits.append(f"{specs['fp16_tflops']:.0f} TF fp16")
    if hw.torch_version:
        bits.append(f"torch {hw.torch_version}")
    if hw.cuda_version:
        bits.append(f"cuda {hw.cuda_version}")
    return " | ".join(bits)


def capability_report(hw: Optional[Hardware] = None) -> Dict[str, bool]:
    """What this GPU can and cannot run. Saves a lot of confusing crashes.

    The compute capability gates are real and non-negotiable:
      - sm_75 (T4): no bf16, no FlashAttention-2, no FP8, no Marlin kernels.
        You will be on fp16 + xformers/FlashInfer paths. Plenty of tutorials
        silently assume sm_80+ and fail here.
      - sm_80 (A100): bf16, FA2, Marlin int4. No FP8.
      - sm_89 (L4/L40S/4090): adds FP8 (e4m3/e5m2).
      - sm_90 (H100): adds FP8 with tensor-core acceleration worth using, plus
        the Hopper-only FlashAttention-3 path.
    """
    hw = hw or capture()
    cap = hw.gpu_compute_capability
    sm = float(cap) if cap else 0.0
    return {
        "has_gpu": bool(hw.gpu_name),
        "bf16": sm >= 8.0,
        "flash_attention_2": sm >= 8.0,
        "fp8": sm >= 8.9,
        "marlin_int4": sm >= 8.0,
        "fp4_nvfp4": sm >= 10.0,
        "tensorrt_llm_supported": sm >= 7.5,
        "cuda_graphs": bool(hw.gpu_name),
    }
