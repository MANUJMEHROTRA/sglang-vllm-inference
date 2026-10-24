"""Build .ipynb files from plain Python sources.

Why generate notebooks instead of editing them directly: .ipynb is JSON with
escaped strings, which makes prose painful to write, impossible to review in a
diff, and prone to merge conflicts over execution counts and output blobs. The
real source of truth lives in ``tools/nbsrc/*.py`` as readable text; this
module turns it into notebooks.

    python tools/build_notebooks.py

Notebooks are generated with no outputs. Run them on Colab, then save the
executed copies into ``results/`` (the result JSON is what matters; the
notebook outputs are a convenience).
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent


def repo_slug() -> str:
    """``owner/name`` for the Colab badge links.

    Resolution order: ``$REPO_SLUG`` env var, then the git remote, then a
    placeholder. The placeholder is deliberately obvious so a wrong badge is
    noticed immediately rather than 404-ing quietly for a reader.
    """
    if os.environ.get("REPO_SLUG"):
        return os.environ["REPO_SLUG"]
    try:
        url = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "remote", "get-url", "origin"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "YOUR-GITHUB-USERNAME/sglang-vllm-inference"
    return slug_from_url(url) or "YOUR-GITHUB-USERNAME/sglang-vllm-inference"


def slug_from_url(url: str) -> str:
    """``owner/name`` from either remote form.

    Both shapes have to work, and getting it wrong is silent -- the badge
    simply 404s for every reader:

        git@github.com:owner/name.git        -> owner/name
        https://github.com/owner/name.git    -> owner/name

    Taking the text after the last ``:`` handles only the SSH form; on an
    HTTPS URL it leaves ``//github.com/owner/name``, which then survives as a
    bogus three-segment slug.
    """
    url = url.strip().removesuffix(".git")
    if url.startswith("git@") or (":" in url and "//" not in url):
        url = url.split(":", 1)[-1]          # SSH: host:owner/name
    parts = [p for p in url.split("/") if p and p != "github.com"]
    return "/".join(parts[-2:]) if len(parts) >= 2 else ""


def md(text: str) -> Dict[str, Any]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": _lines(text),
    }


def code(text: str, collapsed: bool = False) -> Dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"jupyter": {"source_hidden": collapsed}} if collapsed else {},
        "outputs": [],
        "source": _lines(text),
    }


def _lines(text: str) -> List[str]:
    """nbformat stores source as a list of lines, each keeping its newline."""
    text = text.strip("\n")
    lines = text.split("\n")
    return [line + "\n" for line in lines[:-1]] + [lines[-1]]


def colab_badge(notebook_name: str) -> str:
    slug = repo_slug()
    url = f"https://colab.research.google.com/github/{slug}/blob/main/notebooks/{notebook_name}"
    return f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({url})"


def notebook(cells: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4", "toc_visible": True},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write(path: Path, cells: List[Dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(notebook(cells), indent=1) + "\n")
    n_code = sum(1 for c in cells if c["cell_type"] == "code")
    n_md = len(cells) - n_code
    print(f"  {path.name:<46} {len(cells):>3} cells ({n_md} md, {n_code} code)")
    return path


# ---------------------------------------------------------------------------
# Blocks reused across notebooks, so the setup story is identical everywhere.
# ---------------------------------------------------------------------------

SETUP_CODE = '''
# --- Colab setup -----------------------------------------------------------
# Safe to re-run. Clones the repo on Colab; uses the local checkout elsewhere.
import os, subprocess, sys
from pathlib import Path

REPO = "{repo}"
ROOT = Path("/content/sglang-vllm-inference")

if Path("/content").exists():                      # we are on Colab
    if not ROOT.exists():
        subprocess.run(["git", "clone", "--depth", "1",
                        f"https://github.com/{{REPO}}.git", str(ROOT)], check=True)
    os.chdir(ROOT)
else:                                              # local checkout
    ROOT = Path.cwd()
    while not (ROOT / "bench").exists() and ROOT != ROOT.parent:
        ROOT = ROOT.parent
    os.chdir(ROOT)

sys.path.insert(0, str(ROOT))
print("working directory:", Path.cwd())

%pip install -q aiohttp nest_asyncio matplotlib pandas
'''

BANNER_CODE = '''
from bench import capture, capability_report, summary_line, gpu_specs_for
from bench.plots import use_theme
use_theme()

HW = capture()
CAPS = capability_report(HW)
SPECS = gpu_specs_for(HW.gpu_name)

print(summary_line(HW))
print()
for name, available in CAPS.items():
    print(f"  {'yes' if available else ' no'}  {name}")

if not CAPS["has_gpu"]:
    print("\\nNo GPU detected. On Colab: Runtime -> Change runtime type -> T4 GPU.")
'''


def setup_cells(notebook_name: str, title: str, subtitle: str, time_estimate: str,
                needs_gpu: bool = True) -> List[Dict[str, Any]]:
    """The identical opening of every notebook: badge, intro, setup, banner."""
    gpu_note = (
        "**Runtime:** GPU required. `Runtime -> Change runtime type -> T4 GPU` "
        "(or L4/A100 on Colab Pro).\n\n"
        if needs_gpu
        else "**Runtime:** CPU is fine for this notebook.\n\n"
    )
    return [
        md(
            f"# {title}\n\n"
            f"{colab_badge(notebook_name)}\n\n"
            f"{subtitle}\n\n"
            f"{gpu_note}"
            f"**Time to run:** {time_estimate}\n\n"
            "---"
        ),
        code(SETUP_CODE.format(repo=repo_slug())),
        code(BANNER_CODE),
    ]
