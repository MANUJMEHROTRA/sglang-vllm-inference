"""Generate notebooks/*.ipynb from the readable sources in tools/nbsrc/.

    python tools/build_notebooks.py
    REPO_SLUG=you/your-repo python tools/build_notebooks.py   # fix Colab badges

Edit the .py sources, never the .ipynb files -- regenerating overwrites them.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.nbgen import repo_slug, write  # noqa: E402

MODULES = [
    "tools.nbsrc.nb00_foundations",
    "tools.nbsrc.nb01_vllm",
    "tools.nbsrc.nb02_sglang",
    "tools.nbsrc.nb03_llamacpp",
    "tools.nbsrc.nb04_tensorrt",
    "tools.nbsrc.nb05_triton",
    "tools.nbsrc.nb06_encoders",
    "tools.nbsrc.nb07_comparison",
    "tools.nbsrc.nb08_kernels",
]


def check(path: Path) -> list[str]:
    """Structural checks on a generated notebook.

    These catch the mistakes that only show up when a cell is executed on
    Colab 20 minutes into a session: an unbalanced string from a bad escape,
    a cell that is accidentally empty, or source that is not a list of lines.
    """
    problems: list[str] = []
    nb = json.loads(path.read_text())

    for i, cell in enumerate(nb["cells"]):
        src = "".join(cell["source"])
        where = f"{path.name} cell {i} ({cell['cell_type']})"
        if not src.strip():
            problems.append(f"{where}: empty")
        if not isinstance(cell["source"], list):
            problems.append(f"{where}: source is not a list of lines")
        if cell["cell_type"] == "code":
            # Compile each code cell, skipping IPython magics which are not
            # valid Python but are valid notebook input.
            stripped = "\n".join(
                "" if ln.lstrip().startswith(("%", "!")) else ln
                for ln in src.split("\n")
            )
            try:
                compile(stripped, f"{path.name}:cell{i}", "exec")
            except SyntaxError as exc:
                problems.append(f"{where}: SyntaxError line {exc.lineno}: {exc.msg}")
    return problems


def stamp_site(slug: str) -> None:
    """Write the repo slug into docs/index.html.

    The site is served from docs/ as its own root, so notebook links have to be
    absolute github.com URLs. Stamping here means one command keeps the
    notebooks and the site consistent.
    """
    index = REPO_ROOT / "docs" / "index.html"
    if not index.exists():
        return
    text = index.read_text()
    import re

    new_text = re.sub(r'(<body data-repo=")[^"]*(")', rf"\1{slug}\2", text, count=1)
    if new_text != text:
        index.write_text(new_text)
        print(f"  stamped docs/index.html with data-repo=\"{slug}\"")


def main() -> int:
    out_dir = REPO_ROOT / "notebooks"
    slug = repo_slug()
    print(f"repo slug for Colab badges: {slug}")
    if slug.startswith("YOUR-GITHUB"):
        print("  (no git remote found -- set REPO_SLUG=owner/name, or add a remote,")
        print("   then re-run to fix the Colab badges and the site's notebook links)")
    print()

    all_problems: list[str] = []
    for module_name in MODULES:
        module = importlib.import_module(module_name)
        path = write(out_dir / module.NAME, module.CELLS)
        all_problems += check(path)

    print()
    stamp_site(slug)
    if all_problems:
        print(f"{len(all_problems)} problem(s):")
        for p in all_problems:
            print(f"  {p}")
        return 1
    print(f"{len(MODULES)} notebooks generated, all cells parse cleanly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
