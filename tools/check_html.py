"""Structural checks on the GitHub Pages site.

    python tools/check_html.py

Catches the things that break a static site quietly: a nav link pointing at an
id that does not exist, a stylesheet or module path that 404s on Pages, a
figure without an accessible label, an SVG without a viewBox (so it will not
scale on a phone).
"""

from __future__ import annotations

import re
import sys
from html.parser import HTMLParser
from pathlib import Path

DOCS = Path(__file__).resolve().parent.parent / "docs"

# Void elements never need closing.
VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link",
        "meta", "param", "source", "track", "wbr"}


class Collector(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[tuple[str, int]] = []
        self.errors: list[str] = []
        self.ids: set[str] = set()
        self.hrefs: list[str] = []
        self.assets: list[str] = []
        self.svgs: list[dict] = []
        self.figures = 0
        self.figcaptions = 0
        self._in_svg = False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if "id" in a:
            if a["id"] in self.ids:
                self.errors.append(f"duplicate id: {a['id']}")
            self.ids.add(a["id"])
        if tag == "a" and a.get("href"):
            self.hrefs.append(a["href"])
        if tag == "link" and a.get("href"):
            self.assets.append(a["href"])
        if tag == "script" and a.get("src"):
            self.assets.append(a["src"])
        if tag == "figure":
            self.figures += 1
        if tag == "figcaption":
            self.figcaptions += 1
        if tag == "svg" and not self._in_svg:
            self._in_svg = True
            # HTMLParser lowercases attribute names, so viewBox arrives as
            # "viewbox". Checking the camelCase spelling silently fails
            # everything -- which it did, the first time this ran.
            self.svgs.append({"viewBox": a.get("viewbox") or a.get("viewBox"),
                              "role": a.get("role"),
                              "aria-label": a.get("aria-label"),
                              "line": self.getpos()[0]})
        if tag not in VOID:
            self.stack.append((tag, self.getpos()[0]))

    def handle_startendtag(self, tag, attrs):
        pass        # self-closing: nothing to balance

    def handle_endtag(self, tag):
        if tag == "svg":
            self._in_svg = False
        if tag in VOID:
            return
        if not self.stack:
            self.errors.append(f"</{tag}> with nothing open (line {self.getpos()[0]})")
            return
        if self.stack[-1][0] == tag:
            self.stack.pop()
        else:
            open_tag, line = self.stack[-1]
            self.errors.append(
                f"</{tag}> at line {self.getpos()[0]} closes <{open_tag}> opened "
                f"at line {line}")
            # Recover: pop until we find it, if it is anywhere open.
            for i in range(len(self.stack) - 1, -1, -1):
                if self.stack[i][0] == tag:
                    del self.stack[i:]
                    break


def main() -> int:
    index = DOCS / "index.html"
    if not index.exists():
        print(f"FAIL: {index} does not exist")
        return 1

    text = index.read_text()
    p = Collector()
    p.feed(text)
    p.close()

    failures = 0

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal failures
        print(f"  {'ok  ' if ok else 'FAIL'} {name}{'  ' + detail if detail else ''}")
        if not ok:
            failures += 1

    print("structure")
    check("tags balanced", not p.errors and not p.stack,
          "; ".join(p.errors[:4]) or (f"unclosed: {[t for t, _ in p.stack]}"
                                      if p.stack else ""))

    print("\ninternal anchors resolve")
    frags = [h[1:] for h in p.hrefs if h.startswith("#")]
    missing = sorted(set(frags) - p.ids)
    check(f"{len(frags)} nav/in-page links", not missing, f"missing ids: {missing}")

    print("\nreferenced assets exist")
    for asset in p.assets:
        if asset.startswith(("http://", "https://", "data:")):
            continue
        check(asset, (DOCS / asset).exists())

    print("\nJS module imports resolve")
    for js in sorted(DOCS.glob("assets/*.js")):
        for spec in re.findall(r'from\s+"(\./[^"]+)"', js.read_text()):
            target = (js.parent / spec[2:]).resolve()
            check(f"{js.name} imports {spec}", target.exists())

    print("\nelement ids the JS requires")
    required = set()
    for js in sorted(DOCS.glob("assets/*.js")):
        required |= set(re.findall(r'\$\("([^"]+)"\)', js.read_text()))
    absent = sorted(required - p.ids)
    check(f"{len(required)} ids looked up by site.js", not absent,
          f"absent from the HTML: {absent}")

    print("\ndiagram accessibility")
    check(f"{len(p.svgs)} inline SVGs all have a viewBox",
          all(s["viewBox"] for s in p.svgs),
          f"missing on lines {[s['line'] for s in p.svgs if not s['viewBox']]}")
    check("all have role=img",
          all(s["role"] == "img" for s in p.svgs),
          f"missing on lines {[s['line'] for s in p.svgs if s['role'] != 'img']}")
    check("all have a non-trivial aria-label",
          all(s["aria-label"] and len(s["aria-label"]) > 25 for s in p.svgs),
          f"missing/short on lines "
          f"{[s['line'] for s in p.svgs if not s['aria-label'] or len(s['aria-label']) <= 25]}")
    check("every <figure> has a <figcaption>", p.figures == p.figcaptions,
          f"{p.figures} figures, {p.figcaptions} captions")

    print("\ntheme contract")
    css = (DOCS / "assets" / "site.css").read_text()
    check("dark mode declared under the OS media query",
          "prefers-color-scheme: dark" in css)
    check("dark mode also declared under a data-theme scope",
          'data-theme="dark"' in css)
    check("light stamp can beat OS dark",
          'not([data-theme="light"])' in css)
    check("body has an explicit background", re.search(r"body\s*\{[^}]*background", css))

    print("\nGitHub Pages specifics")
    check(".nojekyll present", (DOCS / ".nojekyll").exists())
    check("data/results.json present", (DOCS / "data" / "results.json").exists())
    # Check real href attributes, not the raw text -- the file explains this
    # very hazard in a comment, which a substring search flags as the hazard.
    bad_links = [h for h in p.hrefs if h.startswith("../")]
    check("no relative ../ hrefs that would 404 on Pages", not bad_links,
          f"{bad_links} -- use data-nb= so site.js builds a github.com URL")
    check("notebook links use data-nb", 'data-nb="' in text)

    print()
    if failures:
        print(f"{failures} CHECK(S) FAILED")
        return 1
    print("ALL HTML CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
