/* Check the site has no horizontal overflow at narrow widths.
 *
 *   node tools/check_responsive.mjs
 *
 * A page that scrolls sideways on a phone is broken, and it is the easiest
 * regression to ship without noticing because it is invisible on a laptop.
 * The usual cause is a grid or flex child with the default min-width:auto
 * refusing to shrink below a wide table or code block.
 *
 * Drives headless Chrome, so it needs Chrome installed; it skips cleanly if
 * Chrome is not found. Note that headless Chrome clamps the layout viewport
 * to a minimum of ~500px, so that is the narrowest width we can actually
 * measure here; the CSS still has to be correct below it.
 */

import { spawn, spawnSync } from "node:child_process";
import { mkdtempSync, cpSync, appendFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const DOCS = join(ROOT, "docs");
const WIDTHS = [500, 760, 1024];

const CHROME_CANDIDATES = [
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  "/Applications/Chromium.app/Contents/MacOS/Chromium",
  "google-chrome", "chromium", "chromium-browser",
];

function findChrome() {
  for (const c of CHROME_CANDIDATES) {
    const probe = spawnSync(c, ["--version"], { stdio: "ignore" });
    if (!probe.error) return c;
  }
  return null;
}

const PROBE = `
<script type="module">
setTimeout(() => {
  const vw = document.documentElement.clientWidth;
  const inScroller = (el) => {
    for (let p = el.parentElement; p; p = p.parentElement) {
      const ox = getComputedStyle(p).overflowX;
      if (ox === "auto" || ox === "scroll" || ox === "hidden") return true;
    }
    return false;
  };
  const bad = [];
  for (const el of document.querySelectorAll("body *")) {
    const r = el.getBoundingClientRect();
    if (r.width === 0 || inScroller(el)) continue;
    if (r.right > vw + 1 || r.left < -1) {
      const cls = typeof el.className === "string" && el.className
        ? "." + el.className.trim().split(/\\s+/).join(".") : "";
      bad.push(el.tagName.toLowerCase() + (el.id ? "#" + el.id : "") + cls +
               " w=" + Math.round(r.width));
    }
  }
  document.title = "RESULT " + JSON.stringify({
    vw, scrollW: document.documentElement.scrollWidth,
    offenders: [...new Set(bad)].slice(0, 10),
  });
}, 2500);
</script>
`;

const chrome = findChrome();
if (!chrome) {
  console.log("Chrome not found -- skipping the responsive check.");
  process.exit(0);
}

// Serve a copy of docs/ with the probe appended, so the real file is untouched.
const work = mkdtempSync(join(tmpdir(), "site-resp-"));
cpSync(DOCS, work, { recursive: true });
appendFileSync(join(work, "index.html"), PROBE);

const PORT = 8731;
const server = spawn(process.execPath, ["-e", `
  const http = require("http"), fs = require("fs"), path = require("path");
  const types = { ".html":"text/html", ".css":"text/css", ".js":"text/javascript",
                  ".json":"application/json", ".svg":"image/svg+xml" };
  http.createServer((req, res) => {
    const p = path.join(${JSON.stringify(work)}, req.url.split("?")[0] === "/" ? "index.html" : req.url.split("?")[0]);
    fs.readFile(p, (err, buf) => {
      if (err) { res.writeHead(404); return res.end(); }
      res.writeHead(200, { "Content-Type": types[path.extname(p)] || "text/plain" });
      res.end(buf);
    });
  }).listen(${PORT});
`], { stdio: "ignore" });

await new Promise((r) => setTimeout(r, 700));

let failures = 0;
try {
  for (const width of WIDTHS) {
    const out = spawnSync(chrome, [
      "--headless=new", "--disable-gpu", "--no-sandbox",
      "--virtual-time-budget=9000", `--window-size=${width},900`, "--dump-dom",
      `http://127.0.0.1:${PORT}/index.html`,
    ], { encoding: "utf8", maxBuffer: 64 * 1024 * 1024 });

    const m = (out.stdout || "").match(/<title>RESULT ([^<]*)<\/title>/);
    if (!m) {
      console.log(`  FAIL ${width}px: probe did not report (page error?)`);
      failures++;
      continue;
    }
    const r = JSON.parse(m[1].replace(/&quot;/g, '"').replace(/&amp;/g, "&"));
    const overflow = r.scrollW - r.vw;
    if (overflow > 1) {
      console.log(`  FAIL layout ${r.vw}px: scrollWidth ${r.scrollW} ` +
                  `(+${overflow}px of horizontal scroll)`);
      for (const o of r.offenders) console.log(`         ${o}`);
      failures++;
    } else {
      console.log(`  ok   layout ${r.vw}px: no horizontal overflow ` +
                  `(scrollWidth ${r.scrollW})`);
    }
  }
} finally {
  server.kill();
  rmSync(work, { recursive: true, force: true });
}

console.log(failures ? `\n${failures} WIDTH(S) OVERFLOW` : "\nNo horizontal overflow at any tested width.");
process.exit(failures ? 1 : 0);
