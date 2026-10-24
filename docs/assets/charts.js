/* Minimal SVG charting for the inference-engine study site.
 *
 * Hand-rolled rather than pulled from a CDN: the charts needed here are a
 * line, a scatter and a bar, and writing them directly keeps the page
 * dependency-free, keeps the palette identical to bench/plots.py, and lets
 * every chart carry a table view.
 *
 * Conventions enforced here, matching the notebooks:
 *   - colour follows the ENTITY (engine), never its rank
 *   - every chart with >= 2 series gets a legend
 *   - scatter marks also vary shape, so colour is never the only channel
 *   - never two y-axes: a second measure gets its own chart
 *   - hairline solid grid, no dashes
 */

const NS = "http://www.w3.org/2000/svg";

export const ENGINE_META = {
  vllm:          { label: "vLLM",             color: "var(--series-1)", shape: "circle" },
  sglang:        { label: "SGLang",           color: "var(--series-2)", shape: "square" },
  llamacpp:      { label: "llama.cpp",        color: "var(--series-3)", shape: "triangle" },
  tensorrt_llm:  { label: "TensorRT-LLM",     color: "var(--series-4)", shape: "diamond" },
  triton:        { label: "Triton Server",    color: "var(--series-5)", shape: "square" },
  onnxruntime:   { label: "ONNX Runtime",     color: "var(--series-6)", shape: "circle" },
  torch_compile: { label: "torch.compile",    color: "var(--series-7)", shape: "diamond" },
  hf:            { label: "HF transformers",  color: "var(--series-baseline)", shape: "circle" },
};

export const engineLabel = (k) => (ENGINE_META[k] || { label: k }).label;
export const engineColor = (k) => (ENGINE_META[k] || { color: "var(--series-1)" }).color;

function el(name, attrs = {}, parent = null) {
  const node = document.createElementNS(NS, name);
  for (const [k, v] of Object.entries(attrs)) {
    if (v !== null && v !== undefined) node.setAttribute(k, v);
  }
  if (parent) parent.appendChild(node);
  return node;
}

export function fmt(v, digits) {
  if (v === null || v === undefined || Number.isNaN(v)) return "-";
  const a = Math.abs(v);
  if (digits !== undefined) return v.toFixed(digits);
  if (a >= 1000) return Math.round(v).toLocaleString();
  if (a >= 100) return v.toFixed(0);
  if (a >= 10) return v.toFixed(1);
  if (a >= 1) return v.toFixed(2);
  return v.toFixed(3);
}

function scale(domain, range, log) {
  const [d0, d1] = domain;
  const [r0, r1] = range;
  if (log) {
    const l0 = Math.log10(Math.max(d0, 1e-9));
    const l1 = Math.log10(Math.max(d1, 1e-9));
    return (v) => r0 + ((Math.log10(Math.max(v, 1e-9)) - l0) / (l1 - l0 || 1)) * (r1 - r0);
  }
  return (v) => r0 + ((v - d0) / (d1 - d0 || 1)) * (r1 - r0);
}

function niceTicks(min, max, count = 5) {
  if (!Number.isFinite(min) || !Number.isFinite(max)) return [0, 1];
  if (min === max) return [min];
  const span = max - min;
  const raw = span / count;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = (norm >= 7.5 ? 10 : norm >= 3.5 ? 5 : norm >= 1.5 ? 2 : 1) * mag;
  const out = [];
  for (let t = Math.ceil(min / step) * step; t <= max + step * 0.001; t += step) {
    out.push(Math.abs(t) < step * 1e-9 ? 0 : t);
  }
  return out;
}

function logTicks(min, max) {
  const out = [];
  for (let e = Math.floor(Math.log10(Math.max(min, 1e-9)));
       e <= Math.ceil(Math.log10(Math.max(max, 1e-9))); e++) {
    for (const m of [1, 2, 5]) {
      const v = m * Math.pow(10, e);
      if (v >= min * 0.95 && v <= max * 1.05) out.push(v);
    }
  }
  return out.length >= 2 ? out : niceTicks(min, max);
}

function marker(parent, shape, cx, cy, color, size = 5) {
  const common = { fill: color, stroke: "var(--surface)", "stroke-width": 1.5 };
  if (shape === "square") {
    return el("rect", { x: cx - size, y: cy - size, width: size * 2, height: size * 2,
                        rx: 1.5, ...common }, parent);
  }
  if (shape === "triangle") {
    const s = size * 1.2;
    return el("polygon", { points: `${cx},${cy - s} ${cx + s},${cy + s * 0.8} ${cx - s},${cy + s * 0.8}`,
                           ...common }, parent);
  }
  if (shape === "diamond") {
    const s = size * 1.25;
    return el("polygon", { points: `${cx},${cy - s} ${cx + s},${cy} ${cx},${cy + s} ${cx - s},${cy}`,
                           ...common }, parent);
  }
  return el("circle", { cx, cy, r: size, ...common }, parent);
}

/* ---------------------------------------------------------------- tooltip */

function attachTooltip(body) {
  let tip = body.querySelector(".tooltip");
  if (!tip) {
    tip = document.createElement("div");
    tip.className = "tooltip";
    body.appendChild(tip);
  }
  return {
    show(html, ev) {
      tip.innerHTML = html;
      const rect = body.getBoundingClientRect();
      let x = ev.clientX - rect.left + 14;
      const y = ev.clientY - rect.top - 10;
      if (x + tip.offsetWidth > rect.width) x = ev.clientX - rect.left - tip.offsetWidth - 14;
      tip.style.left = `${Math.max(0, x)}px`;
      tip.style.top = `${Math.max(0, y)}px`;
      tip.style.opacity = "1";
    },
    hide() { tip.style.opacity = "0"; },
  };
}

/* ------------------------------------------------------------ line/scatter */

/**
 * series: [{ key, label, color, shape, points: [{x, y, label?}] }]
 * opts:   { xLabel, yLabel, xLog, yLog, connect, annotateLast, yZero }
 */
export function lineChart(container, series, opts = {}) {
  const W = 760, H = opts.height || 340;
  const M = { top: 14, right: opts.annotateLast ? 58 : 18, bottom: 46, left: 62 };
  container.innerHTML = "";

  const visible = series.filter((s) => s.points.length && !s.hidden);
  if (!visible.length) {
    container.innerHTML = '<div class="empty-state">No data for this selection.</div>';
    return;
  }

  const xs = visible.flatMap((s) => s.points.map((p) => p.x));
  const ys = visible.flatMap((s) => s.points.map((p) => p.y));
  let xMin = Math.min(...xs), xMax = Math.max(...xs);
  let yMin = Math.min(...ys), yMax = Math.max(...ys);

  if (opts.yZero && !opts.yLog) yMin = Math.min(0, yMin);
  if (!opts.yLog) {
    const pad = (yMax - yMin) * 0.1 || Math.abs(yMax) * 0.1 || 1;
    yMax += pad;
    if (!opts.yZero) yMin -= pad;
  } else {
    yMin *= 0.75; yMax *= 1.35;
  }
  if (xMin === xMax) { xMin *= 0.9; xMax *= 1.1 || 1; }

  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img",
                          "aria-label": opts.ariaLabel || opts.yLabel || "chart" }, container);

  const x = scale([xMin, xMax], [M.left, W - M.right], opts.xLog);
  const y = scale([yMin, yMax], [H - M.bottom, M.top], opts.yLog);

  const xt = opts.xLog ? logTicks(xMin, xMax) : niceTicks(xMin, xMax, 6);
  const yt = opts.yLog ? logTicks(yMin, yMax) : niceTicks(yMin, yMax, 5);

  // grid + axes: solid hairlines, one shade off the surface
  for (const t of yt) {
    el("line", { x1: M.left, x2: W - M.right, y1: y(t), y2: y(t),
                 stroke: "var(--grid)", "stroke-width": 1 }, svg);
    el("text", { x: M.left - 9, y: y(t) + 4, "text-anchor": "end", "font-size": 11,
                 fill: "var(--text-muted)" }, svg).textContent = fmt(t);
  }
  for (const t of xt) {
    el("text", { x: x(t), y: H - M.bottom + 18, "text-anchor": "middle", "font-size": 11,
                 fill: "var(--text-muted)" }, svg).textContent = fmt(t);
  }
  el("line", { x1: M.left, x2: W - M.right, y1: H - M.bottom, y2: H - M.bottom,
               stroke: "var(--axis)", "stroke-width": 1 }, svg);

  el("text", { x: (M.left + W - M.right) / 2, y: H - 6, "text-anchor": "middle",
               "font-size": 12, fill: "var(--text-secondary)" }, svg)
    .textContent = opts.xLabel || "";
  el("text", { x: 14, y: (M.top + H - M.bottom) / 2, "font-size": 12,
               fill: "var(--text-secondary)", "text-anchor": "middle",
               transform: `rotate(-90 14 ${(M.top + H - M.bottom) / 2})` }, svg)
    .textContent = opts.yLabel || "";

  const tip = attachTooltip(container.closest(".chart-body") || container);

  for (const s of visible) {
    const pts = [...s.points].sort((a, b) => a.x - b.x);
    if (opts.connect !== false && pts.length > 1) {
      el("polyline", {
        points: pts.map((p) => `${x(p.x)},${y(p.y)}`).join(" "),
        fill: "none", stroke: s.color, "stroke-width": 2,
        "stroke-linejoin": "round", "stroke-linecap": "round",
      }, svg);
    }
    // shape "none" draws a bare line -- used for reference curves such as the
    // roofline itself, where a marker per sample would be noise.
    if (s.shape !== "none") {
      for (const p of pts) {
        marker(svg, s.shape || "circle", x(p.x), y(p.y), s.color);
        // Hit target larger than the mark, per the interaction rules.
        const hit = el("circle", { cx: x(p.x), cy: y(p.y), r: 13, fill: "transparent",
                                   style: "cursor:pointer" }, svg);
        hit.addEventListener("mousemove", (ev) => tip.show(
          `<div class="tt-title">${s.label}</div>` +
          `<div class="tt-row">${opts.xLabel || "x"}: <b>${fmt(p.x)}</b></div>` +
          `<div class="tt-row">${opts.yLabel || "y"}: <b>${fmt(p.y)}</b></div>` +
          (p.label ? `<div class="tt-row">${p.label}</div>` : ""), ev));
        hit.addEventListener("mouseleave", tip.hide);
      }
    }
    if (opts.annotateLast && pts.length) {
      const last = pts[pts.length - 1];
      el("text", { x: x(last.x) + 10, y: y(last.y) + 4, "font-size": 11,
                   fill: "var(--text-muted)" }, svg).textContent = last.endLabel || s.label;
    }
  }
}

/* -------------------------------------------------------------------- bars */

/** rows: [{ label, value, color, note? }] */
export function barChart(container, rows, opts = {}) {
  container.innerHTML = "";
  if (!rows.length) {
    container.innerHTML = '<div class="empty-state">No data for this selection.</div>';
    return;
  }
  const horizontal = opts.horizontal !== false;
  const W = 760;
  const band = 34;
  const M = { top: 10, right: 110, bottom: 40, left: opts.labelWidth || 150 };
  const H = M.top + M.bottom + rows.length * band;

  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img",
                          "aria-label": opts.ariaLabel || opts.xLabel || "bar chart" }, container);

  const maxV = Math.max(...rows.map((r) => r.value), 0);
  const x = scale([0, maxV * 1.02 || 1], [M.left, W - M.right], false);

  for (const t of niceTicks(0, maxV * 1.02 || 1, 5)) {
    el("line", { x1: x(t), x2: x(t), y1: M.top, y2: H - M.bottom,
                 stroke: "var(--grid)", "stroke-width": 1 }, svg);
    el("text", { x: x(t), y: H - M.bottom + 17, "text-anchor": "middle",
                 "font-size": 11, fill: "var(--text-muted)" }, svg).textContent = fmt(t);
  }

  rows.forEach((r, i) => {
    const cy = M.top + i * band + band / 2;
    const h = 17;
    el("text", { x: M.left - 11, y: cy + 4, "text-anchor": "end", "font-size": 12.5,
                 fill: "var(--text-secondary)" }, svg).textContent = r.label;
    // 4px rounded data-end, anchored to the baseline.
    el("rect", { x: M.left, y: cy - h / 2, width: Math.max(1, x(r.value) - M.left),
                 height: h, rx: 4, fill: r.color || "var(--series-1)" }, svg);
    el("text", { x: x(r.value) + 9, y: cy + 4, "font-size": 12,
                 fill: "var(--text-secondary)" }, svg)
      .textContent = (opts.valuePrefix || "") + fmt(r.value, opts.digits) + (opts.unit || "");
  });

  el("line", { x1: M.left, x2: M.left, y1: M.top, y2: H - M.bottom,
               stroke: "var(--axis)", "stroke-width": 1 }, svg);
  if (opts.xLabel) {
    el("text", { x: (M.left + W - M.right) / 2, y: H - 6, "text-anchor": "middle",
                 "font-size": 12, fill: "var(--text-secondary)" }, svg).textContent = opts.xLabel;
  }
}

/* ------------------------------------------------------------ legend/table */

export function renderLegend(container, series, onToggle) {
  container.innerHTML = "";
  if (series.length < 2) return;        // one series -> the title names it
  for (const s of series) {
    const item = document.createElement("div");
    item.className = "legend-item" + (s.hidden ? " off" : "");
    item.innerHTML = `<span class="dot" style="background:${s.color}"></span>${s.label}`;
    if (onToggle) {
      item.addEventListener("click", () => { s.hidden = !s.hidden; onToggle(); });
    }
    container.appendChild(item);
  }
}

/** Every chart gets a table twin -- values must never be hover-only. */
export function renderTable(container, columns, rows) {
  const head = columns.map((c) => `<th class="${c.num ? "num" : ""}">${c.label}</th>`).join("");
  const body = rows.map((r) =>
    "<tr>" + columns.map((c) => {
      const v = r[c.key];
      const text = c.format ? c.format(v, r) : (typeof v === "number" ? fmt(v) : (v ?? "-"));
      return `<td class="${c.num ? "num" : ""}">${text}</td>`;
    }).join("") + "</tr>").join("");
  container.innerHTML = `<div class="table-scroll"><table><thead><tr>${head}</tr></thead>` +
                        `<tbody>${body}</tbody></table></div>`;
}
