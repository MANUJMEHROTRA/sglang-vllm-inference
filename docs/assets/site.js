/* Interactive logic for the inference-engine study site.
 *
 * Everything here is derived, not asserted: the roofline and KV-cache panels
 * compute from the same formulas as bench/metrics.py, so the page and the
 * notebooks cannot drift apart. The results section reads whatever
 * data/results.json contains and degrades gracefully when it is empty.
 */

import { barChart, engineColor, engineLabel, ENGINE_META, fmt, lineChart,
         renderLegend, renderTable } from "./charts.js";
import { BATCHES, CONTEXTS, GB, GPUS, MODELS, decodeCeilingTokS, kvBudgetGb,
         kvBytes, params, prefillFlops, ridgePoint, roofAt } from "./models.js";

/* ================================================================== helpers */

const $ = (id) => document.getElementById(id);

/* GitHub Pages serves docs/ as the site root, so a relative ../notebooks/ link
   does not exist on the published site. Point notebook links at github.com
   when a real repo slug has been stamped in, and fall back to the relative
   path for local browsing. */
const REPO = document.body.dataset.repo || "";
const HAS_REPO = REPO && !REPO.startsWith("YOUR-GITHUB");

function nbUrl(file) {
  return HAS_REPO
    ? `https://github.com/${REPO}/blob/main/notebooks/${file}`
    : `../notebooks/${file}`;
}

for (const a of document.querySelectorAll("a[data-nb]")) {
  a.href = nbUrl(a.dataset.nb);
}

function fillSelect(sel, entries, selected) {
  sel.innerHTML = entries
    .map(([v, label]) => `<option value="${v}"${v === selected ? " selected" : ""}>${label}</option>`)
    .join("");
}

function stat(label, value, sub) {
  return `<div class="stat"><div class="label">${label}</div>` +
         `<div class="value">${value}</div>` +
         (sub ? `<div class="sub">${sub}</div>` : "") + "</div>";
}

/* ================================================================== theme */

(function theme() {
  const btn = $("theme-toggle");
  const stored = (() => {
    try { return localStorage.getItem("theme"); } catch { return null; }
  })();
  if (stored) document.documentElement.dataset.theme = stored;

  btn.addEventListener("click", () => {
    const current = document.documentElement.dataset.theme
      || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    const next = current === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("theme", next); } catch { /* private mode */ }
    document.dispatchEvent(new CustomEvent("themechange"));
  });
})();

/* ============================================================== table toggles */

for (const btn of document.querySelectorAll("[data-table]")) {
  btn.addEventListener("click", () => {
    const target = $(btn.dataset.table);
    target.hidden = !target.hidden;
    btn.textContent = target.hidden ? "Table" : "Chart only";
  });
}

/* =============================================================== roofline */

(function roofline() {
  const gpuSel = $("rl-gpu");
  const batchR = $("rl-batch");
  const ctxR = $("rl-ctx");
  fillSelect(gpuSel, Object.entries(GPUS).map(([k, v]) => [k, v.label]), "t4");

  function render() {
    const gpu = GPUS[gpuSel.value];
    const batch = BATCHES[+batchR.value];
    const ctx = CONTEXTS[+ctxR.value];
    $("rl-batch-v").textContent = batch;
    $("rl-ctx-v").textContent = ctx.toLocaleString();

    const m = MODELS["qwen2.5-0.5b"];
    const p = params(m);
    const ridge = ridgePoint(gpu);
    $("rl-ridge").textContent =
      `ridge point ${fmt(ridge, 0)} FLOP/byte · ${gpu.bwGbs} GB/s · ${gpu.tflops} TFLOP/s`;

    // The roof itself: min(peak, intensity * bandwidth)
    const roof = [];
    for (let e = -1; e <= 4; e += 0.1) {
      const ai = Math.pow(10, e);
      roof.push({ x: ai, y: roofAt(gpu, ai) });
    }

    // Decode: weights read once per step, KV per sequence.
    const decodePts = BATCHES.map((bs) => {
      const ai = (2 * p * bs) / (p * 2 + kvBytes(m, ctx, 2) * bs);
      return { x: ai, y: roofAt(gpu, ai), label: `batch ${bs}` };
    });

    // Prefill of one `ctx`-token prompt: dense term plus quadratic attention.
    const preAi = prefillFlops(m, ctx) / (p * 2 + kvBytes(m, ctx, 2));

    const current = decodePts[BATCHES.indexOf(batch)];
    const series = [
      { key: "roof", label: "hardware roofline", color: "var(--text-muted)",
        shape: "none", points: roof },
      { key: "decode", label: "decode (all batch sizes)", color: "var(--series-1)",
        shape: "circle", points: decodePts },
      { key: "prefill", label: `prefill (${ctx.toLocaleString()} tokens)`,
        color: "var(--series-2)", shape: "diamond",
        points: [{ x: preAi, y: roofAt(gpu, preAi),
                   label: "one pass over the whole prompt" }] },
    ];

    lineChart($("rl-chart"), series, {
      xLabel: "arithmetic intensity (FLOP/byte)",
      yLabel: "achievable TFLOP/s",
      xLog: true, yLog: true, height: 330,
      ariaLabel: "Roofline chart: decode sits far left on the memory roof, prefill sits near the ridge point.",
    });
    renderLegend($("rl-legend"), series);

    const pctFlops = (100 * current.y) / gpu.tflops;
    $("rl-readout").innerHTML =
      stat(`Decode at batch ${batch}`, `${fmt(current.x, 1)}`, "FLOP per byte") +
      stat("Of peak arithmetic", `${pctFlops < 1 ? pctFlops.toFixed(2) : pctFlops.toFixed(0)}%`,
           pctFlops < 10 ? "memory-bound: move fewer bytes" : "approaching the ridge") +
      stat("Prefill intensity", `${fmt(preAi, 0)}`,
           preAi > ridge ? "compute-bound: past the ridge" : "still under the memory roof") +
      stat("Decode ceiling", `${fmt(decodeCeilingTokS(m, gpu, batch, ctx), 0)}`,
           "tok/s at this batch (hard limit)");
  }

  gpuSel.addEventListener("change", render);
  batchR.addEventListener("input", render);
  ctxR.addEventListener("input", render);
  document.addEventListener("themechange", render);
  render();
})();

/* ============================================================== KV cache */

(function kvCache() {
  const modelSel = $("kv-model");
  const gpuSel = $("kv-gpu");
  const ctxR = $("kv-ctx");
  const dtypeSel = $("kv-dtype");
  const wdtypeSel = $("kv-wdtype");

  fillSelect(modelSel, Object.entries(MODELS).map(([k, v]) => [k, v.label]), "qwen2.5-0.5b");
  fillSelect(gpuSel, Object.entries(GPUS).map(([k, v]) => [k, `${v.label} — ${v.memGb} GB`]), "t4");

  const OVERHEAD_GB = 1.2;   // CUDA context, activations, fragmentation

  function render() {
    const m = MODELS[modelSel.value];
    const gpu = GPUS[gpuSel.value];
    const ctx = CONTEXTS[+ctxR.value];
    const kvB = parseFloat(dtypeSel.value);
    const wB = parseFloat(wdtypeSel.value);
    $("kv-ctx-v").textContent = ctx.toLocaleString();

    const { weightsGb, kvGb } = kvBudgetGb(m, gpu, wB, 0.90, OVERHEAD_GB);
    const perSeq = kvBytes(m, ctx, kvB);
    const fits = perSeq > 0 ? Math.floor((kvGb * GB) / perSeq) : 0;
    const gqa = m.heads / m.kvHeads;

    const doesntFit = kvGb <= 0;
    $("kv-readout").innerHTML =
      stat("Parameters", `${fmt(params(m) / 1e9, 2)}B`,
           `weights ${fmt(weightsGb, 2)} GB at ${wB} bytes/param`) +
      stat("KV per sequence", `${fmt(perSeq / 1024 ** 2, 1)} MB`,
           `at ${ctx.toLocaleString()} tokens, ${kvB} bytes/value`) +
      stat("Budget for KV", doesntFit ? "none" : `${fmt(kvGb, 1)} GB`,
           doesntFit ? "the weights do not fit on this GPU" : `after weights + ~${OVERHEAD_GB} GB overhead`) +
      stat("Concurrent sequences", doesntFit ? "0" : fits.toLocaleString(),
           `GQA ratio ${gqa}× — without it, ${doesntFit ? "0" : Math.floor(fits / gqa).toLocaleString()}`);

    const points = CONTEXTS.map((c) => {
      const per = kvBytes(m, c, kvB);
      return { x: c, y: Math.max(0.3, Math.floor((kvGb * GB) / per)),
               label: `${fmt(per / 1024 ** 2, 1)} MB/seq` };
    });
    const series = [{ key: "fits", label: `${m.label} on ${gpu.label}`,
                      color: "var(--series-1)", shape: "circle", points }];

    lineChart($("kv-chart"), series, {
      xLabel: "context length per sequence (tokens)",
      yLabel: "sequences that fit",
      xLog: true, yLog: true, height: 300,
      ariaLabel: "Concurrent sequences that fit in the KV cache budget, falling as context length grows.",
    });

    renderTable($("kv-table"),
      [{ key: "ctx", label: "Context", num: true, format: (v) => v.toLocaleString() },
       { key: "per", label: "KV / sequence (MB)", num: true, format: (v) => fmt(v, 1) },
       { key: "fits", label: "Concurrent sequences", num: true,
         format: (v) => Math.floor(v).toLocaleString() },
       { key: "noGqa", label: "Without GQA", num: true,
         format: (v) => Math.floor(v).toLocaleString() }],
      CONTEXTS.map((c) => {
        const per = kvBytes(m, c, kvB);
        const n = Math.floor((kvGb * GB) / per);
        return { ctx: c, per: per / 1024 ** 2, fits: n, noGqa: n / gqa };
      }));
  }

  for (const node of [modelSel, gpuSel, dtypeSel, wdtypeSel]) {
    node.addEventListener("change", render);
  }
  ctxR.addEventListener("input", render);
  document.addEventListener("themechange", render);
  render();
})();

/* ================================================================ results */

const METRIC_META = {
  output_throughput_tok_s: { label: "output tok/s", scale: 1, better: "high" },
  ttft_s_p99: { label: "p99 TTFT (ms)", scale: 1000, better: "low" },
  ttft_s_p50: { label: "p50 TTFT (ms)", scale: 1000, better: "low" },
  tpot_s_mean: { label: "mean TPOT (ms)", scale: 1000, better: "low" },
  itl_s_p99: { label: "p99 ITL (ms)", scale: 1000, better: "low" },
  goodput_req_s: { label: "goodput (req/s)", scale: 1, better: "high" },
  usd_per_mtok: { label: "USD / 1M output tokens", scale: 1, better: "low" },
};

(async function results() {
  const status = $("results-status");
  let data;
  try {
    const resp = await fetch("data/results.json", { cache: "no-store" });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    data = await resp.json();
  } catch (err) {
    status.innerHTML = `<div class="empty-state">
      <p><b>No results loaded.</b> ${String(err.message || err)}</p>
      <p>Run the notebooks, then <code>python -m bench.aggregate</code> to generate
      <code>docs/data/results.json</code>.</p>
      <p style="font-size:13px">Opening this file directly from disk also blocks the fetch;
      serve it with <code>python -m http.server</code> from the <code>docs/</code> folder,
      or view it on GitHub Pages.</p></div>`;
    return;
  }

  const gen = (data.runs || []).filter(
    (r) => r.task === "generative" && r.mode === "closed" && r.concurrency);
  if (!gen.length) {
    status.innerHTML = `<div class="empty-state">
      <p><b>results.json has no comparable generative runs yet.</b></p>
      <p>Run notebooks 01&ndash;04 on the same GPU, then
      <code>python -m bench.aggregate</code>.</p></div>`;
    return;
  }

  status.innerHTML = "";
  $("results-ui").hidden = false;

  const gpuSel = $("res-gpu");
  const wlSel = $("res-workload");
  const metricSel = $("res-metric");

  const counts = (key) => {
    const c = {};
    for (const r of gen) if (r[key]) c[r[key]] = (c[r[key]] || 0) + 1;
    return Object.entries(c).sort((a, b) => b[1] - a[1]);
  };
  const gpus = counts("gpu");
  const workloads = counts("workload");

  fillSelect(gpuSel, gpus.map(([g, n]) => [g, `${g} (${n})`]), gpus[0]?.[0]);
  fillSelect(wlSel, workloads.map(([w, n]) => [w, `${w} (${n})`]), workloads[0]?.[0]);

  let sweepSeries = [];
  let paretoSeries = [];

  function slice() {
    return gen.filter((r) => r.gpu === gpuSel.value && r.workload === wlSel.value);
  }

  function groupByEngine(rows) {
    const g = {};
    for (const r of rows) (g[r.engine] = g[r.engine] || []).push(r);
    return g;
  }

  function buildSeries(rows, metricKey) {
    const meta = METRIC_META[metricKey];
    return Object.entries(groupByEngine(rows)).map(([engine, rs]) => {
      const prev = sweepSeries.find((s) => s.key === engine);
      return {
        key: engine,
        label: engineLabel(engine) + (rs.some((r) => r.synthetic) ? " †" : ""),
        color: engineColor(engine),
        shape: (ENGINE_META[engine] || {}).shape || "circle",
        hidden: prev ? prev.hidden : false,
        points: rs
          .filter((r) => r[metricKey] !== null && r[metricKey] !== undefined)
          .map((r) => ({ x: r.concurrency, y: r[metricKey] * meta.scale,
                         label: `${r.dtype}${r.quantization ? " / " + r.quantization : ""}` }))
          .sort((a, b) => a.x - b.x),
      };
    }).filter((s) => s.points.length);
  }

  function drawSweep() {
    const metricKey = metricSel.value;
    const meta = METRIC_META[metricKey];
    sweepSeries = buildSeries(slice(), metricKey);
    $("sweep-title").textContent = `${meta.label} vs concurrency`;
    lineChart($("sweep-chart"), sweepSeries, {
      xLabel: "concurrent requests", yLabel: meta.label,
      xLog: true, yZero: meta.better === "high", height: 320,
      ariaLabel: `${meta.label} against concurrency, one line per engine.`,
    });
    renderLegend($("sweep-legend"), sweepSeries, drawSweep);

    const visible = sweepSeries.filter((s) => !s.hidden);
    renderTable($("sweep-table"),
      [{ key: "engine", label: "Engine" },
       { key: "concurrency", label: "Concurrency", num: true },
       { key: "value", label: meta.label, num: true, format: (v) => fmt(v) }],
      visible.flatMap((s) => s.points.map((p) => ({
        engine: s.label, concurrency: p.x, value: p.y }))));
  }

  function drawPareto() {
    const rows = slice();
    paretoSeries = Object.entries(groupByEngine(rows)).map(([engine, rs]) => ({
      key: engine,
      label: engineLabel(engine),
      color: engineColor(engine),
      shape: (ENGINE_META[engine] || {}).shape || "circle",
      points: rs
        .filter((r) => r.output_throughput_tok_s && r.ttft_s_p99)
        .map((r) => ({ x: r.output_throughput_tok_s, y: r.ttft_s_p99 * 1000,
                       label: `concurrency ${r.concurrency}` }))
        .sort((a, b) => a.x - b.x),
    })).filter((s) => s.points.length);

    lineChart($("pareto-chart"), paretoSeries, {
      xLabel: "output throughput (tok/s)", yLabel: "p99 TTFT (ms)",
      yLog: true, height: 320,
      ariaLabel: "Throughput against p99 time-to-first-token; each engine traces a curve as concurrency rises.",
    });
    renderLegend($("pareto-legend"), paretoSeries);
  }

  function drawBest() {
    const metricKey = metricSel.value;
    const meta = METRIC_META[metricKey];
    const rows = slice().filter((r) => r[metricKey] !== null && r[metricKey] !== undefined);
    const best = Object.entries(groupByEngine(rows)).map(([engine, rs]) => {
      const pick = rs.reduce((a, b) =>
        (meta.better === "high" ? b[metricKey] > a[metricKey] : b[metricKey] < a[metricKey]) ? b : a);
      return { label: `${engineLabel(engine)} (c=${pick.concurrency})`,
               value: pick[metricKey] * meta.scale,
               color: engineColor(engine),
               engine: engineLabel(engine), concurrency: pick.concurrency,
               dtype: pick.dtype, version: pick.engine_version };
    }).sort((a, b) => meta.better === "high" ? b.value - a.value : a.value - b.value);

    $("best-title").textContent =
      `Best ${meta.label} per engine (${meta.better === "high" ? "higher" : "lower"} is better)`;
    barChart($("best-chart"), best, { xLabel: meta.label, labelWidth: 190 });
    renderTable($("best-table"),
      [{ key: "engine", label: "Engine" },
       { key: "version", label: "Version" },
       { key: "dtype", label: "dtype" },
       { key: "concurrency", label: "Concurrency", num: true },
       { key: "value", label: meta.label, num: true, format: (v) => fmt(v) }],
      best);
  }

  function drawAll() { drawSweep(); drawPareto(); drawBest(); }

  for (const node of [gpuSel, wlSel, metricSel]) node.addEventListener("change", drawAll);
  document.addEventListener("themechange", drawAll);
  drawAll();

  const nSynth = data.counts?.synthetic || 0;
  $("results-meta").innerHTML =
    `${data.counts?.measured ?? gen.length} measured runs · generated ${data.generated_at_iso || "?"}` +
    ` · SLO for goodput: TTFT ≤ ${data.default_slo?.ttft_s ?? 1}s and p99 ITL ≤ ` +
    `${(data.default_slo?.p99_itl_s ?? 0.05) * 1000}ms` +
    (nSynth ? ` · <span class="synthetic-flag">† ${nSynth} illustrative (not measured)</span>` : "");
  $("footer-meta").textContent =
    `Results generated ${data.generated_at_iso || "?"} · schema v${data.schema_version || "?"}.`;
})();

/* ========================================================== decision tree */

(function decisionTree() {
  const NODES = {
    start: {
      q: "What does the model output?",
      options: [
        { label: "A label, a score, or a vector", next: "encoder",
          crumb: "fixed output set" },
        { label: "Open-ended text", next: "gpu", crumb: "open-ended text" },
      ],
    },
    gpu: {
      q: "Do you have an NVIDIA GPU?",
      options: [
        { label: "Yes", next: "fits", crumb: "NVIDIA GPU" },
        { label: "No — CPU, Mac, AMD or Intel", next: "llamacpp_only",
          crumb: "no NVIDIA GPU" },
      ],
    },
    fits: {
      q: "Does the model fit in VRAM?",
      options: [
        { label: "Yes", next: "users", crumb: "fits in VRAM" },
        { label: "No", next: "llamacpp_offload", crumb: "does not fit" },
      ],
    },
    users: {
      q: "How many concurrent users?",
      options: [
        { label: "One — local, private, interactive", next: "single",
          crumb: "single user" },
        { label: "Many — a real serving tier", next: "prefixes", crumb: "many users" },
      ],
    },
    prefixes: {
      q: "Do prompts share long prefixes, branch, or retry?",
      options: [
        { label: "Yes — agents, multi-turn, sampling trees", next: "sglang",
          crumb: "shared / branching prefixes" },
        { label: "No — mostly independent prompts", next: "latency",
          crumb: "independent prompts" },
      ],
    },
    latency: {
      q: "Is latency the product, with a stable model, fixed GPUs and CI for artifacts?",
      options: [
        { label: "Yes — all of those", next: "trtllm", crumb: "latency is the product" },
        { label: "No — I value flexibility and model coverage", next: "vllm",
          crumb: "flexibility matters" },
      ],
    },

    encoder: {
      answer: "Encoder + ONNX Runtime or TensorRT",
      color: "var(--series-6)",
      body: `<p>A 22M&ndash;300M encoder (BERT, DeBERTa-v3, E5, BGE) does
        <b>one forward pass per request</b> &mdash; no KV cache, no autoregression. It is
        10&ndash;100&times; cheaper than a decoder doing the same job, and it will run on
        a CPU.</p>
        <p><b>Serve it with</b> ONNX Runtime or TensorRT; put it behind Triton Inference
        Server if you host several models.</p>
        <p><b>Optimise in this order:</b> length-sorted batching (usually the biggest win
        and it is free) &rarr; truncate inputs &rarr; batch as hard as your latency budget
        allows &rarr; int8 dynamic quantization &rarr; only then a smaller model.</p>
        <p><b>Caveat:</b> an encoder needs labelled training data. If you have none, start
        with a decoder zero-shot, label with it, then distil.</p>`,
      notebooks: [["06", "06_encoder_models_bert.ipynb"], ["05", "05_triton_inference_server.ipynb"]],
    },
    llamacpp_only: {
      answer: "llama.cpp + GGUF",
      color: "var(--series-3)",
      body: `<p>The only engine here that runs on CPU, Apple Silicon, AMD and Intel.
        One static binary, one model file, no CUDA toolkit.</p>
        <p><b>Quant:</b> <code>Q4_K_M</code> unless a larger one fits comfortably. Prefer
        <b>a bigger model at Q4_K_M over a smaller model at F16</b> of the same file
        size, down to about 3 bits.</p>
        <p><b>Expect</b> excellent single-user latency and poor throughput under
        concurrency &mdash; that is the design, not a defect.</p>`,
      notebooks: [["03", "03_llamacpp_gguf.ipynb"]],
    },
    llamacpp_offload: {
      answer: "llama.cpp with partial offload — or quantize until it fits",
      color: "var(--series-3)",
      body: `<p>No other engine here will run a model larger than VRAM. Use
        <code>-ngl N</code> to place N layers on the GPU and the rest on CPU.</p>
        <p><b>Measure the offload curve first.</b> It is <em>convex</em>, not linear:
        the CPU half still runs at CPU speed, so most of the benefit arrives only when
        the model nearly fully fits.</p>
        <p><b>Therefore:</b> a quant level that <em>fully</em> fits usually beats a
        better quant that half-fits, even though its perplexity is worse.</p>`,
      notebooks: [["03", "03_llamacpp_gguf.ipynb"], ["00", "00_foundations_and_setup.ipynb"]],
    },
    single: {
      answer: "llama.cpp for simplicity, or vLLM for features",
      color: "var(--series-3)",
      body: `<p>At batch 1 the lean C++ path is competitive with the datacenter servers
        and sometimes better on TTFT for short prompts, with far less operational
        surface.</p>
        <p><b>Choose vLLM instead</b> if you want LoRA adapters, grammar-constrained
        output, multi-modal input, or an OpenAI-compatible API that other tooling already
        speaks.</p>`,
      notebooks: [["03", "03_llamacpp_gguf.ipynb"], ["01", "01_vllm_deep_dive.ipynb"]],
    },
    sglang: {
      answer: "SGLang",
      color: "var(--series-2)",
      body: `<p>RadixAttention keeps the KV cache in a radix tree, so partial, nested and
        <b>branching</b> prefix matches all reuse computed blocks. On an 87%-shared agent
        workload this is a large TTFT win; on independent prompts it is noise.</p>
        <p><b>Verify the premise:</b> run your own prompt distribution with
        <code>--disable-radix-cache</code> and without. If the gap is small, your prompts
        do not actually share as much as you think, and vLLM is equally good.</p>
        <p><b>Note</b> that its radix cache is on by default while vLLM's prefix caching
        has not always been &mdash; comparing defaults compares flags, not architectures.</p>`,
      notebooks: [["02", "02_sglang_deep_dive.ipynb"], ["07", "07_comparison.ipynb"]],
    },
    trtllm: {
      answer: "TensorRT-LLM (compiled engine)",
      color: "var(--series-4)",
      body: `<p>Ahead-of-time compilation fuses, autotunes and specialises for your exact
        GPU and shape bounds. Reliably the lowest TTFT and batch-1 latency on NVIDIA
        hardware.</p>
        <p><b>The real costs:</b> builds take minutes to hours; engines are not portable
        across GPU models or TensorRT versions; <code>max_input_len</code> is a
        compile-time constant, so exceeding it fails the request rather than slowing it.</p>
        <p><b>Honest gap:</b> against vLLM or SGLang with the <em>same</em> quantization
        and workload, expect modestly better latency and comparable throughput &mdash;
        not the 3&ndash;5&times; in marketing charts, which usually compare a tuned engine
        against another framework's defaults.</p>`,
      notebooks: [["04", "04_tensorrt_llm.ipynb"], ["05", "05_triton_inference_server.ipynb"]],
    },
    vllm: {
      answer: "vLLM",
      color: "var(--series-1)",
      body: `<p>The sane default: excellent throughput, nothing to compile, the widest
        model coverage, and the largest ecosystem.</p>
        <p><b>Tune in this order:</b> <code>max_model_len</code> (KV budget per sequence
        &mdash; the most common cause of low concurrency) &rarr;
        <code>gpu_memory_utilization</code> &rarr; <code>max_num_seqs</code> (the
        latency/throughput dial) &rarr; <code>enable_prefix_caching</code> &rarr;
        <code>max_num_batched_tokens</code> for ITL stability under mixed load.</p>
        <p><b>Then find your operating point:</b> sweep concurrency and deploy at the
        <em>goodput</em> peak, which is usually a lower concurrency than the throughput
        peak.</p>`,
      notebooks: [["01", "01_vllm_deep_dive.ipynb"], ["07", "07_comparison.ipynb"]],
    },
  };

  const pathEl = $("tree-path");
  const bodyEl = $("tree-body");
  const resetBtn = $("tree-reset");
  let crumbs = [];

  function go(key) {
    const node = NODES[key];
    pathEl.innerHTML = crumbs.map((c) => `<span class="tree-crumb">${c}</span>`).join(" › ");

    if (node.q) {
      resetBtn.hidden = crumbs.length === 0;
      bodyEl.innerHTML = `<div class="tree-q">${node.q}</div><div class="tree-options"></div>`;
      const box = bodyEl.querySelector(".tree-options");
      node.options.forEach((opt) => {
        const b = document.createElement("button");
        b.className = "tree-btn";
        b.textContent = opt.label;
        b.addEventListener("click", () => { crumbs.push(opt.crumb); go(opt.next); });
        box.appendChild(b);
      });
      return;
    }

    resetBtn.hidden = false;
    const links = node.notebooks
      .map(([n, file]) => `<a href="${nbUrl(file)}">notebook ${n}</a>`)
      .join(" · ");
    bodyEl.innerHTML = `<div class="note tree-answer" style="border-left-color:${node.color}">
      <h3>${node.answer}</h3>${node.body}
      <p style="margin-bottom:0;font-size:14px">Read more: ${links}</p></div>
      <p style="font-size:14px;color:var(--text-secondary)">
      Serving several models or frameworks on shared hardware? Put whichever engine you
      picked behind <b>Triton Inference Server</b> &mdash; that is the question it answers,
      and it is orthogonal to this one.</p>`;
  }

  resetBtn.addEventListener("click", () => { crumbs = []; go("start"); });
  go("start");
})();
