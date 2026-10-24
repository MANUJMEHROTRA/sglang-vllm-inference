/* Verify the site's interactive maths.
 *
 *   node tools/check_site.mjs
 *
 * The page's roofline and KV-cache panels compute their own numbers in
 * JavaScript. If those formulas drifted from bench/metrics.py, the page would
 * quietly contradict the notebooks. This checks the JS against published
 * parameter counts and against values cross-computed here by hand, and
 * tools/check_parity.py then checks the two languages against each other.
 */

import { BATCHES, CONTEXTS, GB, GPUS, MODELS, decodeCeilingTokS, kvBudgetGb,
         kvBytes, params, prefillFlops, ridgePoint, roofAt }
  from "../docs/assets/models.js";

let failures = 0;
function check(name, cond, detail = "") {
  if (cond) {
    console.log(`  ok  ${name}${detail ? "  " + detail : ""}`);
  } else {
    console.log(`  FAIL ${name}  ${detail}`);
    failures++;
  }
}

console.log("parameter counts vs published figures");
for (const [key, m] of Object.entries(MODELS)) {
  const p = params(m);
  const err = Math.abs(p - m.published) / m.published;
  check(m.label, err < 0.01,
        `${(p / 1e9).toFixed(3)}B vs published ${(m.published / 1e9).toFixed(3)}B ` +
        `(${(100 * err).toFixed(2)}% off)`);
}

console.log("\nKV cache arithmetic");
{
  const m = MODELS["qwen2.5-0.5b"];
  const kv1k = kvBytes(m, 1024, 2);
  // 2(K,V) x 24 layers x 2 kv_heads x 64 head_dim x 1024 tokens x 2 bytes
  check("Qwen2.5-0.5B KV at 1k ctx", kv1k === 2 * 24 * 2 * 64 * 1024 * 2,
        `${(kv1k / 1024 ** 2).toFixed(1)} MB`);
  check("scales linearly in tokens", kvBytes(m, 2048, 2) === 2 * kv1k);
  check("scales linearly in dtype bytes", kvBytes(m, 1024, 1) === kv1k / 2);

  // GQA: 14 query heads over 2 KV heads = 7x smaller cache than plain MHA.
  const mha = { ...m, kvHeads: m.heads };
  check("GQA saves 7x on Qwen2.5-0.5B", kvBytes(mha, 1024, 2) / kv1k === 7,
        `${kvBytes(mha, 1024, 2) / kv1k}x`);

  // Llama-3.1-8B: 32 query heads over 8 KV heads = 4x.
  const l8 = MODELS["llama3.1-8b"];
  check("GQA saves 4x on Llama-3.1-8B",
        kvBytes({ ...l8, kvHeads: l8.heads }, 1024, 2) / kvBytes(l8, 1024, 2) === 4);
}

console.log("\nroofline");
for (const [key, gpu] of Object.entries(GPUS)) {
  const ridge = ridgePoint(gpu);
  check(`${gpu.label} ridge point`, ridge > 50 && ridge < 500,
        `${ridge.toFixed(0)} FLOP/byte`);
}
{
  const gpu = GPUS.t4;
  const ridge = ridgePoint(gpu);
  // On the memory roof, achievable scales linearly with intensity.
  check("below the ridge, roof is bandwidth-limited",
        Math.abs(roofAt(gpu, ridge / 2) - gpu.tflops / 2) < 0.01);
  // Above it, clamped to peak FLOPS.
  check("above the ridge, roof is clamped to peak",
        roofAt(gpu, ridge * 10) === gpu.tflops);
}

console.log("\ndecode is memory-bound at batch 1");
{
  const m = MODELS["qwen2.5-0.5b"];
  const p = params(m);
  const ai = (2 * p) / (p * 2);                  // ignoring the small KV term
  check("arithmetic intensity ~1 FLOP/byte", Math.abs(ai - 1) < 0.01,
        `${ai.toFixed(2)}`);
  const gpu = GPUS.t4;
  const pctPeak = (100 * roofAt(gpu, ai)) / gpu.tflops;
  check("uses under 2% of peak arithmetic", pctPeak < 2, `${pctPeak.toFixed(2)}%`);
}

console.log("\nprefill crosses toward the ridge as the prompt grows");
{
  const m = MODELS["qwen2.5-0.5b"];
  const intensities = [128, 512, 2048, 8192].map(
    (n) => prefillFlops(m, n) / (params(m) * 2 + kvBytes(m, n, 2)));
  check("intensity rises with prompt length",
        intensities.every((v, i) => i === 0 || v > intensities[i - 1]),
        intensities.map((v) => v.toFixed(0)).join(" -> "));
  check("a 2048-token prefill is far above batch-1 decode", intensities[2] > 100,
        `${intensities[2].toFixed(0)} FLOP/byte`);
}

console.log("\nbatching improves throughput sublinearly");
{
  const m = MODELS["qwen2.5-0.5b"];
  const gpu = GPUS.t4;
  const bs1 = decodeCeilingTokS(m, gpu, 1, 1024);
  const bs32 = decodeCeilingTokS(m, gpu, 32, 1024);
  check("batch 32 beats batch 1 by >10x", bs32 > bs1 * 10,
        `${bs1.toFixed(0)} -> ${bs32.toFixed(0)} tok/s`);
  check("but by less than 32x (KV grows with batch)", bs32 < bs1 * 32,
        `${(bs32 / bs1).toFixed(1)}x`);
  check("per-sequence speed falls as batch grows", bs32 / 32 < bs1);
}

console.log("\nKV budget behaves sensibly");
{
  const m7 = MODELS["qwen2.5-7b"];
  const t4 = kvBudgetGb(m7, GPUS.t4, 2);
  check("7B fp16 does not fit on a 16GB T4", t4.kvGb === 0,
        `weights alone are ${t4.weightsGb.toFixed(1)} GB`);

  const t4int4 = kvBudgetGb(m7, GPUS.t4, 0.5);
  check("7B int4 does fit on a T4, with KV room", t4int4.kvGb > 1,
        `${t4int4.kvGb.toFixed(1)} GB left for KV`);

  const a100 = kvBudgetGb(m7, GPUS.a100_80, 2);
  check("7B fp16 on an 80GB A100 leaves lots of KV room", a100.kvGb > 50,
        `${a100.kvGb.toFixed(1)} GB`);

  // The headline concurrency claim for the small model on a free T4.
  const m05 = MODELS["qwen2.5-0.5b"];
  const b = kvBudgetGb(m05, GPUS.t4, 2);
  const fits = Math.floor((b.kvGb * GB) / kvBytes(m05, 4096, 2));
  check("Qwen2.5-0.5B on a T4 holds >100 sequences at 4k ctx", fits > 100,
        `${fits} sequences`);
}

console.log("\nconstants are internally consistent");
check("CONTEXTS ascending", CONTEXTS.every((v, i) => i === 0 || v > CONTEXTS[i - 1]));
check("BATCHES ascending", BATCHES.every((v, i) => i === 0 || v > BATCHES[i - 1]));
check("every model has a published count",
      Object.values(MODELS).every((m) => m.published > 0));
check("head_dim x heads is consistent with hidden size (or explicitly set)",
      Object.values(MODELS).every((m) => m.heads * m.headDim >= m.hidden));

console.log(failures ? `\n${failures} CHECK(S) FAILED` : "\nALL SITE MATH CHECKS PASSED");
process.exit(failures ? 1 : 0);
