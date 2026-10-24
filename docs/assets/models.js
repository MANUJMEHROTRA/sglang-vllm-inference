/* Model shapes, GPU specs, and the cost formulas.
 *
 * These are the same formulas as bench/metrics.py (ModelShape.params,
 * ModelShape.kv_cache_bytes) and the same GPU specs as bench/sysinfo.py.
 * Kept in their own module with no DOM dependency so tools/check_site.mjs can
 * verify the site and the Python harness agree -- if the page and the
 * notebooks disagreed about how big a KV cache is, the page would be lying.
 *
 * All values below are real config.json fields. Parameter counts are checked
 * against published figures in tests/test_bench.py and tools/check_site.mjs.
 */

export const MODELS = {
  "qwen2.5-0.5b": { label: "Qwen2.5-0.5B", layers: 24, hidden: 896, heads: 14,
                    kvHeads: 2, headDim: 64, ffn: 4864, vocab: 151936, tied: true,
                    published: 0.494e9 },
  "qwen2.5-1.5b": { label: "Qwen2.5-1.5B", layers: 28, hidden: 1536, heads: 12,
                    kvHeads: 2, headDim: 128, ffn: 8960, vocab: 151936, tied: true,
                    published: 1.544e9 },
  "qwen2.5-7b":   { label: "Qwen2.5-7B", layers: 28, hidden: 3584, heads: 28,
                    kvHeads: 4, headDim: 128, ffn: 18944, vocab: 152064, tied: false,
                    published: 7.616e9 },
  "llama3.2-1b":  { label: "Llama-3.2-1B", layers: 16, hidden: 2048, heads: 32,
                    kvHeads: 8, headDim: 64, ffn: 8192, vocab: 128256, tied: true,
                    published: 1.236e9 },
  "llama3.1-8b":  { label: "Llama-3.1-8B", layers: 32, hidden: 4096, heads: 32,
                    kvHeads: 8, headDim: 128, ffn: 14336, vocab: 128256, tied: false,
                    published: 8.030e9 },
};

export const GPUS = {
  t4:      { label: "T4 (sm_75)",        memGb: 16, bwGbs: 320,  tflops: 65 },
  l4:      { label: "L4 (sm_89)",        memGb: 24, bwGbs: 300,  tflops: 121 },
  a100_40: { label: "A100 40GB (sm_80)", memGb: 40, bwGbs: 1555, tflops: 312 },
  a100_80: { label: "A100 80GB (sm_80)", memGb: 80, bwGbs: 2039, tflops: 312 },
  h100:    { label: "H100 SXM (sm_90)",  memGb: 80, bwGbs: 3350, tflops: 989 },
  rtx4090: { label: "RTX 4090 (sm_89)",  memGb: 24, bwGbs: 1008, tflops: 165 },
};

export const CONTEXTS = [512, 1024, 2048, 4096, 8192, 16384, 32768];
export const BATCHES = [1, 2, 4, 8, 16, 32, 64, 128, 256];
export const GB = 1024 ** 3;

/** Approximate parameter count: embeddings + blocks (+ untied lm_head). */
export function params(m) {
  const attn = m.hidden * (m.heads * m.headDim)
             + 2 * m.hidden * (m.kvHeads * m.headDim)
             + (m.heads * m.headDim) * m.hidden;
  const mlp = 3 * m.hidden * m.ffn;              // gate, up, down (SwiGLU)
  const perLayer = attn + mlp + 2 * m.hidden;    // + 2 RMSNorm weights
  return m.layers * perLayer + m.vocab * m.hidden * (m.tied ? 1 : 2);
}

/** 2 (K and V) x layers x kv_heads x head_dim x tokens x dtype_bytes */
export function kvBytes(m, tokens, dtypeBytes) {
  return 2 * m.layers * m.kvHeads * m.headDim * tokens * dtypeBytes;
}

/** FLOPs for one prefill pass: dense term plus quadratic attention. */
export function prefillFlops(m, seqLen) {
  return 2 * params(m) * seqLen + 4 * m.layers * seqLen * seqLen * m.hidden;
}

/** Peak FLOPS / peak bandwidth. Left of it, you are memory-bound. */
export function ridgePoint(gpu) {
  return (gpu.tflops * 1e12) / (gpu.bwGbs * 1e9);
}

/** Achievable TFLOP/s at a given arithmetic intensity. */
export function roofAt(gpu, intensity) {
  return Math.min(gpu.tflops, (intensity * gpu.bwGbs * 1e9) / 1e12);
}

/** Upper bound on decode tok/s from bandwidth alone. */
export function decodeCeilingTokS(m, gpu, batch, ctx, weightBytes = 2, kvDtypeBytes = 2) {
  const bytesPerStep = params(m) * weightBytes + kvBytes(m, ctx, kvDtypeBytes) * batch;
  return ((gpu.bwGbs * 1e9) / bytesPerStep) * batch;
}

/** KV budget left after weights and runtime overhead. */
export function kvBudgetGb(m, gpu, weightBytes = 2, util = 0.90, overheadGb = 1.2) {
  const weightsGb = (params(m) * weightBytes) / GB;
  return { weightsGb, kvGb: Math.max(0, gpu.memGb * util - weightsGb - overheadGb) };
}
