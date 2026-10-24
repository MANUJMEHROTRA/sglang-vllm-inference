"""Notebook 08 - Finding the kernel worth optimizing, then optimizing it."""

from tools.nbgen import code, md, setup_cells

NAME = "08_kernel_optimization.ipynb"
TITLE = "08 - Kernel profiling and optimization in Triton"

CELLS = setup_cells(
    NAME,
    TITLE,
    "Choosing an engine is a procurement decision. Making one faster is an "
    "engineering one. This notebook covers the part people skip -- how to decide "
    "WHAT to optimize -- and then writes, validates and benchmarks real Triton "
    "kernels against that decision.",
    "~40 minutes",
) + [

    md(r"""
## The method: never optimize before you measure

Three questions, in this order. Skipping any of them is how people spend a week
making something 3x faster that was 2% of runtime.

### 1. Where does the time go? (Amdahl)

If a kernel is 5% of runtime, making it infinitely fast buys you 5%. The
maximum speedup from perfecting a component is:

```
speedup_max = 1 / (1 - fraction_of_time)
```

So: profile the whole step, sort by total time, and only look at the top.

### 2. Which roof is that kernel under? (roofline)

From notebook 00: compute the kernel's arithmetic intensity and compare it to
the GPU's ridge point.

- **Memory-bound** (left of ridge) -> the only wins are moving fewer bytes:
  fusion, quantization, better layout, avoiding round trips.
- **Compute-bound** (right of ridge) -> the wins are better math: tensor cores,
  bigger tiles, narrower dtypes, better instruction mix.

Applying a compute-bound fix to a memory-bound kernel does nothing. This is the
most common wasted effort in GPU work.

### 3. How far from the roof is it? (headroom)

A kernel at 85% of achievable bandwidth has ~15% left. One at 20% has 4x on the
table. Compute `achieved_bytes / elapsed_time` and divide by the GPU's peak.
That single ratio tells you whether to keep going.

```
                 +-----------------------------+
                 | 1. profile: what is hot?    |
                 +-------------+---------------+
                               v
                 +-----------------------------+
                 | 2. roofline: which roof?    |
                 +------+---------------+------+
                        |               |
              memory-bound          compute-bound
                        |               |
              fuse ops                tensor cores
              quantize                bigger tiles
              fix layout              narrower dtype
                        |               |
                        +-------+-------+
                                v
                 +-----------------------------+
                 | 3. headroom: % of peak?     |
                 |    high -> stop. low -> go. |
                 +-----------------------------+
```
"""),

    code('''
%pip install -q triton transformers

import torch, time
import triton
import triton.language as tl
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
from bench.plots import INK, PALETTE, grouped_bar, save, use_theme

use_theme()
assert torch.cuda.is_available(), "This notebook needs a CUDA GPU."

device = "cuda"
dtype = torch.float16
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=dtype).to(device).eval()

PEAK_BW_GB_S = SPECS.get("hbm_gb_s", 320.0)
PEAK_TFLOPS = SPECS.get("fp16_tflops", 65.0)
RIDGE = PEAK_TFLOPS * 1e12 / (PEAK_BW_GB_S * 1e9)
print(f"{HW.gpu_name}: {PEAK_BW_GB_S:.0f} GB/s, {PEAK_TFLOPS:.0f} TFLOP/s fp16")
print(f"ridge point: {RIDGE:.0f} FLOP/byte")
print(f"triton {triton.__version__}")
'''),

    md("""
## Step 1 - Profile a real decode step

Not a synthetic microbenchmark: the actual thing we care about. We set up a KV
cache with a 512-token prefill, then profile *only* the single-token decode
steps, which is where a chat server spends almost all of its time.
"""),

    code('''
from torch.profiler import ProfilerActivity, profile

prompt = "Explain memory bandwidth in GPUs. " * 30
ids = tokenizer(prompt, return_tensors="pt").input_ids[:, :512].to(device)

with torch.no_grad():
    out = model(input_ids=ids, use_cache=True)
    past = out.past_key_values
    next_id = out.logits[:, -1, :].argmax(-1, keepdim=True)

@torch.no_grad()
def decode_step():
    global past, next_id
    o = model(input_ids=next_id, past_key_values=past, use_cache=True)
    past = o.past_key_values
    next_id = o.logits[:, -1, :].argmax(-1, keepdim=True)

for _ in range(5):          # warm up: CUDA context, cuBLAS autotune, allocator
    decode_step()
torch.cuda.synchronize()

N_PROFILED = 20
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for _ in range(N_PROFILED):
        decode_step()
    torch.cuda.synchronize()

# torch renamed cuda_time_total -> device_time_total; accept either.
try:
    print(prof.key_averages().table(sort_by="device_time_total", row_limit=18))
except Exception:
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=18))
'''),

    code('''
import pandas as pd

# Turn the profile into a time budget, grouped by what the kernel DOES.
# Names vary across torch/cuBLAS versions, so classify by substring and keep
# an explicit "other" bucket rather than silently dropping time.
def classify(name: str) -> str:
    n = name.lower()
    if any(k in n for k in ("gemm", "gemv", "matmul", "cutlass", "addmm", "linear",
                            "nn_", "sgemm", "hgemm", "ampere", "turing", "cublas")):
        return "linear layers (GEMM/GEMV)"
    if any(k in n for k in ("attention", "sdpa", "flash", "bmm", "softmax", "scaled_dot")):
        return "attention"
    if any(k in n for k in ("norm", "rms", "layer_norm")):
        return "normalization"
    if any(k in n for k in ("rope", "rotary", "embedding", "index", "cat", "copy", "slice")):
        return "cache / layout ops"
    if any(k in n for k in ("mul", "add", "silu", "gelu", "sigmoid", "elementwise",
                            "vectorized", "unrolled")):
        return "elementwise"
    return "other"

def gpu_us(event) -> float:
    """GPU time for one profiler row, across torch naming changes."""
    for attr in ("device_time_total", "cuda_time_total", "self_device_time_total"):
        value = getattr(event, attr, None)
        if value:
            return float(value)
    return 0.0

rows = [{"name": e.key, "group": classify(e.key),
         "cuda_us": gpu_us(e) / N_PROFILED, "calls": e.count / N_PROFILED}
        for e in prof.key_averages() if gpu_us(e) > 0]
assert rows, "no GPU events captured -- is the model actually on CUDA?"
df = pd.DataFrame(rows)

budget = (df.groupby("group")["cuda_us"].sum().sort_values(ascending=False)
            .reset_index())
budget["pct"] = 100 * budget.cuda_us / budget.cuda_us.sum()
budget["max_speedup_if_free"] = 1 / (1 - budget.pct / 100)

total_us = budget.cuda_us.sum()
print(f"GPU time per decode step: {total_us:.0f} us "
      f"({1e6/total_us:.0f} steps/s ceiling for this implementation)\\n")
print(budget.to_string(index=False, float_format=lambda v: f"{v:8.2f}"))
'''),

    code('''
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.6))

colors = [PALETTE["blue"], PALETTE["orange"], PALETTE["aqua"],
          PALETTE["yellow"], PALETTE["magenta"], INK["muted"]]
ax1.barh(budget.group[::-1], budget.cuda_us[::-1], 0.62,
         color=colors[:len(budget)][::-1], linewidth=0)
ax1.set_xlabel("microseconds per decode step")
ax1.set_title("Where a decode step actually goes")
ax1.grid(axis="x")
for i, (v, p) in enumerate(zip(budget.cuda_us[::-1], budget.pct[::-1])):
    ax1.annotate(f"{v:.0f}us ({p:.0f}%)", (v, i), textcoords="offset points",
                 xytext=(6, 0), va="center", fontsize=9, color=INK["secondary"])

ax2.barh(budget.group[::-1], budget.max_speedup_if_free[::-1], 0.62,
         color=colors[:len(budget)][::-1], linewidth=0)
ax2.axvline(1.0, color=INK["axis"], linewidth=1.0)
ax2.set_xlabel("max possible speedup if this became free")
ax2.set_title("Amdahl's law: the ceiling on each option")
ax2.grid(axis="x")

plt.tight_layout(); save(fig, "08_time_budget"); plt.show()

top = budget.iloc[0]
print(f"\\nTarget: {top.group} at {top.pct:.0f}% of the step.")
print(f"Even making it free caps the win at {top.max_speedup_if_free:.2f}x.")
print("Anything below ~10% of the budget is not worth touching yet.")
'''),

    md(r"""
## Step 2 - Classify the hot kernels

For a small model at batch 1, the profile almost always says **linear layers**.
That deserves a closer look, because the name is misleading: at batch 1 these
are not matrix-matrix products at all.

A decode step multiplies a `[1, K]` activation vector by a `[K, N]` weight
matrix. That is a **GEMV** (matrix-vector), and its arithmetic intensity is
brutal:

```
FLOPs  = 2 * K * N              (one multiply-add per weight)
bytes  = K * N * 2              (read every weight once, fp16)
         + small activation terms

intensity = 2KN / 2KN = 1 FLOP per byte
```

**One FLOP per byte**, against a ridge point of ~200. These kernels run at
roughly 0.5% of the GPU's arithmetic capability and are *entirely* limited by
how fast weights come out of HBM.

Which hands us the optimization immediately: **the only way to speed up a GEMV
is to make the weights smaller.** Not better tiling, not tensor cores --
smaller. That is 4-bit quantization, and it is why every production engine
ships a fused dequantize-and-multiply kernel.

Let's confirm the diagnosis with measurement rather than assuming it.
"""),

    code('''
def timeit(fn, warmup=25, iters=100):
    """CUDA-synchronised timing. Use CUDA events: perf_counter around async
    work measures queueing, not execution."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters / 1000.0      # seconds


cfg = model.config
K, N = cfg.hidden_size, cfg.intermediate_size      # the FFN up-projection
print(f"profiling a representative GEMV: [1, {K}] x [{K}, {N}]")

x_vec = torch.randn(1, K, device=device, dtype=dtype)
W = torch.randn(K, N, device=device, dtype=dtype) * 0.02

t_fp16 = timeit(lambda: x_vec @ W)
bytes_moved = K * N * 2
flops = 2 * K * N
achieved_bw = bytes_moved / t_fp16 / 1e9
intensity = flops / bytes_moved

print(f"\\n  latency            {t_fp16*1e6:8.1f} us")
print(f"  bytes moved        {bytes_moved/1024**2:8.2f} MB")
print(f"  arithmetic int.    {intensity:8.2f} FLOP/byte   (ridge is {RIDGE:.0f})")
print(f"  achieved bandwidth {achieved_bw:8.1f} GB/s")
print(f"  % of peak BW       {100*achieved_bw/PEAK_BW_GB_S:8.1f}%")
print(f"  % of peak FLOPS    {100*(flops/t_fp16/1e12)/PEAK_TFLOPS:8.3f}%")
print("\\nDiagnosis: memory-bound, already near peak bandwidth.")
print("There is no tuning win here -- cuBLAS is doing fine. The ONLY")
print("remaining lever is reading fewer bytes.")
'''),

    md("""
## Step 3 - Warm-up: RMSNorm in Triton

Before the real target, write a simple kernel to learn the tooling and to
establish how we validate and measure. RMSNorm is a good first kernel: it is
purely memory-bound, small, and easy to check against a reference.

$$\\text{RMSNorm}(x) = \\frac{x}{\\sqrt{\\frac{1}{N}\\sum x_i^2 + \\epsilon}} \\odot w$$

The whole row fits in registers for a 896-wide model, so one program handles
one row: load once, reduce, scale, store. No second pass over memory.
"""),

    code('''
@triton.jit
def _rmsnorm_fwd(X, W, Y, stride_row, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    # Load the whole row once; it stays in registers for the rest of the kernel.
    x = tl.load(X + row * stride_row + cols, mask=mask, other=0.0).to(tl.float32)

    # Reduce in fp32 even though the data is fp16: summing 896 squared values
    # in fp16 loses accuracy fast, and the reduction is free compared to the
    # memory traffic.
    var = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(Y + row * stride_row + cols, (x * rstd * w).to(Y.dtype.element_ty), mask=mask)


def rmsnorm_triton(x, weight, eps=1e-6):
    out = torch.empty_like(x)
    rows, N = x.reshape(-1, x.shape[-1]).shape
    BLOCK = triton.next_power_of_2(N)
    _rmsnorm_fwd[(rows,)](
        x, weight, out, x.stride(-2), N, eps, BLOCK=BLOCK,
        num_warps=max(1, min(16, BLOCK // 256)),
    )
    return out


def rmsnorm_ref(x, weight, eps=1e-6):
    v = x.float().pow(2).mean(-1, keepdim=True)
    return (x.float() * torch.rsqrt(v + eps)).to(x.dtype) * weight


# --- correctness first, always ---------------------------------------------
for rows in (1, 7, 64, 1024):
    x = torch.randn(rows, K, device=device, dtype=dtype)
    w = torch.randn(K, device=device, dtype=dtype)
    got, want = rmsnorm_triton(x, w), rmsnorm_ref(x, w)
    err = (got.float() - want.float()).abs().max().item()
    assert torch.allclose(got, want, atol=2e-2, rtol=1e-2), f"rows={rows} max err {err}"
    print(f"  rows={rows:<5} max abs error {err:.2e}  ok")
'''),

    code('''
# --- then measure -----------------------------------------------------------
print(f"{'rows':>6} {'torch':>10} {'triton':>10} {'speedup':>9} {'GB/s':>9} {'% peak':>8}")
print("-" * 58)
rms_rows, rms_speedup, rms_bw = [], [], []
for rows in (1, 8, 64, 512, 4096):
    x = torch.randn(rows, K, device=device, dtype=dtype)
    w = torch.randn(K, device=device, dtype=dtype)
    t_ref = timeit(lambda: rmsnorm_ref(x, w))
    t_tri = timeit(lambda: rmsnorm_triton(x, w))
    # One read of x, one write of y. Weights are tiny and cached.
    gbs = (2 * rows * K * 2) / t_tri / 1e9
    rms_rows.append(rows); rms_speedup.append(t_ref / t_tri); rms_bw.append(gbs)
    print(f"{rows:>6} {t_ref*1e6:>9.1f}u {t_tri*1e6:>9.1f}u {t_ref/t_tri:>8.2f}x "
          f"{gbs:>8.0f} {100*gbs/PEAK_BW_GB_S:>7.0f}%")

print("\\nAt small row counts the win is launch overhead: the torch version runs")
print("several kernels (pow, mean, rsqrt, mul, mul); ours runs one.")
print("At large row counts both saturate bandwidth and converge -- as they must,")
print("because neither can beat the memory roof.")
'''),

    md(r"""
## Step 4 - Fusion: the residual add that costs a round trip

Every transformer block does this twice:

```python
x = x + attn_out          # residual add        -> reads 2N, writes N
x = rmsnorm(x) * w        # normalization       -> reads N,  writes N
```

Five tensor-sized memory operations. But the add's output is consumed
*immediately* by the norm -- it never needed to visit HBM at all. Fusing them:

```
unfused:  read x, read res, write sum | read sum, write y   = 5N
fused:    read x, read res, write res_out, write y          = 4N
```

A 20% traffic reduction on a memory-bound op, which should land as roughly a
20% speedup. Not spectacular -- but it costs nothing at run time, it is exactly
what vLLM's `fused_add_rms_norm` does, and the *reasoning* generalises to every
producer-consumer pair of elementwise ops in a model.

We keep `res_out` because the residual is needed again by the next block.
"""),

    code('''
@triton.jit
def _fused_add_rmsnorm(X, RES, W, Y, ROUT, stride_row, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    off = row * stride_row + cols

    x = tl.load(X + off, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(RES + off, mask=mask, other=0.0).to(tl.float32)
    s = x + r

    # The new residual must be written out; the normalized value does NOT
    # round-trip -- it stays in registers and feeds the normalization directly.
    tl.store(ROUT + off, s.to(ROUT.dtype.element_ty), mask=mask)

    var = tl.sum(s * s, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(Y + off, (s * rstd * w).to(Y.dtype.element_ty), mask=mask)


def fused_add_rmsnorm(x, residual, weight, eps=1e-6):
    y, rout = torch.empty_like(x), torch.empty_like(x)
    rows, N = x.reshape(-1, x.shape[-1]).shape
    BLOCK = triton.next_power_of_2(N)
    _fused_add_rmsnorm[(rows,)](x, residual, weight, y, rout, x.stride(-2), N, eps,
                                BLOCK=BLOCK, num_warps=max(1, min(16, BLOCK // 256)))
    return y, rout


def unfused_ref(x, residual, weight, eps=1e-6):
    s = x + residual
    return rmsnorm_ref(s, weight, eps), s


x = torch.randn(512, K, device=device, dtype=dtype)
res = torch.randn(512, K, device=device, dtype=dtype)
w = torch.randn(K, device=device, dtype=dtype)

y_t, r_t = fused_add_rmsnorm(x, res, w)
y_r, r_r = unfused_ref(x, res, w)
assert torch.allclose(y_t, y_r, atol=2e-2, rtol=1e-2)
# The residual tolerance is looser than you might expect: we accumulate in
# fp32 and cast once, torch adds in fp16. Both are "right"; they differ by
# about one fp16 ulp, which at these magnitudes is a few times 1e-3.
assert torch.allclose(r_t, r_r, atol=5e-3, rtol=5e-3)
print("correctness ok")

t_unf = timeit(lambda: unfused_ref(x, res, w))
t_fus = timeit(lambda: fused_add_rmsnorm(x, res, w))
print(f"\\n  unfused  {t_unf*1e6:7.1f} us   ({5*512*K*2/t_unf/1e9:6.0f} GB/s over 5N traffic)")
print(f"  fused    {t_fus*1e6:7.1f} us   ({4*512*K*2/t_fus/1e9:6.0f} GB/s over 4N traffic)")
print(f"  speedup  {t_unf/t_fus:7.2f}x   (traffic model predicts 5/4 = 1.25x)")
'''),

    md(r"""
## Step 5 - The real target: a fused int4 dequantize + GEMV

Back to the profile. Linear layers are the budget, they are memory-bound at
~1 FLOP/byte, and cuBLAS is already near peak bandwidth. The only lever left is
**fewer bytes**.

Store weights as 4-bit integers with group-wise scales and zero points, and
dequantize *inside the kernel*, in registers:

```
fp16 weights:  K * N * 2 bytes
int4 weights:  K * N / 2 bytes  +  2 * (K/G) * N * 2 bytes  (scales + zeros)
               \__________/        \______________________/
                4x smaller          small: G=128 adds ~3%
```

Crucially the dequantized values **never touch HBM**. A naive implementation
that dequantizes to an fp16 buffer and then calls cuBLAS is *slower* than fp16
-- it moves all the original bytes plus the int4 ones. Fusion is not an
optimisation here; it is the entire point.

### The quantization scheme

Asymmetric, group-wise along K, group size 128:

```
for each group of 128 consecutive K values, per output column n:
    scale = (max - min) / 15
    zero  = round(-min / scale)
    q     = clamp(round(w / scale + zero), 0, 15)      # 4 bits
    w_hat = (q - zero) * scale
```

Asymmetric (with a zero point) rather than symmetric because weight
distributions are not centred on zero -- the same reasoning as GGUF's K-quants
in notebook 03, which also store a per-sub-block minimum.

Packing: two 4-bit values per byte, along K, so byte `i` holds `k=2i` in the
low nibble and `k=2i+1` in the high nibble.
"""),

    code('''
GROUP = 128

def quantize_int4(W, group=GROUP):
    """W: [K, N] fp16 -> packed uint8 [K//2, N], scale/zero [K//group, N] fp16."""
    Kk, Nn = W.shape
    assert Kk % group == 0, "K must be divisible by the group size"
    Wg = W.float().reshape(Kk // group, group, Nn)
    wmax, wmin = Wg.amax(dim=1), Wg.amin(dim=1)              # [G, N]
    scale = ((wmax - wmin) / 15.0).clamp(min=1e-6)
    zero = torch.round(-wmin / scale)
    q = torch.clamp(torch.round(Wg / scale.unsqueeze(1) + zero.unsqueeze(1)), 0, 15)
    q = q.reshape(Kk, Nn).to(torch.uint8)
    packed = (q[0::2, :] | (q[1::2, :] << 4)).contiguous()   # [K//2, N]
    return packed, scale.to(torch.float16), zero.to(torch.float16)


def dequantize_int4(packed, scale, zero, Kk, group=GROUP):
    """Reference dequantization, for checking the kernel."""
    lo = (packed & 0xF).float()
    hi = ((packed >> 4) & 0xF).float()
    q = torch.empty(Kk, packed.shape[1], device=packed.device, dtype=torch.float32)
    q[0::2], q[1::2] = lo, hi
    g = torch.arange(Kk, device=packed.device) // group
    return (q - zero.float()[g]) * scale.float()[g]


packed, scale, zero = quantize_int4(W)
W_hat = dequantize_int4(packed, scale, zero, K)

rel_err = ((W_hat - W.float()).norm() / W.float().norm()).item()
orig_mb = W.numel() * 2 / 1024**2
new_mb = (packed.numel() + 2 * scale.numel() * 2) / 1024**2
print(f"  fp16          {orig_mb:7.2f} MB")
print(f"  int4 + scales {new_mb:7.2f} MB   ({orig_mb/new_mb:.2f}x smaller)")
print(f"  effective bits/weight {8 * new_mb * 1024**2 / W.numel():.2f}")
print(f"  relative Frobenius error {rel_err:.4f}")
'''),

    code('''
@triton.jit
def _dequant_gemv(X, QW, SCALE, ZERO, Y, N, K,
                  GROUP_SIZE: tl.constexpr,
                  BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """y[1, N] = x[1, K] @ dequant(QW)[K, N]

    One program computes BLOCK_N output columns. BLOCK_K must divide
    GROUP_SIZE so that every k in a tile shares one scale/zero, which lets us
    load those once per tile instead of once per element.
    """
    pid = tl.program_id(0)
    n_off = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_off < N

    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k_idx = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_idx < K

        x = tl.load(X + k_idx, mask=k_mask, other=0.0).to(tl.float32)

        # Byte i holds k=2i in the low nibble and k=2i+1 in the high nibble.
        byte_row = k_idx // 2
        b = tl.load(QW + byte_row[:, None] * N + n_off[None, :],
                    mask=k_mask[:, None] & n_mask[None, :], other=0).to(tl.int32)
        nib = tl.where((k_idx % 2)[:, None] == 1, (b >> 4) & 0xF, b & 0xF).to(tl.float32)

        # One scale/zero per (group, column) -- constant across this k-tile.
        g = k0 // GROUP_SIZE
        s = tl.load(SCALE + g * N + n_off, mask=n_mask, other=0.0).to(tl.float32)
        z = tl.load(ZERO + g * N + n_off, mask=n_mask, other=0.0).to(tl.float32)

        # Dequantization happens HERE, in registers. These values never
        # become a tensor in HBM -- that is the whole optimization.
        w = (nib - z[None, :]) * s[None, :]
        acc += tl.sum(x[:, None] * w, axis=0)

    tl.store(Y + n_off, acc.to(Y.dtype.element_ty), mask=n_mask)


def dequant_gemv(x, packed, scale, zero, Kk, Nn,
                 BLOCK_N=64, BLOCK_K=64, num_warps=4):
    assert GROUP % BLOCK_K == 0, "BLOCK_K must divide GROUP"
    y = torch.empty(Nn, device=x.device, dtype=torch.float16)
    grid = (triton.cdiv(Nn, BLOCK_N),)
    _dequant_gemv[grid](x.reshape(-1), packed, scale, zero, y, Nn, Kk,
                        GROUP_SIZE=GROUP, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                        num_warps=num_warps)
    return y.reshape(1, Nn)


# --- correctness ------------------------------------------------------------
y_tri = dequant_gemv(x_vec, packed, scale, zero, K, N)
y_ref = (x_vec.float() @ W_hat).half()           # same quantized weights
err = (y_tri.float() - y_ref.float()).abs().max().item()
rel = err / y_ref.float().abs().max().item()
print(f"  max abs error vs dequantized reference: {err:.4f}  (relative {rel:.2e})")
assert rel < 1e-2, "kernel disagrees with the reference dequantization"
print("  correctness ok")

y_fp16 = (x_vec.float() @ W.float()).half()      # against ORIGINAL weights
q_rel = ((y_tri.float() - y_fp16.float()).norm() / y_fp16.float().norm()).item()
w_rel = ((W_hat - W.float()).norm() / W.float().norm()).item()
print(f"\\n  relative error in the WEIGHTS  {w_rel:.4f}")
print(f"  relative error in the OUTPUT   {q_rel:.4f}")
print(f"  ratio {q_rel/w_rel:.2f}x -- quantization error does NOT average away")
'''),

    md(r"""
### Two results worth pausing on

**The weight error is at its theoretical floor.** Round-to-nearest 4-bit
quantization of roughly Gaussian weights cannot do better than about 10%
relative error. The arithmetic: a group of 128 Gaussian samples spans about
5.6σ, split into 15 steps gives a step of ~0.37σ, and uniform rounding error
has RMS of step/√12 ≈ 0.108σ. If you measured ~0.10 above, your quantizer is
working correctly -- there is no bug to find.

This is precisely why **AWQ and GPTQ exist**. They cannot beat that floor by
rounding better; they beat it by *choosing which weights absorb the error*.
AWQ scales up the ~1% of channels with the largest activations before
quantizing, so their error shrinks while unimportant channels absorb more.
GPTQ quantizes column by column and updates the remaining weights to
compensate for the error already committed. Both are error *allocation*
strategies, which is the same idea as llama.cpp's importance matrix from
notebook 03.

**The error does not wash out in the matmul.** You might expect 896 independent
errors to cancel in the accumulation. They do not: both the signal and the
error grow as √K, so the *relative* error is preserved almost exactly. The
ratio you printed should be close to 1.0.

So why do 4-bit models still work? Not because the arithmetic error is small --
it is 10%. Because the *network* is robust to it: the residual stream is highly
redundant, softmax and normalization are contractive, and the next-token
argmax is usually decided by a margin far larger than the noise. That is an
empirical property of trained transformers, not a mathematical guarantee, and
it is why quantization quality must be *measured* (notebook 03's perplexity
sweep) rather than derived.
"""),

    code('''
# --- performance ------------------------------------------------------------
t_fp16 = timeit(lambda: x_vec @ W)
t_int4 = timeit(lambda: dequant_gemv(x_vec, packed, scale, zero, K, N))

bytes_fp16 = K * N * 2
bytes_int4 = packed.numel() + 2 * scale.numel() * 2

print(f"{'':<22}{'latency':>10} {'bytes':>12} {'GB/s':>9} {'% peak BW':>11}")
print("-" * 68)
for name, t, b in [("cuBLAS fp16 GEMV", t_fp16, bytes_fp16),
                   ("Triton int4 GEMV", t_int4, bytes_int4)]:
    gbs = b / t / 1e9
    print(f"{name:<22}{t*1e6:>9.1f}u {b/1024**2:>10.2f}MB {gbs:>8.0f} "
          f"{100*gbs/PEAK_BW_GB_S:>10.0f}%")

print(f"\\n  speedup            {t_fp16/t_int4:.2f}x")
print(f"  traffic reduction  {bytes_fp16/bytes_int4:.2f}x")
print(f"  efficiency         {100*(t_fp16/t_int4)/(bytes_fp16/bytes_int4):.0f}% "
      "of the traffic-model prediction")
'''),

    md("""
### Reading that efficiency number

If the speedup matches the traffic reduction, the kernel is as good as the
memory model allows and you are done. Expect it to fall short, for reasons
worth understanding because they recur in every quantized kernel:

1. **Each packed byte is loaded twice** -- once for its low nibble, once for its
   high nibble. The second load hits L1 so it is cheap, not free. Loading
   `BLOCK_K/2` bytes and un-interleaving in registers removes this, at the cost
   of fiddlier index arithmetic. (Exercise 1.)
2. **Dequantization costs arithmetic** -- a subtract and a multiply per weight.
   Free in a pure-bandwidth model; not free when the kernel is this close to
   the roof.
3. **Tile shape and occupancy.** `BLOCK_N`/`BLOCK_K`/`num_warps` decide whether
   there are enough warps in flight to hide memory latency. Autotuning next.
4. **cuBLAS is a hard baseline.** It has been tuned for a decade. Beating it on
   raw fp16 is unlikely; beating it by *changing the problem* -- moving a
   quarter of the bytes -- is the entire strategy.

Production kernels (Marlin, AWQ/GPTQ fused kernels, llama.cpp's MMQ) add
dedicated int4 unpacking instructions, async memory pipelining, and layouts
chosen so the dequantized tile lands directly in tensor-core registers. They
get closer to the 4x. What you have here reproduces the *idea* faithfully and
shows you how to measure the gap.
"""),

    code('''
# --- autotuning: let the machine search the tile space ----------------------
configs = [(bn, bk, w) for bn in (32, 64, 128) for bk in (32, 64, 128)
           for w in (2, 4, 8) if GROUP % bk == 0]

print(f"{'BLOCK_N':>8} {'BLOCK_K':>8} {'warps':>6} {'latency':>10} {'vs fp16':>9}")
print("-" * 46)
best = None
for bn, bk, warps in configs:
    try:
        fn = lambda: dequant_gemv(x_vec, packed, scale, zero, K, N, bn, bk, warps)
        fn()                                    # compile + correctness smoke test
        t = timeit(fn, warmup=10, iters=50)
    except Exception as exc:
        print(f"{bn:>8} {bk:>8} {warps:>6}   failed: {type(exc).__name__}")
        continue
    if best is None or t < best[0]:
        best = (t, bn, bk, warps)
    print(f"{bn:>8} {bk:>8} {warps:>6} {t*1e6:>9.1f}u {t_fp16/t:>8.2f}x")

t_best, bn, bk, warps = best
print(f"\\nbest: BLOCK_N={bn} BLOCK_K={bk} num_warps={warps} "
      f"-> {t_best*1e6:.1f}us ({t_fp16/t_best:.2f}x vs fp16)")
print(f"autotuning gained {t_int4/t_best:.2f}x over the default tile shape.")
print("\\nThis is why triton.autotune exists, and why a kernel tuned on one GPU")
print("should be re-tuned on another -- the best tile depends on SM count,")
print("L2 size and memory latency.")
'''),

    code('''
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))

grouped_bar(["fp16 cuBLAS", "int4 Triton\\n(default)", "int4 Triton\\n(tuned)"],
            {"latency (us)": [t_fp16*1e6, t_int4*1e6, t_best*1e6]},
            ylabel="microseconds", title=f"GEMV [1,{K}] x [{K},{N}]",
            colors={"latency (us)": PALETTE["blue"]}, ax=ax1)

gb = [bytes_fp16/t_fp16/1e9, bytes_int4/t_int4/1e9, bytes_int4/t_best/1e9]
ax2.bar(range(3), gb, 0.6, color=PALETTE["aqua"], linewidth=0)
ax2.axhline(PEAK_BW_GB_S, color=INK["muted"], linewidth=1.3)
ax2.annotate(f"peak {PEAK_BW_GB_S:.0f} GB/s", (2.4, PEAK_BW_GB_S),
             textcoords="offset points", xytext=(0, 4), ha="right",
             fontsize=9, color=INK["muted"])
ax2.set_xticks(range(3))
ax2.set_xticklabels(["fp16", "int4\\n(default)", "int4\\n(tuned)"])
ax2.set_ylabel("achieved GB/s"); ax2.set_title("Bandwidth utilisation")
for i, v in enumerate(gb):
    ax2.annotate(f"{v:.0f}", (i, v), textcoords="offset points", xytext=(0, 3),
                 ha="center", fontsize=9, color=INK["secondary"])

plt.tight_layout(); save(fig, "08_dequant_gemv"); plt.show()
'''),

    md(r"""
## Step 6 - Going deeper: Nsight

The torch profiler tells you *which* kernel is hot. Nsight tells you *why*.
Neither runs usefully inside a notebook cell, so these are the commands to run
on a real machine with a script:

**Nsight Systems** -- the timeline. Answers "is the GPU even busy?"

```bash
nsys profile -o decode --trace=cuda,nvtx,osrt python decode_script.py
nsys stats decode.nsys-rep
```

Look for gaps between kernels (launch overhead or CPU-bound scheduling -> CUDA
graphs), and for memcpys on the critical path (a layout problem).

**Nsight Compute** -- one kernel, in depth. Answers "why is this kernel slow?"

```bash
ncu --set full --kernel-name _dequant_gemv -o gemv python kernel_script.py
ncu --metrics \
  dram__bytes.sum.per_second,\
  sm__throughput.avg.pct_of_peak_sustained_elapsed,\
  launch__occupancy_limit_registers \
  python kernel_script.py
```

The three numbers that matter, in order:

| Metric | Meaning | What to do |
|---|---|---|
| `dram__bytes.sum.per_second` | achieved bandwidth | near peak? you are done |
| `sm__throughput...pct_of_peak` | compute utilisation | high + low BW -> compute-bound |
| `achieved_occupancy` | warps in flight vs max | low -> too many registers or too few blocks |

The diagnosis table:

- **high bandwidth, low compute** -> memory-bound. Move fewer bytes. You are
  at the roof; only quantization or fusion helps.
- **low bandwidth, high compute** -> compute-bound. Better math.
- **low bandwidth, low compute** -> *latency*-bound. Not enough parallelism to
  hide memory latency. Raise occupancy, increase blocks, or software-pipeline
  the loads. This is the most commonly misdiagnosed case, and it is where
  tuning `num_warps` pays.
"""),

    md('''
## Step 7 - Getting a kernel into a real engine

A faster kernel in a notebook is worth nothing until it runs in the serving
path. Three routes, in increasing order of effort and permanence:

**1. Register a custom op and monkey-patch.** Fastest to try, fine for an
experiment, works with `torch.compile` if you declare a `FakeTensor` rule:

```python
import torch

@torch.library.custom_op("mylib::dequant_gemv", mutates_args=())
def _op(x: torch.Tensor, qw: torch.Tensor, scale: torch.Tensor,
        zero: torch.Tensor) -> torch.Tensor:
    return dequant_gemv(x, qw, scale, zero, x.shape[-1], qw.shape[-1])

@_op.register_fake
def _(x, qw, scale, zero):
    return x.new_empty((x.shape[0], qw.shape[-1]))

# then replace the module's forward in the loaded model
```

**2. vLLM's custom-op layer.** vLLM routes quantized linear layers through a
`QuantizationConfig` / `LinearMethodBase` pair. Implementing one puts your
kernel on the real serving path with the real scheduler around it, which is the
only way to find out whether a 1.8x microbenchmark becomes a 1.1x end-to-end
win (it usually does -- Amdahl again).

**3. Upstream it.** vLLM, SGLang and llama.cpp all take kernel contributions.
Bring a benchmark across several shapes and GPUs, and a correctness test.

**Always close the loop end-to-end.** A kernel that is 2x faster on 30% of the
step is a 1.2x win overall. Re-run notebook 01's concurrency sweep with your
kernel in place and compare result files -- that is what the shared schema is
for.
'''),

    md("""
## The decision framework, condensed

**What to optimize**

1. Profile the real workload, not a microbenchmark.
2. Sort by total time. Ignore anything under ~10% of the budget.
3. Compute each candidate's arithmetic intensity; compare to the ridge point.
4. Compute achieved bandwidth (or FLOPS) as a fraction of peak. Low fraction =
   headroom; high fraction = move on.
5. Estimate the Amdahl ceiling before writing a line of code.

**How to optimize, by diagnosis**

| Diagnosis | Fixes, in order |
|---|---|
| Memory-bound, near peak BW | quantize (fewer bytes), fuse neighbours, shrink KV |
| Memory-bound, far from peak | coalesce accesses, raise occupancy, pipeline loads |
| Compute-bound | tensor cores, bigger tiles, narrower dtype, fewer ops |
| Latency-bound (both low) | more parallelism, more warps, CUDA graphs |
| Many tiny kernels | fuse them, or capture a CUDA graph |

**The order to try things on a serving stack**, highest expected value first:

1. **Batching / scheduling config.** Free, and usually the biggest win.
2. **Quantization.** A few hours, typically 2-4x on decode.
3. **Prefix caching**, if the workload shares prefixes. Free when it applies.
4. **A different engine.** Days; usually a modest factor.
5. **Custom kernels.** Weeks; targeted wins on one hot spot.

Custom kernels are last for a reason. They are the most fun and the least
leveraged -- which is exactly why it is worth knowing how to tell when you have
actually reached that point.

### Exercises

1. Remove the duplicate byte load: load `BLOCK_K/2` bytes and un-interleave the
   nibbles in registers. Measure the gain and compare to the predicted traffic
   change.
2. Extend `_dequant_gemv` to `BLOCK_M > 1` (a real GEMM) and find the batch
   size at which fp16 cuBLAS wins again. That crossover is the point where the
   kernel stops being memory-bound, and it is why engines keep both paths.
3. Replace the `GROUP=128` scales with `GROUP=32`. Measure the accuracy gain
   and the throughput cost, then decide which you would ship.
4. Profile a **prefill** step instead of decode. The budget should shift toward
   attention and real GEMMs. Does your int4 kernel still help? (Predict first:
   which roof is prefill under?)
5. Put the tuned kernel behind a `torch.library.custom_op`, patch it into the
   model, and re-run notebook 00's batching sweep. Compare the end-to-end win
   against the microbenchmark win.

**Back to:** [07 - The comparison](07_comparison.ipynb) to re-run the
aggregation with your new results, or the
[concept site](../docs/index.html) for the visual summary.
"""),
]
