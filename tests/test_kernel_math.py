"""Validate notebook 08's int4 kernel WITHOUT a GPU.

The Triton kernel in notebook 08 can only run on CUDA, but the part most
likely to be wrong is not the GPU code -- it is the index arithmetic: which
byte holds which weight, which nibble, which group's scale. Get that wrong and
nothing crashes; you just get quietly incorrect numbers.

So we emulate the kernel's addressing in numpy, line for line, and check it
against a straightforward reference. If this passes, the kernel is computing
the right thing and any remaining issue is performance, not correctness.

    python tests/test_kernel_math.py

Requires numpy only.
"""

from __future__ import annotations

import sys

try:
    import numpy as np
except ImportError:  # pragma: no cover
    print("numpy not installed -- skipping kernel-math tests (pip install numpy)")
    sys.exit(0)

GROUP = 128
K, N = 896, 4864        # Qwen2.5-0.5B hidden_size x intermediate_size


def quantize_int4(W, group=GROUP):
    """Group-wise asymmetric 4-bit, packed two-per-byte along K.

    Identical arithmetic to the torch version in notebook 08.
    """
    Kk, Nn = W.shape
    assert Kk % group == 0, "K must be divisible by the group size"
    Wg = W.astype(np.float32).reshape(Kk // group, group, Nn)
    wmax, wmin = Wg.max(axis=1), Wg.min(axis=1)
    scale = np.maximum((wmax - wmin) / 15.0, 1e-6)
    zero = np.round(-wmin / scale)
    q = np.clip(np.round(Wg / scale[:, None, :] + zero[:, None, :]), 0, 15)
    q = q.reshape(Kk, Nn).astype(np.uint8)
    # byte i holds k=2i in the low nibble, k=2i+1 in the high nibble
    packed = (q[0::2, :] | (q[1::2, :] << 4)).astype(np.uint8)
    return packed, scale.astype(np.float16), zero.astype(np.float16)


def dequantize_int4(packed, scale, zero, Kk, group=GROUP):
    lo = (packed & 0xF).astype(np.float32)
    hi = ((packed >> 4) & 0xF).astype(np.float32)
    q = np.empty((Kk, packed.shape[1]), dtype=np.float32)
    q[0::2], q[1::2] = lo, hi
    g = np.arange(Kk) // group
    return (q - zero.astype(np.float32)[g]) * scale.astype(np.float32)[g]


def emulate_dequant_gemv(x, packed, scale, zero, Kk, Nn, BLOCK_N, BLOCK_K,
                         group=GROUP):
    """Line-for-line emulation of the Triton kernel ``_dequant_gemv``.

    Every statement mirrors one in the kernel, including the masking, so a
    disagreement here is a real disagreement in the kernel.
    """
    assert group % BLOCK_K == 0, "BLOCK_K must divide GROUP"
    y = np.zeros(Nn, dtype=np.float32)

    for pid in range((Nn + BLOCK_N - 1) // BLOCK_N):
        n_off = pid * BLOCK_N + np.arange(BLOCK_N)
        n_mask = n_off < Nn
        n_safe = np.where(n_mask, n_off, 0)
        acc = np.zeros(BLOCK_N, dtype=np.float32)

        for k0 in range(0, Kk, BLOCK_K):
            k_idx = k0 + np.arange(BLOCK_K)
            k_mask = k_idx < Kk
            k_safe = np.where(k_mask, k_idx, 0)

            xv = np.where(k_mask, x[k_safe], 0.0).astype(np.float32)

            byte_row = k_safe // 2
            b = packed[byte_row[:, None], n_safe[None, :]].astype(np.int32)
            b = np.where(k_mask[:, None] & n_mask[None, :], b, 0)
            nib = np.where((k_safe % 2)[:, None] == 1,
                           (b >> 4) & 0xF, b & 0xF).astype(np.float32)

            g = k0 // group            # constant across the tile, by construction
            s = np.where(n_mask, scale.astype(np.float32)[g, n_safe], 0.0)
            z = np.where(n_mask, zero.astype(np.float32)[g, n_safe], 0.0)

            acc += (xv[:, None] * ((nib - z[None, :]) * s[None, :])).sum(axis=0)

        y[n_safe] = np.where(n_mask, acc, y[n_safe])
    return y


def test_nibble_packing_is_lossless():
    """Every 4-bit code must survive the pack/unpack round trip exactly."""
    rng = np.random.default_rng(0)
    q = rng.integers(0, 16, size=(K, N), dtype=np.uint8)
    packed = (q[0::2, :] | (q[1::2, :] << 4)).astype(np.uint8)

    lo = packed & 0xF
    hi = (packed >> 4) & 0xF
    back = np.empty_like(q)
    back[0::2], back[1::2] = lo, hi
    assert np.array_equal(q, back)


def test_quantization_error_is_at_the_theoretical_floor():
    """~10% relative error is the FLOOR for 4-bit RTN, not a bug.

    A group of 128 Gaussian samples spans ~5.6 sigma; 15 steps gives a step of
    ~0.37 sigma; uniform rounding error has RMS step/sqrt(12) ~ 0.108 sigma.
    This test exists so nobody 'fixes' a correct quantizer.
    """
    rng = np.random.default_rng(0)
    W = (rng.standard_normal((K, N)) * 0.02).astype(np.float32)
    packed, scale, zero = quantize_int4(W)
    W_hat = dequantize_int4(packed, scale, zero, K)

    rel = np.linalg.norm(W_hat - W) / np.linalg.norm(W)
    expected_range = 2 * np.abs(rng.standard_normal((4000, GROUP))).max(axis=1).mean()
    predicted = (expected_range / 15.0) / np.sqrt(12)
    assert abs(rel - predicted) < 0.03, f"rel={rel:.4f} predicted={predicted:.4f}"


def test_zero_point_beats_symmetric():
    """Justifies storing a zero point: asymmetric should be clearly better."""
    rng = np.random.default_rng(1)
    W = (rng.standard_normal((K, N)) * 0.02).astype(np.float32)

    packed, scale, zero = quantize_int4(W)
    asym = np.linalg.norm(dequantize_int4(packed, scale, zero, K) - W)

    Wg = W.reshape(K // GROUP, GROUP, N)
    s_sym = np.maximum(np.abs(Wg).max(axis=1) / 7.0, 1e-6)
    q_sym = np.clip(np.round(Wg / s_sym[:, None, :]), -8, 7)
    sym = np.linalg.norm((q_sym * s_sym[:, None, :]).reshape(K, N) - W)

    assert asym < sym, "asymmetric quantization should beat symmetric"


def test_kernel_addressing_matches_reference():
    """The core check: the kernel's index math over several tile shapes."""
    rng = np.random.default_rng(2)
    W = (rng.standard_normal((K, N)) * 0.02).astype(np.float32)
    packed, scale, zero = quantize_int4(W)
    W_hat = dequantize_int4(packed, scale, zero, K)
    x = rng.standard_normal(K).astype(np.float32)
    y_ref = x @ W_hat

    for block_n, block_k in [(64, 64), (32, 128), (128, 32), (64, 128)]:
        y = emulate_dequant_gemv(x, packed, scale, zero, K, N, block_n, block_k)
        err = np.abs(y - y_ref).max() / np.abs(y_ref).max()
        assert err < 1e-5, f"BLOCK_N={block_n} BLOCK_K={block_k}: rel err {err:.2e}"


def test_masking_handles_non_divisible_n():
    """N not a multiple of BLOCK_N exercises the out-of-range mask."""
    rng = np.random.default_rng(3)
    n_odd = 1000
    W = (rng.standard_normal((K, n_odd)) * 0.02).astype(np.float32)
    packed, scale, zero = quantize_int4(W)
    W_hat = dequantize_int4(packed, scale, zero, K)
    x = rng.standard_normal(K).astype(np.float32)

    y = emulate_dequant_gemv(x, packed, scale, zero, K, n_odd, 64, 64)
    ref = x @ W_hat
    assert np.abs(y - ref).max() / np.abs(ref).max() < 1e-5


def test_quantization_error_does_not_average_away():
    """Counterintuitive but important: relative error survives the matmul.

    Signal and error both grow as sqrt(K), so the ratio is preserved. 4-bit
    models work because the *network* is robust, not because the arithmetic
    error is small.
    """
    rng = np.random.default_rng(4)
    W = (rng.standard_normal((K, N)) * 0.02).astype(np.float32)
    packed, scale, zero = quantize_int4(W)
    W_hat = dequantize_int4(packed, scale, zero, K)
    x = rng.standard_normal(K).astype(np.float32)

    w_rel = np.linalg.norm(W_hat - W) / np.linalg.norm(W)
    y_rel = np.linalg.norm(x @ W_hat - x @ W) / np.linalg.norm(x @ W)
    assert 0.8 < y_rel / w_rel < 1.25, f"w={w_rel:.4f} y={y_rel:.4f}"


def test_traffic_reduction_matches_the_claim():
    """The notebook claims ~3.8x less traffic; check the bookkeeping."""
    rng = np.random.default_rng(5)
    W = (rng.standard_normal((K, N)) * 0.02).astype(np.float32)
    packed, scale, zero = quantize_int4(W)

    bytes_fp16 = K * N * 2
    bytes_int4 = packed.nbytes + 2 * scale.nbytes        # scale + zero, fp16
    ratio = bytes_fp16 / bytes_int4
    assert 3.5 < ratio < 4.0, ratio
    bits_per_weight = 8 * bytes_int4 / (K * N)
    assert 4.0 < bits_per_weight < 4.6, bits_per_weight


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ok  {name}")
            passed += 1
    print(f"\n{passed} kernel-math tests passed")
