"""Scalar and binary quantization for embedding compression."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from terrabit._typing import NDArrayF32

QuantizationMethod = Literal[
    "float16",
    "fp8",
    "int8",
    "int4",
    "int3",
    "int2",
    "binary",
    "binary_med",
    "binary_zscore",
    "binary_itq",
    "turbo8",
    "turbo4",
    "turbo3",
    "turbo2",
]


def quantize_float16(x: NDArrayF32) -> np.ndarray[tuple[int, ...], np.dtype[np.float16]]:
    """Lossless-ish cast to float16. ~2x compression."""
    return x.astype(np.float16)


def dequantize_float16(
    x: np.ndarray[tuple[int, ...], np.dtype[np.float16]],
) -> NDArrayF32:
    return x.astype(np.float32)


def quantize_fp8(
    x: NDArrayF32,
) -> np.ndarray[tuple[int, ...], np.dtype[np.uint8]]:
    """Quantize to an E4M3-like FP8 byte encoding (stored as uint8)."""
    x_abs = np.abs(x).astype(np.float32)
    sign = (x < 0).astype(np.uint8)

    eps = np.float32(1e-12)
    exponents = np.floor(np.log2(np.maximum(x_abs, eps))).astype(np.int32)
    exponents = np.clip(exponents, -7, 8)
    exponents_f = exponents.astype(np.float32)

    mantissa = x_abs / np.power(np.float32(2.0), exponents_f) - np.float32(1.0)
    mantissa_q = np.clip(np.round(mantissa * np.float32(8.0)), 0, 7).astype(np.uint8)
    exponent_bits = (exponents + 7).astype(np.uint8)

    encoded = (sign << 7) | (exponent_bits << 3) | mantissa_q
    return np.where(x_abs == 0, np.uint8(0), encoded).astype(np.uint8)


def dequantize_fp8(
    q: np.ndarray[tuple[int, ...], np.dtype[np.uint8]],
) -> NDArrayF32:
    """Dequantize E4M3-like FP8 bytes back to float32."""
    sign = ((q >> 7) & 0x1).astype(np.float32)
    exponent_bits = ((q >> 3) & 0xF).astype(np.int32)
    mantissa_bits = (q & 0x7).astype(np.float32)

    exponent = (exponent_bits - 7).astype(np.float32)
    magnitude = (np.float32(1.0) + mantissa_bits / np.float32(8.0)) * np.power(
        np.float32(2.0), exponent
    )
    reconstructed = np.where(sign > 0, -magnitude, magnitude).astype(np.float32)
    reconstructed = np.where(q == 0, np.float32(0.0), reconstructed)
    return reconstructed.astype(np.float32)


def quantize_int8(
    x: NDArrayF32,
) -> tuple[np.ndarray[tuple[int, ...], np.dtype[np.int8]], NDArrayF32, NDArrayF32]:
    """Per-channel affine quantization to int8. Returns (quantized, scale, zero_point)."""
    vmin = x.min(axis=0, keepdims=True).astype(np.float32)
    vmax = x.max(axis=0, keepdims=True).astype(np.float32)
    scale = (vmax - vmin) / 255.0
    scale = np.where(scale > 0, scale, np.float32(1.0))
    zero_point = vmin
    quantized = np.clip(np.round((x - zero_point) / scale) - 128, -128, 127).astype(np.int8)
    return quantized, scale.squeeze(0), zero_point.squeeze(0)


def dequantize_int8(
    q: np.ndarray[tuple[int, ...], np.dtype[np.int8]],
    scale: NDArrayF32,
    zero_point: NDArrayF32,
) -> NDArrayF32:
    return (q.astype(np.float32) + 128) * scale + zero_point


def _pack_bits(
    q_values: np.ndarray[tuple[int, ...], np.dtype[np.uint8]],
    bits: int,
) -> tuple[np.ndarray[tuple[int, ...], np.dtype[np.uint8]], int]:
    """Pack fixed-width values into bytes along feature axis."""
    if bits not in (2, 3, 4):
        msg = f"Unsupported bit width for packing: {bits}"
        raise ValueError(msg)

    n, d = q_values.shape
    total_bits = d * bits
    n_bytes = (total_bits + 7) // 8
    packed = np.zeros((n, n_bytes), dtype=np.uint8)

    byte_idx = 0
    bit_offset = 0
    mask = (1 << bits) - 1

    for dim_idx in range(d):
        value = q_values[:, dim_idx] & mask
        remaining = 8 - bit_offset
        if bits <= remaining:
            shift = remaining - bits
            packed[:, byte_idx] |= (value << shift).astype(np.uint8)
            bit_offset += bits
            if bit_offset == 8:
                byte_idx += 1
                bit_offset = 0
        else:
            high_bits = bits - remaining
            packed[:, byte_idx] |= (value >> high_bits).astype(np.uint8)
            byte_idx += 1
            packed[:, byte_idx] |= (value << (8 - high_bits)).astype(np.uint8)
            bit_offset = high_bits

    return packed, d


def _unpack_bits(
    packed: np.ndarray[tuple[int, ...], np.dtype[np.uint8]],
    n_dims: int,
    bits: int,
) -> np.ndarray[tuple[int, ...], np.dtype[np.uint8]]:
    """Unpack fixed-width values from bytes along feature axis (vectorized).

    2-bit and 4-bit use direct per-nibble/per-dibit bit-shifts (~40x faster
    than a generic approach).  3-bit falls back to unpackbits+reshape (no clean
    byte alignment) but still avoids the slow dot-product with powers.
    """
    if bits not in (2, 3, 4):
        msg = f"Unsupported bit width for unpacking: {bits}"
        raise ValueError(msg)

    if bits == 2:
        # 4 values per byte at bit positions [6:8, 4:6, 2:4, 0:2]
        out = np.empty((packed.shape[0], packed.shape[1] * 4), dtype=np.uint8)
        out[:, 0::4] = (packed >> 6) & 0x3
        out[:, 1::4] = (packed >> 4) & 0x3
        out[:, 2::4] = (packed >> 2) & 0x3
        out[:, 3::4] = packed & 0x3
        return out[:, :n_dims]

    if bits == 4:
        # 2 values per byte: high nibble then low nibble
        out = np.empty((packed.shape[0], packed.shape[1] * 2), dtype=np.uint8)
        out[:, 0::2] = (packed >> 4) & 0xF
        out[:, 1::2] = packed & 0xF
        return out[:, :n_dims]

    # bits == 3: no clean byte alignment — use unpackbits then explicit shifts
    bits_2d = np.unpackbits(packed, axis=1, bitorder="big")[:, : n_dims * 3]
    bits_3d = bits_2d.reshape(packed.shape[0], n_dims, 3)
    return ((bits_3d[:, :, 0] << 2) | (bits_3d[:, :, 1] << 1) | bits_3d[:, :, 2]).astype(np.uint8)


def _quantize_intn(
    x: NDArrayF32,
    bits: int,
) -> tuple[np.ndarray[tuple[int, ...], np.dtype[np.uint8]], NDArrayF32, NDArrayF32, int]:
    levels = float((1 << bits) - 1)
    vmin = x.min(axis=0, keepdims=True).astype(np.float32)
    vmax = x.max(axis=0, keepdims=True).astype(np.float32)
    scale = (vmax - vmin) / np.float32(levels)
    scale = np.where(scale > 0, scale, np.float32(1.0))
    zero_point = vmin

    q_values = np.clip(np.round((x - zero_point) / scale), 0, int(levels)).astype(np.uint8)
    packed, n_dims = _pack_bits(q_values, bits)
    return packed, scale.squeeze(0), zero_point.squeeze(0), n_dims


def _dequantize_intn(
    packed: np.ndarray[tuple[int, ...], np.dtype[np.uint8]],
    scale: NDArrayF32,
    zero_point: NDArrayF32,
    n_dims: int,
    bits: int,
) -> NDArrayF32:
    q_values = _unpack_bits(packed, n_dims, bits).astype(np.float32)
    return q_values * scale + zero_point


def quantize_int4(
    x: NDArrayF32,
) -> tuple[np.ndarray[tuple[int, ...], np.dtype[np.uint8]], NDArrayF32, NDArrayF32, int]:
    """Per-channel affine quantization to 4-bit values packed in bytes."""
    return _quantize_intn(x, bits=4)


def dequantize_int4(
    packed: np.ndarray[tuple[int, ...], np.dtype[np.uint8]],
    scale: NDArrayF32,
    zero_point: NDArrayF32,
    n_dims: int,
) -> NDArrayF32:
    """Dequantize packed int4 bytes back to float32."""
    return _dequantize_intn(packed, scale, zero_point, n_dims, bits=4)


def quantize_int3(
    x: NDArrayF32,
) -> tuple[np.ndarray[tuple[int, ...], np.dtype[np.uint8]], NDArrayF32, NDArrayF32, int]:
    """Per-channel affine quantization to 3-bit values packed in bytes."""
    return _quantize_intn(x, bits=3)


def dequantize_int3(
    packed: np.ndarray[tuple[int, ...], np.dtype[np.uint8]],
    scale: NDArrayF32,
    zero_point: NDArrayF32,
    n_dims: int,
) -> NDArrayF32:
    """Dequantize packed int3 bytes back to float32."""
    return _dequantize_intn(packed, scale, zero_point, n_dims, bits=3)


def quantize_int2(
    x: NDArrayF32,
) -> tuple[np.ndarray[tuple[int, ...], np.dtype[np.uint8]], NDArrayF32, NDArrayF32, int]:
    """Per-channel affine quantization to 2-bit values packed in bytes."""
    return _quantize_intn(x, bits=2)


def dequantize_int2(
    packed: np.ndarray[tuple[int, ...], np.dtype[np.uint8]],
    scale: NDArrayF32,
    zero_point: NDArrayF32,
    n_dims: int,
) -> NDArrayF32:
    """Dequantize packed int2 bytes back to float32."""
    return _dequantize_intn(packed, scale, zero_point, n_dims, bits=2)


def quantize_binary(x: NDArrayF32) -> np.ndarray[tuple[int, ...], np.dtype[np.uint8]]:
    """Sign-bit quantization: each dim -> 1 bit, packed into uint8. ~32x compression.

    Returns packed binary array of shape (n_samples, ceil(n_dims / 8)).
    """
    signs = (x > 0).astype(np.uint8)
    _, d = signs.shape
    pad = (8 - d % 8) % 8
    if pad > 0:
        signs = np.pad(signs, ((0, 0), (0, pad)))
    return np.packbits(signs, axis=1)


def dequantize_binary(
    packed: np.ndarray[tuple[int, ...], np.dtype[np.uint8]],
    n_dims: int,
) -> NDArrayF32:
    """Unpack binary to +1/-1 float32."""
    bits = np.unpackbits(packed, axis=1)[:, :n_dims]
    return 2.0 * bits.astype(np.float32) - 1.0


def quantize_binary_med(
    x: NDArrayF32,
) -> tuple[np.ndarray[tuple[int, ...], np.dtype[np.uint8]], NDArrayF32]:
    """Per-dimension median-threshold binary quantization."""
    threshold = np.median(x, axis=0).astype(np.float32)
    bits = (x > threshold).astype(np.uint8)
    pad = (8 - bits.shape[1] % 8) % 8
    if pad > 0:
        bits = np.pad(bits, ((0, 0), (0, pad)))
    return np.packbits(bits, axis=1), threshold


def dequantize_binary_med(
    packed: np.ndarray[tuple[int, ...], np.dtype[np.uint8]],
    n_dims: int,
) -> NDArrayF32:
    """Unpack binary_med codes into +/-1 around learned thresholds."""
    bits = np.unpackbits(packed, axis=1)[:, :n_dims].astype(np.float32)
    return np.where(bits > 0, np.float32(1.0), np.float32(-1.0))


def quantize_binary_zscore(
    x: NDArrayF32,
) -> tuple[np.ndarray[tuple[int, ...], np.dtype[np.uint8]], NDArrayF32, NDArrayF32]:
    """Z-score per dimension, then sign threshold at zero."""
    mean = x.mean(axis=0).astype(np.float32)
    std = x.std(axis=0).astype(np.float32)
    std = np.where(std > 1e-8, std, np.float32(1.0))
    z = (x - mean) / std
    bits = (z > 0).astype(np.uint8)
    pad = (8 - bits.shape[1] % 8) % 8
    if pad > 0:
        bits = np.pad(bits, ((0, 0), (0, pad)))
    return np.packbits(bits, axis=1), mean, std


def dequantize_binary_zscore(
    packed: np.ndarray[tuple[int, ...], np.dtype[np.uint8]],
    n_dims: int,
    mean: NDArrayF32,
    std: NDArrayF32,
) -> NDArrayF32:
    """Map unpacked bits back to standardized +/-1 and un-normalize."""
    bits = np.unpackbits(packed, axis=1)[:, :n_dims]
    z = np.where(bits > 0, np.float32(1.0), np.float32(-1.0))
    return z.astype(np.float32) * std + mean


def _pca_whiten_full(x: NDArrayF32) -> tuple[NDArrayF32, NDArrayF32, NDArrayF32, NDArrayF32]:
    """Return whitened coordinates and PCA params for full-dimensional transform."""
    mean = x.mean(axis=0).astype(np.float32)
    xc = x - mean
    cov = np.cov(xc, rowvar=False).astype(np.float32)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    components = eigvecs[:, order].astype(np.float32)
    std = np.sqrt(np.clip(eigvals, 1e-8, None)).astype(np.float32)
    z = (xc @ components) / std
    return z.astype(np.float32), mean, components, std


def quantize_binary_itq(
    x: NDArrayF32,
    *,
    seed: int = 0,
    n_iter: int = 20,
) -> tuple[
    np.ndarray[tuple[int, ...], np.dtype[np.uint8]], NDArrayF32, NDArrayF32, NDArrayF32, NDArrayF32
]:
    """ITQ-style binary coding: PCA whitening + learned orthogonal rotation + sign."""
    z, mean, components, std = _pca_whiten_full(x)
    d = z.shape[1]

    rng = np.random.default_rng(seed)
    r0, _ = np.linalg.qr(rng.standard_normal((d, d)).astype(np.float32))
    rotation = r0.astype(np.float32)

    for _ in range(n_iter):
        b = np.where(z @ rotation >= 0, np.float32(1.0), np.float32(-1.0))
        u, _s, vt = np.linalg.svd(b.T @ z, full_matrices=False)
        rotation = (vt.T @ u.T).astype(np.float32)

    bits = ((z @ rotation) > 0).astype(np.uint8)
    pad = (8 - bits.shape[1] % 8) % 8
    if pad > 0:
        bits = np.pad(bits, ((0, 0), (0, pad)))
    packed = np.packbits(bits, axis=1)
    return packed, mean, components, std, rotation


def dequantize_binary_itq(
    packed: np.ndarray[tuple[int, ...], np.dtype[np.uint8]],
    n_dims: int,
    mean: NDArrayF32,
    components: NDArrayF32,
    std: NDArrayF32,
    rotation: NDArrayF32,
) -> NDArrayF32:
    """Approximate inverse for ITQ-style codes."""
    bits = np.unpackbits(packed, axis=1)[:, :n_dims]
    b = np.where(bits > 0, np.float32(1.0), np.float32(-1.0)).astype(np.float32)
    z_hat = b @ rotation.T
    x_pca = z_hat * std
    return (x_pca @ components.T + mean).astype(np.float32)


_ORTHO_MATRIX_CACHE: dict[tuple[int, int], np.ndarray] = {}


def _random_orthogonal_matrix(d: int, seed: int = 0) -> np.ndarray:
    """Generate a deterministic random orthogonal matrix via QR decomposition.

    Result is cached per (d, seed) — QR on large d is expensive and the matrix
    is identical across all calls with the same arguments.
    """
    key = (d, seed)
    if key not in _ORTHO_MATRIX_CACHE:
        rng = np.random.default_rng(seed)
        z = rng.standard_normal((d, d)).astype(np.float32)
        q, r = np.linalg.qr(z)
        # Ensure uniform Haar distribution: fix sign ambiguity from QR
        q *= np.sign(np.diag(r))[np.newaxis, :]
        _ORTHO_MATRIX_CACHE[key] = q
    return _ORTHO_MATRIX_CACHE[key]


def quantize_turbo(
    x: NDArrayF32,
    bits: int,
    seed: int = 0,
) -> tuple[np.ndarray, NDArrayF32, NDArrayF32, int, int]:
    """TurboQuant: random rotation + per-channel affine quantization.

    Rotating by a random orthogonal matrix spreads information uniformly across
    dimensions, making per-channel scalar quantization near-optimal.

    Returns (quantized, scale, zero_point, n_dims, seed).
    """
    d = x.shape[1]
    r_matrix = _random_orthogonal_matrix(d, seed=seed)
    x_rot = x @ r_matrix  # (n, d) @ (d, d) -> (n, d); distances preserved
    if bits == 8:
        q, scale, zp = quantize_int8(x_rot)
        return q, scale, zp, d, seed
    packed, scale, zp, n_dims = _quantize_intn(x_rot, bits)
    return packed, scale, zp, n_dims, seed


def dequantize_turbo(
    quantized: np.ndarray,
    scale: NDArrayF32,
    zero_point: NDArrayF32,
    n_dims: int,
    bits: int,
    seed: int = 0,
) -> NDArrayF32:
    """Dequantize TurboQuant: unpack, inverse-rotate back to original space."""
    if bits == 8:
        x_rot = dequantize_int8(quantized, scale, zero_point)
    else:
        x_rot = _dequantize_intn(quantized, scale, zero_point, n_dims, bits)
    r_matrix = _random_orthogonal_matrix(n_dims, seed=seed)
    return x_rot @ r_matrix.T


def hamming_distance(
    a: np.ndarray[tuple[int, ...], np.dtype[np.uint8]],
    b: np.ndarray[tuple[int, ...], np.dtype[np.uint8]],
) -> np.ndarray[tuple[int, ...], np.dtype[np.int64]]:
    """Hamming distance between packed binary vectors. a: (n, k), b: (m, k) -> (n, m)."""
    xor = np.bitwise_xor(a[:, np.newaxis, :], b[np.newaxis, :, :])
    popcount_lut = np.array([i.bit_count() for i in range(256)], dtype=np.int64)
    return popcount_lut[xor].sum(axis=2)


def quantize(
    x: NDArrayF32,
    method: QuantizationMethod,
) -> dict[str, Any]:
    """Quantize embeddings. Returns dict with quantized data + any calibration params."""
    if method == "float16":
        return {"quantized": quantize_float16(x), "method": "float16"}
    if method == "fp8":
        return {"quantized": quantize_fp8(x), "method": "fp8"}
    if method == "int8":
        q, scale, zp = quantize_int8(x)
        return {"quantized": q, "scale": scale, "zero_point": zp, "method": "int8"}
    if method == "int4":
        q, scale, zp, n_dims = quantize_int4(x)
        return {
            "quantized": q,
            "scale": scale,
            "zero_point": zp,
            "n_dims": n_dims,
            "method": "int4",
        }
    if method == "int3":
        q, scale, zp, n_dims = quantize_int3(x)
        return {
            "quantized": q,
            "scale": scale,
            "zero_point": zp,
            "n_dims": n_dims,
            "method": "int3",
        }
    if method == "int2":
        q, scale, zp, n_dims = quantize_int2(x)
        return {
            "quantized": q,
            "scale": scale,
            "zero_point": zp,
            "n_dims": n_dims,
            "method": "int2",
        }
    if method == "binary":
        return {
            "quantized": quantize_binary(x),
            "n_dims": x.shape[1],
            "method": "binary",
        }
    if method == "binary_med":
        q, threshold = quantize_binary_med(x)
        return {
            "quantized": q,
            "n_dims": x.shape[1],
            "threshold": threshold,
            "method": "binary_med",
        }
    if method == "binary_zscore":
        q, mean, std = quantize_binary_zscore(x)
        return {
            "quantized": q,
            "n_dims": x.shape[1],
            "mean": mean,
            "std": std,
            "method": "binary_zscore",
        }
    if method == "binary_itq":
        q, mean, components, std, rotation = quantize_binary_itq(x)
        return {
            "quantized": q,
            "n_dims": x.shape[1],
            "mean": mean,
            "components": components,
            "std": std,
            "rotation": rotation,
            "method": "binary_itq",
        }
    if method in ("turbo8", "turbo4", "turbo3", "turbo2"):
        bits = int(method[5:])
        q, scale, zp, n_dims, seed = quantize_turbo(x, bits=bits)
        return {
            "quantized": q,
            "scale": scale,
            "zero_point": zp,
            "n_dims": n_dims,
            "turbo_bits": bits,
            "turbo_seed": seed,
            "method": method,
        }
    msg = f"Unknown quantization method: {method}"
    raise ValueError(msg)


def dequantize(result: dict[str, Any]) -> NDArrayF32:
    """Dequantize back to float32 (lossy reconstruction)."""
    method = result["method"]
    if method == "float16":
        return dequantize_float16(result["quantized"])
    if method == "fp8":
        return dequantize_fp8(result["quantized"])
    if method == "int8":
        return dequantize_int8(result["quantized"], result["scale"], result["zero_point"])
    if method == "int4":
        return dequantize_int4(
            result["quantized"],
            result["scale"],
            result["zero_point"],
            result["n_dims"],
        )
    if method == "int3":
        return dequantize_int3(
            result["quantized"],
            result["scale"],
            result["zero_point"],
            result["n_dims"],
        )
    if method == "int2":
        return dequantize_int2(
            result["quantized"],
            result["scale"],
            result["zero_point"],
            result["n_dims"],
        )
    if method == "binary":
        return dequantize_binary(result["quantized"], result["n_dims"])
    if method == "binary_med":
        return dequantize_binary_med(result["quantized"], result["n_dims"])
    if method == "binary_zscore":
        return dequantize_binary_zscore(
            result["quantized"],
            result["n_dims"],
            result["mean"],
            result["std"],
        )
    if method == "binary_itq":
        return dequantize_binary_itq(
            result["quantized"],
            result["n_dims"],
            result["mean"],
            result["components"],
            result["std"],
            result["rotation"],
        )
    if method in ("turbo8", "turbo4", "turbo3", "turbo2"):
        return dequantize_turbo(
            result["quantized"],
            result["scale"],
            result["zero_point"],
            result["n_dims"],
            bits=result["turbo_bits"],
            seed=result["turbo_seed"],
        )
    msg = f"Unknown quantization method: {method}"
    raise ValueError(msg)
