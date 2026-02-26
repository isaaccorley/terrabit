"""Scalar and binary quantization for embedding compression."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from terrabit._typing import NDArrayF32

QuantizationMethod = Literal["float16", "fp8", "int8", "int4", "int3", "int2", "binary"]


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
    encoded = np.where(x_abs == 0, np.uint8(0), encoded).astype(np.uint8)
    return encoded


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
    """Unpack fixed-width values from bytes along feature axis."""
    if bits not in (2, 3, 4):
        msg = f"Unsupported bit width for unpacking: {bits}"
        raise ValueError(msg)

    n = packed.shape[0]
    unpacked = np.zeros((n, n_dims), dtype=np.uint8)

    byte_idx = 0
    bit_offset = 0
    mask = (1 << bits) - 1

    for dim_idx in range(n_dims):
        remaining = 8 - bit_offset
        if bits <= remaining:
            shift = remaining - bits
            unpacked[:, dim_idx] = (packed[:, byte_idx] >> shift) & mask
            bit_offset += bits
            if bit_offset == 8:
                byte_idx += 1
                bit_offset = 0
        else:
            high_bits = bits - remaining
            part1 = (packed[:, byte_idx] & ((1 << remaining) - 1)) << high_bits
            byte_idx += 1
            part2 = packed[:, byte_idx] >> (8 - high_bits)
            unpacked[:, dim_idx] = (part1 | part2) & mask
            bit_offset = high_bits

    return unpacked


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
    n, d = signs.shape
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
    return (2.0 * bits.astype(np.float32) - 1.0)


def hamming_distance(
    a: np.ndarray[tuple[int, ...], np.dtype[np.uint8]],
    b: np.ndarray[tuple[int, ...], np.dtype[np.uint8]],
) -> np.ndarray[tuple[int, ...], np.dtype[np.int64]]:
    """Hamming distance between packed binary vectors. a: (n, k), b: (m, k) -> (n, m)."""
    xor = np.bitwise_xor(a[:, np.newaxis, :], b[np.newaxis, :, :])
    popcount_lut = np.array([bin(i).count("1") for i in range(256)], dtype=np.int64)
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
    msg = f"Unknown quantization method: {method}"
    raise ValueError(msg)
