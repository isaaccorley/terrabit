"""Benchmark storage encodings on a shared embedding sample.

Compares two storage tracks on the same sampled embeddings:
- Parquet-native encodings/compression, including BYTE_STREAM_SPLIT
- External codecs on flattened numeric arrays, including pcodec

Usage:
  uv run python scripts/run_storage_bench.py embeddings/ -o results/report_storage_100k.json
"""

from __future__ import annotations

import argparse
import importlib
import json
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from rich.console import Console

from terrabit.io import reservoir_subsample
from terrabit.quantization import QuantizationMethod, dequantize, quantize

console = Console()
_pcodec = cast("Any", importlib.import_module("pcodec"))
_standalone = cast("Any", importlib.import_module("pcodec.standalone"))


def _make_chunk_config(**kwargs: Any) -> Any:
    return _pcodec.ChunkConfig(**kwargs)

DEFAULT_METHODS: tuple[QuantizationMethod, ...] = (
    "float16",
    "fp8",
    "int8",
    "int4",
    "int2",
    "binary",
)


@dataclass(frozen=True)
class ParquetConfig:
    name: str
    compression: str | None
    compression_level: int | None
    use_byte_stream_split: bool
    use_dictionary: bool
    data_page_version: str


@dataclass(frozen=True)
class ExternalCodecConfig:
    name: str
    codec: str
    layout: str


def _parse_methods(methods_arg: str) -> tuple[QuantizationMethod, ...]:
    methods = tuple(m.strip() for m in methods_arg.split(",") if m.strip())
    invalid = sorted(set(methods) - set(DEFAULT_METHODS))
    if invalid:
        msg = f"Unknown methods: {invalid}. Allowed: {DEFAULT_METHODS}"
        raise ValueError(msg)
    return cast("tuple[QuantizationMethod, ...]", methods)


PARQUET_CONFIGS: tuple[ParquetConfig, ...] = (
    ParquetConfig("snappy", "snappy", None, False, True, "1.0"),
    ParquetConfig("zstd", "zstd", 1, False, True, "1.0"),
    ParquetConfig("zstd_byte_stream_split", "zstd", 1, True, False, "2.0"),
)

EXTERNAL_CONFIGS: tuple[ExternalCodecConfig, ...] = (
    ExternalCodecConfig("raw", "raw", "row_major"),
    ExternalCodecConfig("zstd_row_major", "zstd", "row_major"),
    ExternalCodecConfig("zstd_dim_major", "zstd", "dim_major"),
    ExternalCodecConfig("pcodec_row_major", "pcodec", "row_major"),
    ExternalCodecConfig("pcodec_dim_major", "pcodec", "dim_major"),
)


def _fixed_size_list(values_2d: np.ndarray) -> pa.FixedSizeListArray:
    flat = pa.array(values_2d.reshape(-1), type=pa.from_numpy_dtype(values_2d.dtype))
    return pa.FixedSizeListArray.from_arrays(flat, values_2d.shape[1])


def _make_quant_meta(
    method: str,
    original_dims: int,
    quantized: dict[str, Any],
) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "method": method,
        "embedding_col": "embedding",
        "n_dims": int(quantized.get("n_dims", original_dims)),
    }
    for key, value in quantized.items():
        if key in {"quantized", "method"}:
            continue
        if isinstance(value, np.ndarray):
            meta[key] = np.asarray(value, dtype=np.float32).tolist()
        elif isinstance(value, (np.integer, int)):
            meta[key] = int(value)
        elif isinstance(value, (np.floating, float)):
            meta[key] = float(value)
    return meta


def _dequantize_with_meta(q_values: np.ndarray, meta: dict[str, Any]) -> np.ndarray:
    dq_input: dict[str, Any] = {"method": meta["method"], "quantized": q_values}
    for key, value in meta.items():
        if key in {"method", "embedding_col"}:
            continue
        if isinstance(value, list):
            dq_input[key] = np.asarray(value, dtype=np.float32)
        else:
            dq_input[key] = value
    return dequantize(dq_input)


def _restore_2d(flat: np.ndarray, shape: tuple[int, int], layout: str) -> np.ndarray:
    if layout == "row_major":
        return np.ascontiguousarray(flat.reshape(shape))
    if layout == "dim_major":
        return np.ascontiguousarray(flat.reshape((shape[1], shape[0])).T)
    msg = f"Unknown layout: {layout}"
    raise ValueError(msg)


def _reshape_for_layout(values_2d: np.ndarray, layout: str) -> np.ndarray:
    if layout == "row_major":
        return np.ascontiguousarray(values_2d.reshape(-1))
    if layout == "dim_major":
        return np.ascontiguousarray(values_2d.T.reshape(-1))
    msg = f"Unknown layout: {layout}"
    raise ValueError(msg)


def _serialize_raw(values_2d: np.ndarray, layout: str) -> bytes:
    flat = _reshape_for_layout(values_2d, layout)
    return flat.tobytes(order="C")


def _compress_zstd(raw_bytes: bytes) -> bytes:
    codec = pa.Codec("zstd", compression_level=1)
    return codec.compress(raw_bytes)


def _decompress_zstd(blob: bytes, expected_len: int) -> bytes:
    codec = pa.Codec("zstd", compression_level=1)
    return codec.decompress(blob, expected_len)


def _compress_pcodec(values_2d: np.ndarray, layout: str) -> bytes:
    flat = _reshape_for_layout(values_2d, layout)
    enable_8_bit = flat.dtype.itemsize == 1
    return _standalone.simple_compress(
        flat,
        _make_chunk_config(compression_level=8, enable_8_bit=enable_8_bit),
    )


def _decompress_pcodec(blob: bytes) -> np.ndarray:
    restored = _standalone.simple_decompress(blob)
    if not isinstance(restored, np.ndarray):
        raise TypeError("pcodec returned non-array result")
    return restored


def _benchmark_parquet(
    values_2d: np.ndarray,
    original_dims: int,
    config: ParquetConfig,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    table = pa.table({"embedding": _fixed_size_list(values_2d)})
    if meta is not None:
        schema_meta = dict(table.schema.metadata or {})
        schema_meta[b"quantization"] = json.dumps(meta, separators=(",", ":")).encode("utf-8")
        table = table.replace_schema_metadata(schema_meta)

    with tempfile.TemporaryDirectory(prefix="terrabit_storage_bench_") as tmp_dir:
        out_path = Path(tmp_dir) / "sample.parquet"

        t0 = time.perf_counter()
        kwargs: dict[str, Any] = {
            "compression_level": config.compression_level,
            "use_byte_stream_split": config.use_byte_stream_split,
            "use_dictionary": config.use_dictionary,
            "data_page_version": config.data_page_version,
        }
        if config.compression is not None:
            kwargs["compression"] = config.compression
        pq.write_table(table, out_path, **kwargs)
        write_s = time.perf_counter() - t0

        file_size = out_path.stat().st_size

        t1 = time.perf_counter()
        restored = pq.read_table(out_path).column("embedding")
        read_s = time.perf_counter() - t1

        arr = restored.chunk(0) if len(restored.chunks) == 1 else restored.combine_chunks()
        q_values = arr.values.to_numpy(zero_copy_only=False).reshape((len(arr), arr.type.list_size))
        max_abs_error = float(np.max(np.abs(q_values.astype(np.float32) - values_2d.astype(np.float32))))

        dequantize_s: float | None = None
        if meta is not None:
            t2 = time.perf_counter()
            _dequantize_with_meta(q_values, meta)
            dequantize_s = time.perf_counter() - t2

        return {
            "storage_backend": "parquet",
            "config": config.name,
            "bytes": file_size,
            "bytes_per_vector": file_size / values_2d.shape[0],
            "bits_per_dim": (file_size * 8) / (values_2d.shape[0] * original_dims),
            "write_s": write_s,
            "read_s": read_s,
            "dequantize_s": dequantize_s,
            "max_abs_error": max_abs_error,
        }


def _benchmark_external(
    values_2d: np.ndarray,
    original_dims: int,
    config: ExternalCodecConfig,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    raw_bytes = _serialize_raw(values_2d, config.layout)

    t0 = time.perf_counter()
    if config.codec == "raw":
        blob = raw_bytes
    elif config.codec == "zstd":
        blob = _compress_zstd(raw_bytes)
    elif config.codec == "pcodec":
        blob = _compress_pcodec(values_2d, config.layout)
    else:
        raise ValueError(f"Unknown codec: {config.codec}")
    write_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    if config.codec == "raw":
        restored = np.frombuffer(blob, dtype=values_2d.dtype)
    elif config.codec == "zstd":
        restored = np.frombuffer(_decompress_zstd(blob, len(raw_bytes)), dtype=values_2d.dtype)
    else:
        restored = _decompress_pcodec(blob)
    read_s = time.perf_counter() - t1

    restored_2d = _restore_2d(restored, values_2d.shape, config.layout)
    max_abs_error = float(np.max(np.abs(restored_2d.astype(np.float32) - values_2d.astype(np.float32))))

    dequantize_s: float | None = None
    if meta is not None:
        t2 = time.perf_counter()
        _dequantize_with_meta(restored_2d, meta)
        dequantize_s = time.perf_counter() - t2

    return {
        "storage_backend": "external",
        "config": config.name,
        "codec": config.codec,
        "layout": config.layout,
        "bytes": len(blob),
        "bytes_per_vector": len(blob) / values_2d.shape[0],
        "bits_per_dim": (len(blob) * 8) / (values_2d.shape[0] * original_dims),
        "write_s": write_s,
        "read_s": read_s,
        "dequantize_s": dequantize_s,
        "restored_values": int(restored.size),
        "max_abs_error": max_abs_error,
    }


def _quantize_sample(
    sample: np.ndarray,
    methods: tuple[QuantizationMethod, ...],
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for method in methods:
        console.print(f"[dim]quantize {method}[/dim]")
        out[method] = quantize(sample, method)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Path to source parquet embeddings directory")
    parser.add_argument(
        "--output",
        "-o",
        default="results/report_storage_100k.json",
        help="Output JSON report path",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100_000,
        help="Reservoir sample size",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50_000,
        help="Batch size for reservoir sampling",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Sampling seed",
    )
    parser.add_argument(
        "--methods",
        default=",".join(DEFAULT_METHODS),
        help="Comma-separated quantization methods",
    )
    parser.add_argument(
        "--embedding-col",
        default="embedding",
        help="Embedding column name",
    )
    args = parser.parse_args()

    methods = _parse_methods(args.methods)

    console.rule("[bold]Storage Benchmark")
    console.print(f"source: {args.path}")
    console.print(f"sample_size: {args.sample_size}  seed: {args.seed}")
    console.print(f"methods: {methods}")

    sample = reservoir_subsample(
        args.path,
        n_samples=args.sample_size,
        batch_size=args.batch_size,
        embedding_col=args.embedding_col,
        seed=args.seed,
    )
    console.print(f"sampled: {sample.shape[0]} rows x {sample.shape[1]} dims")

    quantized = _quantize_sample(sample, methods)

    report: dict[str, Any] = {
        "source_path": args.path,
        "sample_size": int(sample.shape[0]),
        "n_dims": int(sample.shape[1]),
        "seed": args.seed,
        "methods": {},
        "parquet_configs": [asdict(cfg) for cfg in PARQUET_CONFIGS],
        "external_configs": [asdict(cfg) for cfg in EXTERNAL_CONFIGS],
    }

    report["methods"]["float32"] = {
        "quantized_shape": list(sample.shape),
        "quantized_dtype": str(sample.dtype),
        "quantization_meta": None,
        "parquet": [
            _benchmark_parquet(sample, sample.shape[1], cfg) for cfg in PARQUET_CONFIGS
        ],
        "external": [
            _benchmark_external(sample, sample.shape[1], cfg) for cfg in EXTERNAL_CONFIGS
        ],
    }

    for method, q in quantized.items():
        q_values = np.asarray(q["quantized"])
        meta = _make_quant_meta(method, sample.shape[1], q)
        console.print(f"[bold]{method}[/bold]  stored_shape={tuple(q_values.shape)}")

        parquet_results = [
            _benchmark_parquet(q_values, sample.shape[1], cfg, meta) for cfg in PARQUET_CONFIGS
        ]
        external_results = [
            _benchmark_external(q_values, sample.shape[1], cfg, meta) for cfg in EXTERNAL_CONFIGS
        ]

        report["methods"][method] = {
            "quantized_shape": list(q_values.shape),
            "quantized_dtype": str(q_values.dtype),
            "quantization_meta": meta,
            "parquet": parquet_results,
            "external": external_results,
        }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    console.print(f"[bold green]saved[/bold green] {output_path}")


if __name__ == "__main__":
    main()
