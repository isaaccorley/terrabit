"""Quantize parquet embeddings while preserving dataset structure and columns.

The script mirrors the source directory tree under output folders (one per
quantization method) and replaces only the embedding column in each parquet
file. All non-embedding columns and schema metadata are preserved.
Calibration metadata needed for dequantization is written into each output
parquet file's footer metadata.

Usage:
        uv run python scripts/quantize.py embeddings/ -o results/quantized_dataset
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from rich.console import Console
from threadpoolctl import threadpool_limits

from terrabit.io import iter_parquet_files
from terrabit.quantization import QuantizationMethod, quantize

console = Console()


@dataclass(frozen=True)
class ParquetWriteOptions:
    compression: str | None = "snappy"
    compression_level: int | None = None
    use_byte_stream_split: bool = False
    row_group_size: int | None = None

    @property
    def use_dictionary(self) -> bool:
        return not self.use_byte_stream_split

    @property
    def data_page_version(self) -> str:
        return "2.0" if self.use_byte_stream_split else "1.0"


ALL_METHODS: tuple[QuantizationMethod, ...] = (
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
)


def _resolve_jobs(*, requested_jobs: int, n_files: int, n_methods: int) -> int:
    if requested_jobs > 0:
        return max(1, min(requested_jobs, n_files))

    cpu_count = os.cpu_count() or 1
    auto_jobs = max(1, cpu_count // 2)
    if n_methods >= 6:
        auto_jobs = max(1, round(auto_jobs * 0.9))
    auto_jobs = min(auto_jobs, n_files)
    return max(1, auto_jobs)


def _worker_init() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")


def _extract_embeddings_from_chunked_array(col: pa.ChunkedArray) -> np.ndarray:
    if len(col.chunks) == 1:
        arr = col.chunks[0]
    else:
        arr = col.combine_chunks()

    if isinstance(arr, pa.FixedSizeListArray):
        flat = arr.values.to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
        return flat.reshape((len(arr), arr.type.list_size))

    if isinstance(arr, (pa.ListArray, pa.LargeListArray)):
        flat = arr.values.to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
        offsets = arr.offsets.to_numpy()
        dim = int(offsets[1] - offsets[0])
        lengths = np.diff(offsets)
        if not np.all(lengths == dim):
            msg = "Variable-length embedding lists are not supported"
            raise ValueError(msg)
        return flat.reshape((len(arr), dim))

    msg = "Embedding column must be a list/fixed-size-list array"
    raise ValueError(msg)


def _to_fixed_size_list(values_2d: np.ndarray) -> pa.FixedSizeListArray:
    flat = values_2d.reshape(-1)
    value_array = pa.array(flat, type=pa.from_numpy_dtype(values_2d.dtype))
    return pa.FixedSizeListArray.from_arrays(value_array, values_2d.shape[1])


def _write_parquet_table(
    table: pa.Table,
    out_file: Path,
    write_options: ParquetWriteOptions,
) -> None:
    kwargs: dict[str, Any] = {
        "compression_level": write_options.compression_level,
        "use_byte_stream_split": write_options.use_byte_stream_split,
        "use_dictionary": write_options.use_dictionary,
        "data_page_version": write_options.data_page_version,
        "row_group_size": write_options.row_group_size,
    }
    if write_options.compression is not None:
        kwargs["compression"] = write_options.compression
    pq.write_table(table, out_file, **kwargs)


def _process_one_file(
    src_file: str,
    src_root: str,
    out_root: str,
    methods: tuple[QuantizationMethod, ...],
    embedding_col: str,
    write_options: ParquetWriteOptions,
) -> dict[str, Any]:
    threadpool_limits(limits=1)

    src_path = Path(src_file)
    source_root = Path(src_root)
    output_root = Path(out_root)

    rel_path = src_path.relative_to(source_root)
    table = pq.ParquetFile(src_path).read()

    col_index = table.schema.get_field_index(embedding_col)
    if col_index < 0:
        raise ValueError(f"Embedding column '{embedding_col}' not found in {src_path}")

    x = _extract_embeddings_from_chunked_array(table.column(col_index))
    original_field = table.schema.field(col_index)

    output_files: list[str] = []
    for method in methods:
        quantized = quantize(x, method)
        quantized_values = np.asarray(quantized["quantized"])
        quantized_array = _to_fixed_size_list(quantized_values)

        quantized_field = pa.field(
            original_field.name,
            quantized_array.type,
            nullable=original_field.nullable,
            metadata=original_field.metadata,
        )
        quantized_table = table.set_column(col_index, quantized_field, quantized_array)

        quant_meta: dict[str, Any] = {
            "method": method,
            "embedding_col": embedding_col,
            "n_dims": int(quantized.get("n_dims", x.shape[1])),
        }
        for k, v in quantized.items():
            if k in ("quantized", "method"):
                continue
            if isinstance(v, np.ndarray):
                quant_meta[k] = np.asarray(v, dtype=np.float32).tolist()
            elif isinstance(v, (np.integer, int)):
                quant_meta[k] = int(v)
            elif isinstance(v, (np.floating, float)):
                quant_meta[k] = float(v)

        schema_meta = dict(quantized_table.schema.metadata or {})
        schema_meta[b"quantization"] = json.dumps(quant_meta, separators=(",", ":")).encode("utf-8")
        quantized_table = quantized_table.replace_schema_metadata(schema_meta)

        out_file = output_root / method / rel_path
        out_file.parent.mkdir(parents=True, exist_ok=True)
        _write_parquet_table(quantized_table, out_file, write_options)

        output_files.append(str(out_file))

    return {
        "file": str(src_path),
        "rows": int(table.num_rows),
        "dims": int(x.shape[1]),
        "outputs": output_files,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Path to source parquet embeddings directory")
    parser.add_argument(
        "--output",
        "-o",
        default="results/quantized_dataset",
        help="Output root (method subfolders will be created)",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default=",".join(ALL_METHODS),
        help="Comma-separated quantization methods",
    )
    parser.add_argument(
        "--embedding-col",
        default="embedding",
        help="Name of embedding column to quantize",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=0,
        help="Worker processes (0 = auto)",
    )
    parser.add_argument(
        "--limit-files",
        type=int,
        default=0,
        help="If >0, process only first N parquet files (for smoke tests)",
    )
    parser.add_argument(
        "--parquet-compression",
        choices=("none", "snappy", "zstd"),
        default="snappy",
        help="Parquet column compression codec",
    )
    parser.add_argument(
        "--parquet-compression-level",
        type=int,
        default=None,
        help="Optional Parquet compression level",
    )
    parser.add_argument(
        "--parquet-byte-stream-split",
        action="store_true",
        help="Enable Parquet BYTE_STREAM_SPLIT encoding before compression",
    )
    parser.add_argument(
        "--parquet-row-group-size",
        type=int,
        default=None,
        help="Optional Parquet row group size",
    )
    args = parser.parse_args()

    methods = tuple(m.strip() for m in args.methods.split(",") if m.strip())
    invalid = sorted(set(methods) - set(ALL_METHODS))
    if invalid:
        msg = f"Unknown methods: {invalid}. Allowed: {ALL_METHODS}"
        raise ValueError(msg)

    source_root = Path(args.path)
    out_root = Path(args.output)
    file_paths = list(iter_parquet_files(str(source_root)))
    if args.limit_files > 0:
        file_paths = file_paths[: args.limit_files]
    if not file_paths:
        raise ValueError("No parquet files found")

    write_options = ParquetWriteOptions(
        compression=None if args.parquet_compression == "none" else args.parquet_compression,
        compression_level=args.parquet_compression_level,
        use_byte_stream_split=args.parquet_byte_stream_split,
        row_group_size=args.parquet_row_group_size,
    )

    worker_count = _resolve_jobs(
        requested_jobs=args.jobs,
        n_files=len(file_paths),
        n_methods=len(methods),
    )

    console.rule("[bold]Quantizing Dataset")
    console.print(f"source root: {source_root}")
    console.print(f"output root: {out_root}")
    console.print(f"methods: {methods}")
    console.print(f"files: {len(file_paths)}  workers: {worker_count}")
    console.print(
        "parquet: "
        f"compression={write_options.compression or 'none'}  "
        f"byte_stream_split={write_options.use_byte_stream_split}  "
        f"compression_level={write_options.compression_level}  "
        f"row_group_size={write_options.row_group_size}"
    )

    completed = 0
    total_rows = 0
    with ProcessPoolExecutor(max_workers=worker_count, initializer=_worker_init) as executor:
        futures = [
            executor.submit(
                _process_one_file,
                file_path,
                str(source_root),
                str(out_root),
                methods,
                args.embedding_col,
                write_options,
            )
            for file_path in file_paths
        ]
        for future in as_completed(futures):
            result = future.result()
            completed += 1
            total_rows += int(result["rows"])
            if completed % 25 == 0 or completed == len(file_paths):
                console.print(f"processed {completed}/{len(file_paths)} files  rows={total_rows}")

    console.print("\n[bold green]Quantization complete[/bold green]")
    for method in methods:
        console.print(f"  {method}: {out_root / method}")


if __name__ == "__main__":
    main()
