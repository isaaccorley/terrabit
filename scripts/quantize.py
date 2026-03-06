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
ALL_METHODS: tuple[QuantizationMethod, ...] = (
    "float16",
    "fp8",
    "int8",
    "int4",
    "int3",
    "int2",
    "binary",
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


def _process_one_file(
    src_file: str,
    src_root: str,
    out_root: str,
    methods: tuple[QuantizationMethod, ...],
    embedding_col: str,
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
        if "scale" in quantized:
            quant_meta["scale"] = np.asarray(quantized["scale"], dtype=np.float32).tolist()
        if "zero_point" in quantized:
            quant_meta["zero_point"] = np.asarray(
                quantized["zero_point"], dtype=np.float32
            ).tolist()
        if "turbo_bits" in quantized:
            quant_meta["turbo_bits"] = int(quantized["turbo_bits"])
        if "turbo_seed" in quantized:
            quant_meta["turbo_seed"] = int(quantized["turbo_seed"])

        schema_meta = dict(quantized_table.schema.metadata or {})
        schema_meta[b"quantization"] = json.dumps(quant_meta, separators=(",", ":")).encode("utf-8")
        quantized_table = quantized_table.replace_schema_metadata(schema_meta)

        out_file = output_root / method / rel_path
        out_file.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(quantized_table, out_file)

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
