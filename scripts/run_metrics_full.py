"""Run exact quantization metrics by loading full datasets into memory.

This script is for smaller datasets or high-memory machines. It computes exact
metrics (including kNN recall and cosine-structure correlation) by materializing
original and reconstructed embeddings in memory.

Usage:
  uv run python scripts/run_metrics_full.py \
    --original-root embeddings \
    --quantized-root results/quantized_dataset_full_allmethods_j8 \
    -o results/report_metrics_full.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from rich.console import Console

from terrabit.metrics import (
    cosine_similarity_correlation,
    knn_recall_multi,
    reconstruction_cosine,
    reconstruction_mse,
)
from terrabit.quantization import dequantize

console = Console()
KNNMetric = Literal["cosine", "euclidean"]
DEFAULT_KS = (10, 25, 50)
DEFAULT_KNN_METRICS: tuple[KNNMetric, ...] = ("cosine", "euclidean")


def _serialize(obj: object) -> object:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return cast("Any", obj).tolist()
    if isinstance(obj, tuple):
        return list(obj)
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize(v) for v in obj]
    return obj


def _list_parquet_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.parquet"))


def _extract_2d(col: pa.Array | pa.ChunkedArray) -> np.ndarray:
    if isinstance(col, pa.ChunkedArray):
        arr = col.chunks[0] if len(col.chunks) == 1 else col.combine_chunks()
    else:
        arr = col

    if isinstance(arr, pa.FixedSizeListArray):
        values = arr.values.to_numpy(zero_copy_only=False)
        return values.reshape((len(arr), arr.type.list_size))

    if isinstance(arr, (pa.ListArray, pa.LargeListArray)):
        values = arr.values.to_numpy(zero_copy_only=False)
        offsets = arr.offsets.to_numpy()
        dim = int(offsets[1] - offsets[0])
        lengths = np.diff(offsets)
        if not np.all(lengths == dim):
            raise ValueError("Variable-length embedding lists are not supported")
        return values.reshape((len(arr), dim))

    raise ValueError(f"Unsupported array type: {type(arr)}")


def _parse_quant_meta(schema: pa.Schema) -> dict[str, Any]:
    raw = (schema.metadata or {}).get(b"quantization")
    if raw is None:
        raise ValueError("Missing parquet footer metadata key: quantization")
    return cast("dict[str, Any]", json.loads(raw.decode("utf-8")))


def _load_original_all(original_root: Path, batch_size: int) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for file_path in _list_parquet_files(original_root):
        pf = pq.ParquetFile(file_path)
        file_chunks = [
            _extract_2d(batch.column(0)).astype(np.float32, copy=False)
            for batch in pf.iter_batches(columns=["embedding"], batch_size=batch_size)
        ]
        chunks.extend(file_chunks)
    if not chunks:
        raise ValueError(f"No parquet embeddings found under {original_root}")
    return np.concatenate(chunks, axis=0)


def _load_reconstructed_all(method_root: Path, batch_size: int) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for file_path in _list_parquet_files(method_root):
        pf = pq.ParquetFile(file_path)
        qmeta = _parse_quant_meta(pf.schema_arrow)
        for batch in pf.iter_batches(columns=["embedding"], batch_size=batch_size):
            q_values = _extract_2d(batch.column(0))
            dq_input: dict[str, Any] = {
                "method": qmeta["method"],
                "quantized": q_values,
            }
            if "scale" in qmeta:
                dq_input["scale"] = np.asarray(qmeta["scale"], dtype=np.float32)
            if "zero_point" in qmeta:
                dq_input["zero_point"] = np.asarray(qmeta["zero_point"], dtype=np.float32)
            if "n_dims" in qmeta:
                dq_input["n_dims"] = int(qmeta["n_dims"])
            chunks.append(dequantize(dq_input).astype(np.float32, copy=False))
    if not chunks:
        raise ValueError(f"No parquet embeddings found under {method_root}")
    return np.concatenate(chunks, axis=0)


def _knn_suite(
    x_orig: np.ndarray,
    x_recon: np.ndarray,
    *,
    ks: tuple[int, ...],
    metrics: tuple[KNNMetric, ...],
) -> dict[str, float]:
    out: dict[str, float] = {}
    for metric in metrics:
        recalls = knn_recall_multi(x_orig, x_recon, ks=ks, metric=metric)
        vals = []
        for k in ks:
            value = float(recalls[k])
            out[f"knn_recall_{k}_{metric}"] = value
            vals.append(value)
        out[f"knn_recall_mean_{metric}"] = float(np.mean(vals))
    return out


def _resolve_methods(quantized_root: Path, methods_arg: str) -> list[str]:
    if methods_arg:
        return [m.strip() for m in methods_arg.split(",") if m.strip()]

    discovered = sorted(p.name for p in quantized_root.iterdir() if p.is_dir())
    preferred = ["binary", "int2", "int3", "int4", "int8", "fp8", "float16"]
    preferred_set = set(preferred)
    ordered = [m for m in preferred if m in discovered]
    ordered.extend(m for m in discovered if m not in preferred_set)
    return ordered


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--original-root", required=True, help="Path to original embeddings parquet root"
    )
    parser.add_argument(
        "--quantized-root", required=True, help="Path to quantized method directories"
    )
    parser.add_argument("--output", "-o", default="results/report_metrics_full.json")
    parser.add_argument(
        "--methods",
        type=str,
        default="",
        help="Comma-separated methods to evaluate; default discovers under quantized root",
    )
    parser.add_argument(
        "--ks",
        type=str,
        default=",".join(str(k) for k in DEFAULT_KS),
        help="Comma-separated k values for kNN recall",
    )
    parser.add_argument(
        "--knn-metrics",
        type=str,
        default=",".join(DEFAULT_KNN_METRICS),
        help="Comma-separated kNN metrics (cosine,euclidean)",
    )
    parser.add_argument("--n-sample-pairs", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    original_root = Path(args.original_root)
    quantized_root = Path(args.quantized_root)

    ks = tuple(sorted({int(x.strip()) for x in args.ks.split(",") if x.strip()}))
    parsed_knn = tuple(x.strip() for x in args.knn_metrics.split(",") if x.strip())

    invalid_metrics = sorted(set(parsed_knn) - set(DEFAULT_KNN_METRICS))
    if invalid_metrics:
        raise ValueError(f"Unknown kNN metrics: {invalid_metrics}. Allowed: {DEFAULT_KNN_METRICS}")
    if not ks:
        raise ValueError("At least one k is required")

    knn_metrics = cast("tuple[KNNMetric, ...]", parsed_knn)
    methods = _resolve_methods(quantized_root, args.methods)

    console.rule("[bold]Full In-Memory Quantization Metrics")
    console.print("Warning: this mode can require very large RAM.")

    console.print("\n[bold]Loading full original embeddings...[/bold]")
    x_orig = _load_original_all(original_root, batch_size=args.batch_size)
    console.print(f"Loaded original: {x_orig.shape[0]} x {x_orig.shape[1]}")

    results: list[dict[str, Any]] = []
    for method in methods:
        method_root = quantized_root / method
        if not method_root.exists():
            raise ValueError(f"Method path does not exist: {method_root}")

        console.print(f"\n[bold]evaluate: {method}[/bold]")
        x_recon = _load_reconstructed_all(method_root, batch_size=args.batch_size)
        if x_recon.shape != x_orig.shape:
            raise ValueError(
                f"Shape mismatch for {method}: original={x_orig.shape}, reconstructed={x_recon.shape}"
            )

        row = {
            "method": method,
            "reconstruction_mse": reconstruction_mse(x_orig, x_recon),
            "reconstruction_cosine": reconstruction_cosine(x_orig, x_recon),
            **_knn_suite(x_orig, x_recon, ks=ks, metrics=knn_metrics),
            "cosine_sim_corr": cosine_similarity_correlation(
                x_orig,
                x_recon,
                n_sample_pairs=args.n_sample_pairs,
                seed=args.seed,
            ),
        }
        results.append(row)
        console.print(
            f"  recon_cos={row['reconstruction_cosine']:.4f}  "
            f"mse={row['reconstruction_mse']:.6f}  "
            f"kNN(cos)@10={row.get('knn_recall_10_cosine', float('nan')):.3f}"
        )

        del x_recon

    payload = {
        "mode": "full_in_memory",
        "notes": "All reported metrics are computed in-memory over full arrays.",
        "original_root": str(original_root),
        "quantized_root": str(quantized_root),
        "methods": methods,
        "ks": list(ks),
        "knn_metrics": list(knn_metrics),
        "n_sample_pairs": int(args.n_sample_pairs),
        "batch_size": int(args.batch_size),
        "results": results,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_serialize(payload), indent=2))
    console.print(f"\n[bold green]Report saved to {out_path}[/bold green]")


if __name__ == "__main__":
    main()
