"""Run quantization metrics with batched streaming to avoid OOM.

This script computes exact full-dataset reconstruction metrics (MSE and cosine)
via streaming and computes kNN/cosine-correlation metrics on a fixed-size
reservoir sample.

It expects:
- original embeddings under --original-root
- quantized outputs under --quantized-root/<method>/...
  with per-file parquet footer metadata key "quantization".

Usage:
  uv run python scripts/run_metrics_batched.py \
    --original-root embeddings \
    --quantized-root results/quantized_dataset_full_allmethods_j8 \
    -o results/report_metrics_batched.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from rich.console import Console

from terrabit.metrics import cosine_similarity_correlation, knn_recall_multi
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


def _resolve_methods(quantized_root: Path, methods_arg: str) -> list[str]:
    if methods_arg:
        return [m.strip() for m in methods_arg.split(",") if m.strip()]

    discovered = sorted(p.name for p in quantized_root.iterdir() if p.is_dir())
    preferred = [
        "binary",
        "binary_med",
        "binary_zscore",
        "binary_itq",
        "int2",
        "int3",
        "int4",
        "int8",
        "fp8",
        "float16",
    ]
    preferred_set = set(preferred)
    ordered = [m for m in preferred if m in discovered]
    ordered.extend(m for m in discovered if m not in preferred_set)
    return ordered


def _resolve_jobs(*, requested_jobs: int, n_files: int) -> int:
    if n_files <= 0:
        return 1
    if requested_jobs > 0:
        return max(1, min(requested_jobs, n_files))
    cpu_count = os.cpu_count() or 1
    return max(1, min(cpu_count // 2, n_files))


def _worker_init() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")


def _file_pairs(
    *,
    original_root: Path,
    method_root: Path,
    limit_files: int,
) -> list[tuple[Path, Path]]:
    original_files = _list_parquet_files(original_root)
    quant_files = _list_parquet_files(method_root)

    rel_orig = {p.relative_to(original_root): p for p in original_files}
    rel_quant = {p.relative_to(method_root): p for p in quant_files}

    if set(rel_orig.keys()) != set(rel_quant.keys()):
        raise ValueError("Path mismatch between original and quantized files")

    rel_paths = sorted(rel_orig.keys())
    if limit_files > 0:
        rel_paths = rel_paths[:limit_files]
    return [(rel_orig[r], rel_quant[r]) for r in rel_paths]


def _dequantize_with_meta(
    *,
    q_values: np.ndarray,
    qmeta: dict[str, Any],
) -> np.ndarray:
    dq_input: dict[str, Any] = {"method": qmeta["method"], "quantized": q_values}
    for k, v in qmeta.items():
        if k in ("method", "embedding_col"):
            continue
        if isinstance(v, list):
            dq_input[k] = np.asarray(v, dtype=np.float32)
        else:
            dq_input[k] = v
    return dequantize(dq_input)


def _exact_metrics_for_file(
    *,
    orig_file: str,
    quant_file: str,
    batch_size: int,
) -> dict[str, float | int]:
    orig_pf = pq.ParquetFile(orig_file)
    quant_pf = pq.ParquetFile(quant_file)
    qmeta = _parse_quant_meta(quant_pf.schema_arrow)

    sse = 0.0
    cos_sum = 0.0
    n_rows = 0
    n_values = 0

    orig_batches = orig_pf.iter_batches(columns=["embedding"], batch_size=batch_size)
    quant_batches = quant_pf.iter_batches(columns=["embedding"], batch_size=batch_size)

    for orig_batch, quant_batch in zip(orig_batches, quant_batches, strict=True):
        x_orig = _extract_2d(orig_batch.column(0)).astype(np.float32, copy=False)
        q_values = _extract_2d(quant_batch.column(0))
        x_recon = _dequantize_with_meta(q_values=q_values, qmeta=qmeta)

        diff = x_orig - x_recon
        sse += float(np.sum(diff * diff, dtype=np.float64))
        n_values += int(diff.size)

        dots = np.sum(x_orig * x_recon, axis=1)
        norm_orig = np.linalg.norm(x_orig, axis=1)
        norm_recon = np.linalg.norm(x_recon, axis=1)
        denom = np.where(norm_orig * norm_recon > 0, norm_orig * norm_recon, 1.0)
        cos_sum += float(np.sum(dots / denom, dtype=np.float64))
        n_rows += int(x_orig.shape[0])

    return {
        "sse": sse,
        "cos_sum": cos_sum,
        "n_rows": n_rows,
        "n_values": n_values,
    }


def _reservoir_update_pair(
    *,
    rng: np.random.Generator,
    sample_a: np.ndarray,
    sample_b: np.ndarray,
    batch_a: np.ndarray,
    batch_b: np.ndarray,
    n_seen: int,
    reservoir_size: int,
) -> int:
    """Update two aligned reservoirs with the same slot decisions. Returns updated n_seen."""
    batch_n = int(batch_a.shape[0])

    # Phase 1: fill empty slots directly (no RNG needed)
    if n_seen < reservoir_size:
        fill_end = min(n_seen + batch_n, reservoir_size)
        fill_n = fill_end - n_seen
        sample_a[n_seen:fill_end] = batch_a[:fill_n]
        sample_b[n_seen:fill_end] = batch_b[:fill_n]
        n_seen += fill_n
        batch_a = batch_a[fill_n:]
        batch_b = batch_b[fill_n:]
        batch_n -= fill_n

    # Phase 2: probabilistic replacement for rows beyond reservoir_size
    if batch_n > 0:
        row_indices = np.arange(n_seen, n_seen + batch_n, dtype=np.int64)
        slots = rng.integers(0, row_indices + 1, dtype=np.int64)
        keep_mask = slots < reservoir_size
        keep_slots = slots[keep_mask]
        sample_a[keep_slots] = batch_a[keep_mask]
        sample_b[keep_slots] = batch_b[keep_mask]
        n_seen += batch_n

    return n_seen


def _sample_reservoir(
    *,
    pairs: list[tuple[Path, Path]],
    reservoir_size: int,
    batch_size: int,
    seed: int,
    progress_every: int,
    method: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Reservoir-sample orig and quantized rows, dequantizing only the reservoir at the end.

    Dequantization is deferred to a single pass over the reservoir (reservoir_size rows)
    rather than over the full dataset (~50M rows), which is a ~2500x reduction in dequant work.
    """
    rng = np.random.default_rng(seed)
    sample_orig: np.ndarray | None = None
    sample_quant: np.ndarray | None = None
    qmeta_saved: dict[str, Any] | None = None
    n_seen = 0

    total_files = len(pairs)
    for idx, (orig_file, quant_file) in enumerate(pairs, start=1):
        orig_pf = pq.ParquetFile(orig_file)
        quant_pf = pq.ParquetFile(quant_file)
        qmeta = _parse_quant_meta(quant_pf.schema_arrow)
        if qmeta_saved is None:
            qmeta_saved = qmeta

        orig_batches = orig_pf.iter_batches(columns=["embedding"], batch_size=batch_size)
        quant_batches = quant_pf.iter_batches(columns=["embedding"], batch_size=batch_size)
        for orig_batch, quant_batch in zip(orig_batches, quant_batches, strict=True):
            x_orig = _extract_2d(orig_batch.column(0)).astype(np.float32, copy=False)
            q_values = _extract_2d(quant_batch.column(0))

            if sample_orig is None:
                orig_dim = int(x_orig.shape[1])
                quant_dim = int(q_values.shape[1])
                sample_orig = np.empty((reservoir_size, orig_dim), dtype=np.float32)
                sample_quant = np.empty((reservoir_size, quant_dim), dtype=q_values.dtype)
            assert sample_quant is not None

            n_seen = _reservoir_update_pair(
                rng=rng,
                sample_a=sample_orig,
                sample_b=sample_quant,
                batch_a=x_orig,
                batch_b=q_values,
                n_seen=n_seen,
                reservoir_size=reservoir_size,
            )

        if idx % progress_every == 0 or idx == total_files:
            console.print(f"  {method}: sampling {idx}/{total_files} files  seen_rows={n_seen}")

    if sample_orig is None or sample_quant is None or qmeta_saved is None:
        raise ValueError(f"No data found for method {method}")

    sample_n = min(reservoir_size, n_seen)
    # Dequantize only the reservoir rows — O(reservoir_size) not O(N)
    x_sample_recon = _dequantize_with_meta(q_values=sample_quant[:sample_n], qmeta=qmeta_saved)
    return sample_orig[:sample_n], x_sample_recon, sample_n


def _evaluate_method_streaming(
    *,
    method: str,
    original_root: Path,
    method_root: Path,
    ks: tuple[int, ...],
    knn_metrics: tuple[KNNMetric, ...],
    reservoir_size: int,
    n_sample_pairs: int,
    seed: int,
    batch_size: int,
    jobs: int,
    progress_every: int,
    limit_files: int,
) -> dict[str, Any]:
    pairs = _file_pairs(
        original_root=original_root, method_root=method_root, limit_files=limit_files
    )
    worker_count = _resolve_jobs(requested_jobs=jobs, n_files=len(pairs))
    console.print(f"  {method}: files={len(pairs)} batch_size={batch_size} workers={worker_count}")

    sse = 0.0
    cos_sum = 0.0
    n_rows = 0
    n_values = 0

    exact_t0 = time.perf_counter()
    if worker_count == 1:
        total_files = len(pairs)
        for idx, (orig_file, quant_file) in enumerate(pairs, start=1):
            partial = _exact_metrics_for_file(
                orig_file=str(orig_file),
                quant_file=str(quant_file),
                batch_size=batch_size,
            )
            sse += float(partial["sse"])
            cos_sum += float(partial["cos_sum"])
            n_rows += int(partial["n_rows"])
            n_values += int(partial["n_values"])
            if idx % progress_every == 0 or idx == total_files:
                console.print(f"  {method}: exact {idx}/{total_files} files")
    else:
        completed = 0
        total_files = len(pairs)
        with ProcessPoolExecutor(max_workers=worker_count, initializer=_worker_init) as ex:
            futures = [
                ex.submit(
                    _exact_metrics_for_file,
                    orig_file=str(orig_file),
                    quant_file=str(quant_file),
                    batch_size=batch_size,
                )
                for orig_file, quant_file in pairs
            ]
            for future in as_completed(futures):
                partial = future.result()
                sse += float(partial["sse"])
                cos_sum += float(partial["cos_sum"])
                n_rows += int(partial["n_rows"])
                n_values += int(partial["n_values"])
                completed += 1
                if completed % progress_every == 0 or completed == total_files:
                    console.print(f"  {method}: exact {completed}/{total_files} files")

    exact_elapsed = time.perf_counter() - exact_t0

    sample_t0 = time.perf_counter()
    x_sample_orig, x_sample_recon, sample_n = _sample_reservoir(
        pairs=pairs,
        reservoir_size=reservoir_size,
        batch_size=batch_size,
        seed=seed,
        progress_every=progress_every,
        method=method,
    )
    sample_elapsed = time.perf_counter() - sample_t0

    row: dict[str, Any] = {
        "method": method,
        "n_rows": n_rows,
        "n_values": n_values,
        "reconstruction_mse_full": float(sse / n_values),
        "reconstruction_cosine_full": float(cos_sum / n_rows),
        "sample_size": sample_n,
        "workers": worker_count,
        "exact_pass_elapsed_s": round(exact_elapsed, 2),
        "sample_pass_elapsed_s": round(sample_elapsed, 2),
    }

    for metric in knn_metrics:
        recalls = knn_recall_multi(x_sample_orig, x_sample_recon, ks=ks, metric=metric)
        vals = []
        for k in ks:
            value = float(recalls[k])
            row[f"knn_recall_{k}_{metric}_sampled"] = value
            vals.append(value)
        row[f"knn_recall_mean_{metric}_sampled"] = float(np.mean(vals))

    row["cosine_sim_corr_sampled"] = float(
        cosine_similarity_correlation(
            x_sample_orig,
            x_sample_recon,
            n_sample_pairs=n_sample_pairs,
            seed=seed,
        )
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--original-root", required=True, help="Path to original embeddings parquet root"
    )
    parser.add_argument(
        "--quantized-root", required=True, help="Path to quantized method directories"
    )
    parser.add_argument("--output", "-o", default="results/report_metrics_batched.json")
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
    parser.add_argument("--reservoir-size", type=int, default=20_000)
    parser.add_argument("--n-sample-pairs", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument(
        "--jobs", type=int, default=0, help="Worker processes for exact pass (0=auto)"
    )
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument(
        "--limit-files", type=int, default=0, help="If >0, evaluate first N files only"
    )
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

    console.rule("[bold]Batched Quantization Metrics")
    console.print(f"original root: {original_root}")
    console.print(f"quantized root: {quantized_root}")
    console.print(f"methods: {methods}")

    results: list[dict[str, Any]] = []
    for method in methods:
        method_root = quantized_root / method
        if not method_root.exists():
            raise ValueError(f"Method path does not exist: {method_root}")
        console.print(f"\n[bold]evaluate: {method}[/bold]")
        row = _evaluate_method_streaming(
            method=method,
            original_root=original_root,
            method_root=method_root,
            ks=ks,
            knn_metrics=knn_metrics,
            reservoir_size=args.reservoir_size,
            n_sample_pairs=args.n_sample_pairs,
            seed=args.seed,
            batch_size=args.batch_size,
            jobs=args.jobs,
            progress_every=max(1, args.progress_every),
            limit_files=max(0, args.limit_files),
        )
        results.append(row)
        console.print(
            f"  recon_cos={row['reconstruction_cosine_full']:.4f}  "
            f"mse={row['reconstruction_mse_full']:.6f}  "
            f"kNN(cos)@10={row.get('knn_recall_10_cosine_sampled', float('nan')):.3f}"
        )

    payload = {
        "mode": "batched_streaming",
        "notes": "Reconstruction metrics are exact over full dataset; kNN and cosine correlation are sampled via reservoir.",
        "original_root": str(original_root),
        "quantized_root": str(quantized_root),
        "methods": methods,
        "ks": list(ks),
        "knn_metrics": list(knn_metrics),
        "reservoir_size": int(args.reservoir_size),
        "n_sample_pairs": int(args.n_sample_pairs),
        "batch_size": int(args.batch_size),
        "jobs": int(args.jobs),
        "progress_every": int(args.progress_every),
        "limit_files": int(args.limit_files),
        "results": results,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_serialize(payload), indent=2))
    console.print(f"\n[bold green]Report saved to {out_path}[/bold green]")


if __name__ == "__main__":
    main()
