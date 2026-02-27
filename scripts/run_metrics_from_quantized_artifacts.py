"""Run quantization metrics from pre-materialized parquet artifacts.

Usage:
  uv run python scripts/run_metrics_from_quantized_artifacts.py \
    results/artifacts_quant_1k -o results/report_quant_1k_from_artifacts.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from rich.console import Console

from terrabit.io import iter_embedding_batches
from terrabit.metrics import (
    cosine_similarity_correlation,
    knn_recall_multi,
    reconstruction_cosine,
    reconstruction_mse,
)

console = Console()
KNNMetric = Literal["cosine", "euclidean"]
DEFAULT_KS = (10, 25, 50)
DEFAULT_KNN_METRICS: tuple[KNNMetric, ...] = ("cosine", "euclidean")


def _load_embeddings(path: str) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for x_batch, _ in iter_embedding_batches(path):
        chunks.append(np.asarray(x_batch, dtype=np.float32))
    if not chunks:
        msg = f"No embeddings found under: {path}"
        raise ValueError(msg)
    return np.concatenate(chunks, axis=0)


def _knn_recall_suite(
    x_orig: np.ndarray,
    x_other: np.ndarray,
    *,
    ks: tuple[int, ...],
    metrics: tuple[KNNMetric, ...],
) -> dict[str, float]:
    out: dict[str, float] = {}
    for metric in metrics:
        recalls = knn_recall_multi(x_orig, x_other, ks=ks, metric=metric)
        values = []
        for k in ks:
            value = float(recalls[k])
            out[f"knn_recall_{k}_{metric}"] = value
            values.append(value)
        out[f"knn_recall_mean_{metric}"] = float(np.mean(values))
    return out


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_root", help="Root produced by quantize.py")
    parser.add_argument("--output", "-o", default="results/report_quant_from_artifacts.json")
    parser.add_argument(
        "--methods",
        type=str,
        default="",
        help="Comma-separated subset of methods to evaluate; default is all discovered methods",
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-sample-pairs", type=int, default=50_000)
    args = parser.parse_args()

    artifact_root = Path(args.artifact_root)
    original_dir = artifact_root / "original"
    quantized_root = artifact_root / "quantized"

    ks = tuple(sorted({int(x.strip()) for x in args.ks.split(",") if x.strip()}))
    parsed_knn_metrics = tuple(x.strip() for x in args.knn_metrics.split(",") if x.strip())
    invalid_metrics = sorted(set(parsed_knn_metrics) - set(DEFAULT_KNN_METRICS))
    if invalid_metrics:
        raise ValueError(f"Unknown kNN metrics: {invalid_metrics}. Allowed: {DEFAULT_KNN_METRICS}")
    knn_metrics = cast("tuple[KNNMetric, ...]", parsed_knn_metrics)
    if not ks:
        raise ValueError("At least one k is required")
    if not knn_metrics:
        raise ValueError("At least one kNN metric is required")

    console.rule("[bold]Stage 2: Load Original")
    x_orig = _load_embeddings(str(original_dir))
    console.print(f"Loaded original embeddings: {x_orig.shape[0]} x {x_orig.shape[1]}")

    if args.methods:
        methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    else:
        methods = sorted(p.name for p in quantized_root.iterdir() if p.is_dir())

    results: list[dict[str, Any]] = []
    for method in methods:
        method_dir = quantized_root / method
        if not method_dir.exists():
            raise ValueError(f"Method directory does not exist: {method_dir}")

        console.print(f"\n[bold]evaluate: {method}[/bold]")
        x_recon = _load_embeddings(str(method_dir))
        if x_recon.shape != x_orig.shape:
            raise ValueError(
                f"Shape mismatch for {method}: original={x_orig.shape}, method={x_recon.shape}"
            )

        knn_metrics_out = _knn_recall_suite(
            x_orig,
            x_recon,
            ks=ks,
            metrics=knn_metrics,
        )
        row = {
            "method": method,
            "reconstruction_mse": reconstruction_mse(x_orig, x_recon),
            "reconstruction_cosine": reconstruction_cosine(x_orig, x_recon),
            **knn_metrics_out,
            "cosine_sim_corr": cosine_similarity_correlation(
                x_orig,
                x_recon,
                n_sample_pairs=args.n_sample_pairs,
                seed=args.seed,
            ),
        }
        results.append(row)

        del x_recon

    payload = {
        "artifact_root": str(artifact_root),
        "n_samples": int(x_orig.shape[0]),
        "n_dims": int(x_orig.shape[1]),
        "ks": list(ks),
        "knn_metrics": list(knn_metrics),
        "results": results,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_serialize(payload), indent=2))
    console.print(f"\n[bold green]Report saved to {out_path}[/bold green]")


if __name__ == "__main__":
    main()
