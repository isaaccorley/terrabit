"""Run binary-focused quantization baselines on a fast subsample.

Usage:
  uv run python scripts/run_binary_baselines.py -o results/report_binary_baselines.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
from load_data import load_source_embeddings
from rich.console import Console

from terrabit.metrics import cosine_similarity_correlation, knn_recall_multi
from terrabit.quantization import dequantize, quantize

console = Console()
DEFAULT_METHODS = (
    "binary",
    "binary_med",
    "binary_zscore",
    "binary_itq",
    "int2",
    "int3",
    "int4",
)
DEFAULT_KS = (10, 25, 50)


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


def _metric_row(
    x: np.ndarray, x_recon: np.ndarray, *, method: str, elapsed_s: float
) -> dict[str, Any]:
    recalls = knn_recall_multi(x, x_recon, ks=DEFAULT_KS, metric="cosine")
    q10 = float(recalls[10])
    q25 = float(recalls[25])
    q50 = float(recalls[50])
    corr = float(cosine_similarity_correlation(x, x_recon, seed=42))
    dots = np.sum(x * x_recon, axis=1)
    denom = np.linalg.norm(x, axis=1) * np.linalg.norm(x_recon, axis=1)
    denom = np.where(denom > 0, denom, 1.0)
    recon_cos = float(np.mean(dots / denom))
    return {
        "method": method,
        "reconstruction_cosine": recon_cos,
        "knn_recall_10_cosine": q10,
        "knn_recall_25_cosine": q25,
        "knn_recall_50_cosine": q50,
        "knn_recall_mean_cosine": float(np.mean([q10, q25, q50])),
        "cosine_sim_corr": corr,
        "elapsed_s": round(elapsed_s, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", "-o", default="results/report_binary_baselines.json")
    parser.add_argument("--n-samples", type=int, default=20_000)
    parser.add_argument(
        "--methods",
        type=str,
        default=",".join(DEFAULT_METHODS),
        help="Comma-separated methods",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    methods = tuple(m.strip() for m in args.methods.split(",") if m.strip())

    console.rule("[bold]Binary Baselines")
    console.print(f"sampling n={args.n_samples} from embeddings/")
    t0 = time.perf_counter()
    x, _files_used, _idx = load_source_embeddings(n_samples=args.n_samples, seed=args.seed)
    console.print(f"loaded {x.shape[0]} x {x.shape[1]} in {time.perf_counter() - t0:.1f}s")

    results: list[dict[str, Any]] = []
    for method in methods:
        console.print(f"  evaluate: {method}")
        t1 = time.perf_counter()
        q = quantize(x, method)  # type: ignore[arg-type]
        x_recon = dequantize(q)
        row = _metric_row(x, x_recon, method=method, elapsed_s=time.perf_counter() - t1)

        code_bytes = int(q["quantized"].nbytes)
        param_bytes = sum(
            int(v.nbytes)
            for k, v in q.items()
            if k not in ("quantized", "method") and isinstance(v, np.ndarray)
        )
        orig_bytes = int(x.nbytes)
        row["compression_ratio_code_only"] = round(orig_bytes / code_bytes, 2)
        row["compression_ratio_total"] = round(orig_bytes / (code_bytes + param_bytes), 2)
        row["quantized_bytes"] = code_bytes
        row["param_bytes"] = param_bytes
        results.append(row)
        console.print(
            f"    C@10={row['knn_recall_10_cosine']:.3f}  recon_cos={row['reconstruction_cosine']:.3f}  "
            f"ratio(code)={row['compression_ratio_code_only']:.1f}x"
        )

    payload = {
        "n_samples": int(x.shape[0]),
        "n_dims": int(x.shape[1]),
        "methods": list(methods),
        "results": results,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_serialize(payload), indent=2))
    console.print(f"\n[bold green]Report saved to {out_path}[/bold green]")


if __name__ == "__main__":
    main()
