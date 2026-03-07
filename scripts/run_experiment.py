"""Intrinsic dimension & compression experiments.

Phases:
  1. Subsample + characterize embedding distribution
  2. ID estimation (multiple preprocessing x k-sweep x stability)
  3. Compression sweep (PCA, RP, PCA+whitening) x target dims
  4. Post-compression ID analysis
    5. Label-free quality metrics (kNN recall, cosine sim, isotropy, eff. rank)
    6. Quantization baselines (float16, fp8, int8, int4, int3, int2, binary)

Usage:
  uv run python scripts/run_experiment.py embeddings/ -o results/report.json
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
from rich.console import Console
from rich.table import Table
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection

from terrabit.id_estimation import estimate_intrinsic_dimension
from terrabit.io import preprocess, reservoir_subsample
from terrabit.metrics import (
    cosine_similarity_correlation,
    effective_rank,
    explained_variance_ratio,
    isotropy_score,
    knn_recall_multi,
    reconstruction_cosine,
    reconstruction_mse,
)
from terrabit.quantization import dequantize, quantize

console = Console()
log = logging.getLogger(__name__)

SEED = 42
PREPROCESS_CONFIGS: dict[str, dict[str, bool]] = {
    "raw": {"l2_norm": False, "center": False},
    "centered": {"l2_norm": False, "center": True},
    "l2_normed": {"l2_norm": True, "center": False},
    "centered_l2": {"l2_norm": True, "center": True},
}
QUANTIZATION_METHODS = (
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
KNN_RECALL_KS = (10, 25, 50)
KNN_RECALL_METRICS = ("cosine", "euclidean")


# ---------------------------------------------------------------------------
# Phase 1: Characterization
# ---------------------------------------------------------------------------


def characterize(x: np.ndarray) -> dict[str, Any]:
    """Compute distribution stats on raw embeddings."""
    norms = np.linalg.norm(x, axis=1)
    mean_vec = x.mean(axis=0)
    var_per_dim = x.var(axis=0)

    # Anisotropy: average cosine similarity between random pairs
    rng = np.random.default_rng(SEED)
    n_pairs = min(50_000, len(x) * (len(x) - 1) // 2)
    i_idx = rng.integers(0, len(x), size=n_pairs)
    j_idx = rng.integers(0, len(x), size=n_pairs)
    mask = i_idx != j_idx
    i_idx, j_idx = i_idx[mask], j_idx[mask]
    a, b = x[i_idx], x[j_idx]
    dots = np.sum(a * b, axis=1)
    na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    denom = na * nb
    denom = np.where(denom > 0, denom, 1.0)
    cos_sims = dots / denom

    # Explained variance at standard thresholds
    evr = {str(k): explained_variance_ratio(x, k) for k in (32, 64, 128, 256, 512)}

    return {
        "n_samples": int(x.shape[0]),
        "n_dims": int(x.shape[1]),
        "norm_mean": float(norms.mean()),
        "norm_std": float(norms.std()),
        "norm_min": float(norms.min()),
        "norm_max": float(norms.max()),
        "mean_vec_norm": float(np.linalg.norm(mean_vec)),
        "var_per_dim_mean": float(var_per_dim.mean()),
        "var_per_dim_std": float(var_per_dim.std()),
        "anisotropy_cosine_mean": float(cos_sims.mean()),
        "anisotropy_cosine_std": float(cos_sims.std()),
        "effective_rank": effective_rank(x),
        "isotropy": isotropy_score(x),
        "explained_variance_ratio": evr,
    }


# ---------------------------------------------------------------------------
# Phase 2: ID estimation across preprocessing configs
# ---------------------------------------------------------------------------


def estimate_id_sweep(x_raw: np.ndarray) -> dict[str, Any]:
    """Estimate ID under each preprocessing config."""
    results: dict[str, Any] = {}
    for name, cfg in PREPROCESS_CONFIGS.items():
        console.print(f"  [dim]ID estimation: {name}[/dim]")
        x = preprocess(x_raw, **cfg)
        t0 = time.perf_counter()
        report = estimate_intrinsic_dimension(x, seed=SEED)
        elapsed = time.perf_counter() - t0
        report["elapsed_s"] = round(elapsed, 2)
        results[name] = report
        console.print(
            f"    MLE={report['id_mle']:.1f}±{report['id_mle_std']:.1f}  "
            f"TwoNN={report['id_twonn']:.1f}±{report['id_twonn_std']:.1f}  "
            f"lPCA={report['id_lpca']:.1f}±{report['id_lpca_std']:.1f}  "
            f"({elapsed:.1f}s)"
        )
    return results


# ---------------------------------------------------------------------------
# Phase 3-5: Compression sweep
# ---------------------------------------------------------------------------


def _target_dims(estimated_id: float) -> list[int]:
    """Derive target dims from estimated ID."""
    candidates = {
        max(2, int(estimated_id // 2)),
        max(2, int(estimated_id)),
        max(2, int(2 * estimated_id)),
        max(2, int(4 * estimated_id)),
        32,
        64,
        128,
        256,
        512,
    }
    return sorted(d for d in candidates if 2 <= d <= 1024)


def _compress_pca(
    x: np.ndarray, n_components: int, *, whiten: bool = False
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """PCA compress; return (compressed, reconstructed, meta)."""
    pca = PCA(n_components=n_components, whiten=whiten, random_state=SEED)
    x_comp = pca.fit_transform(x).astype(np.float32)
    x_recon = pca.inverse_transform(x_comp).astype(np.float32)
    meta = {
        "variance_retained": float(pca.explained_variance_ratio_.sum()),
    }
    return x_comp, x_recon, meta


def _compress_rp(x: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Random projection; pseudo-inverse reconstruction."""
    rp = GaussianRandomProjection(n_components=n_components, random_state=SEED)
    x_comp = rp.fit_transform(x).astype(np.float32)
    # Reconstruct via pseudo-inverse of projection matrix
    pinv = np.linalg.pinv(rp.components_.T)
    x_recon = (x_comp @ pinv).astype(np.float32)
    return x_comp, x_recon, {}


COMPRESSION_METHODS = {
    "pca": lambda x, d: _compress_pca(x, d, whiten=False),
    "pca_whitening": lambda x, d: _compress_pca(x, d, whiten=True),
    "rp": _compress_rp,
}


def _knn_recall_suite(x_orig: np.ndarray, x_other: np.ndarray) -> dict[str, float]:
    """Compute kNN recalls for multiple k values and distance metrics."""
    out: dict[str, float] = {}
    for metric in KNN_RECALL_METRICS:
        recalls = knn_recall_multi(x_orig, x_other, ks=KNN_RECALL_KS, metric=metric)
        values = []
        for k in KNN_RECALL_KS:
            recall = recalls[k]
            out[f"knn_recall_{k}_{metric}"] = recall
            values.append(recall)
        out[f"knn_recall_mean_{metric}"] = float(np.mean(values))
    return out


def compression_sweep(
    x_orig: np.ndarray,
    target_dims: list[int],
) -> list[dict[str, Any]]:
    """Run all compression methods x dims, compute metrics + post-compression ID."""
    results: list[dict[str, Any]] = []
    total = len(COMPRESSION_METHODS) * len(target_dims)
    i = 0

    for method_name, compress_fn in COMPRESSION_METHODS.items():
        for dim in target_dims:
            i += 1
            console.print(f"  [{i}/{total}] {method_name} d={dim}")
            t0 = time.perf_counter()
            x_comp, x_recon, meta = compress_fn(x_orig, dim)
            compress_time = time.perf_counter() - t0

            # Metrics
            t1 = time.perf_counter()
            knn_metrics = _knn_recall_suite(x_orig, x_comp)
            metrics = {
                **knn_metrics,
                "cosine_sim_corr": cosine_similarity_correlation(x_orig, x_comp, seed=SEED),
                "reconstruction_mse": reconstruction_mse(x_orig, x_recon),
                "reconstruction_cosine": reconstruction_cosine(x_orig, x_recon),
                "effective_rank_comp": effective_rank(x_comp),
                "isotropy_comp": isotropy_score(x_comp),
            }
            metric_time = time.perf_counter() - t1

            # Post-compression ID (single run, no stability sweep for speed)
            t2 = time.perf_counter()
            post_id = estimate_intrinsic_dimension(x_comp, n_stability_runs=1, seed=SEED)
            id_time = time.perf_counter() - t2

            row = {
                "method": method_name,
                "target_dim": dim,
                **meta,
                **metrics,
                "post_id_mle": post_id["id_mle"],
                "post_id_twonn": post_id["id_twonn"],
                "post_id_lpca": post_id["id_lpca"],
                "compress_time_s": round(compress_time, 2),
                "metric_time_s": round(metric_time, 2),
                "id_time_s": round(id_time, 2),
            }
            results.append(row)

            console.print(
                f"    kNN(cos)@10={metrics['knn_recall_10_cosine']:.3f}  "
                f"kNN(euc)@10={metrics['knn_recall_10_euclidean']:.3f}  "
                f"cos_corr={metrics['cosine_sim_corr']:.3f}  "
                f"recon_cos={metrics['reconstruction_cosine']:.3f}  "
                f"post_ID_mle={post_id['id_mle']:.1f}"
            )

    return results


# ---------------------------------------------------------------------------
# Phase 6: Quantization baselines
# ---------------------------------------------------------------------------


def quantization_sweep(x_orig: np.ndarray) -> list[dict[str, Any]]:
    """Quantize original embeddings, measure reconstruction quality."""
    results: list[dict[str, Any]] = []

    for method in QUANTIZATION_METHODS:
        console.print(f"  quantize: {method}")
        t0 = time.perf_counter()
        q = quantize(x_orig, method)
        x_recon = dequantize(q)
        elapsed = time.perf_counter() - t0

        # Compression ratio (bytes)
        orig_bytes = x_orig.nbytes
        code_bytes = q["quantized"].nbytes
        param_bytes = sum(
            int(v.nbytes)
            for k, v in q.items()
            if k not in ("quantized", "method") and isinstance(v, np.ndarray)
        )
        q_bytes = code_bytes + param_bytes

        knn_metrics = _knn_recall_suite(x_orig, x_recon)
        metrics = {
            "method": method,
            "reconstruction_mse": reconstruction_mse(x_orig, x_recon),
            "reconstruction_cosine": reconstruction_cosine(x_orig, x_recon),
            **knn_metrics,
            "cosine_sim_corr": cosine_similarity_correlation(x_orig, x_recon, seed=SEED),
            "compression_ratio_code_only": round(orig_bytes / code_bytes, 2),
            "compression_ratio_total": round(orig_bytes / q_bytes, 2),
            "orig_bytes": orig_bytes,
            "quantized_bytes": code_bytes,
            "param_bytes": param_bytes,
            "total_bytes": q_bytes,
            "elapsed_s": round(elapsed, 2),
        }
        results.append(metrics)
        console.print(
            f"    recon_cos={metrics['reconstruction_cosine']:.4f}  "
            f"kNN(cos)@10={metrics['knn_recall_10_cosine']:.3f}  "
            f"kNN(euc)@10={metrics['knn_recall_10_euclidean']:.3f}  "
            f"ratio(code)={metrics['compression_ratio_code_only']:.1f}x"
        )

    return results


# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------


def print_id_table(id_results: dict[str, Any]) -> None:
    """Print ID estimation summary as rich table."""
    table = Table(title="Intrinsic Dimension Estimates")
    table.add_column("Preprocessing")
    table.add_column("MLE", justify="right")
    table.add_column("TwoNN", justify="right")
    table.add_column("lPCA", justify="right")
    for name, r in id_results.items():
        table.add_row(
            name,
            f"{r['id_mle']:.1f} ± {r['id_mle_std']:.1f}",
            f"{r['id_twonn']:.1f} ± {r['id_twonn_std']:.1f}",
            f"{r['id_lpca']:.1f} ± {r['id_lpca_std']:.1f}",
        )
    console.print(table)


def print_compression_table(results: list[dict[str, Any]]) -> None:
    """Print compression sweep as rich table."""
    table = Table(title="Compression Sweep Results")
    table.add_column("Method")
    table.add_column("Dim", justify="right")
    table.add_column("kNN C@10", justify="right")
    table.add_column("kNN E@10", justify="right")
    table.add_column("kNN Cμ", justify="right")
    table.add_column("kNN Eμ", justify="right")
    table.add_column("Cos Corr", justify="right")
    table.add_column("Recon Cos", justify="right")
    table.add_column("Post ID (MLE)", justify="right")
    table.add_column("Var Ret", justify="right")
    for r in results:
        var_ret = f"{r.get('variance_retained', 0):.3f}" if "variance_retained" in r else "—"
        table.add_row(
            r["method"],
            str(r["target_dim"]),
            f"{r['knn_recall_10_cosine']:.3f}",
            f"{r['knn_recall_10_euclidean']:.3f}",
            f"{r['knn_recall_mean_cosine']:.3f}",
            f"{r['knn_recall_mean_euclidean']:.3f}",
            f"{r['cosine_sim_corr']:.3f}",
            f"{r['reconstruction_cosine']:.3f}",
            f"{r['post_id_mle']:.1f}",
            var_ret,
        )
    console.print(table)


def print_quantization_table(results: list[dict[str, Any]]) -> None:
    """Print quantization baselines as rich table."""
    table = Table(title="Quantization Baselines")
    table.add_column("Method")
    table.add_column("Recon Cos", justify="right")
    table.add_column("kNN C@10", justify="right")
    table.add_column("kNN E@10", justify="right")
    table.add_column("kNN Cμ", justify="right")
    table.add_column("kNN Eμ", justify="right")
    table.add_column("Cos Corr", justify="right")
    table.add_column("Ratio(code)", justify="right")
    table.add_column("Ratio(total)", justify="right")
    for r in results:
        table.add_row(
            r["method"],
            f"{r['reconstruction_cosine']:.4f}",
            f"{r['knn_recall_10_cosine']:.3f}",
            f"{r['knn_recall_10_euclidean']:.3f}",
            f"{r['knn_recall_mean_cosine']:.3f}",
            f"{r['knn_recall_mean_euclidean']:.3f}",
            f"{r['cosine_sim_corr']:.3f}",
            f"{r['compression_ratio_code_only']:.1f}x",
            f"{r['compression_ratio_total']:.1f}x",
        )
    console.print(table)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


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
        return [_serialize(x) for x in obj]
    return obj


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    global SEED  # simple config; no threading
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Path to parquet embeddings directory")
    parser.add_argument("--output", "-o", default="results/report.json")
    parser.add_argument("--n-subsample", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--skip-compression",
        action="store_true",
        help="Only run characterization + ID estimation",
    )
    parser.add_argument(
        "--skip-quantization",
        action="store_true",
        help="Skip quantization baselines",
    )
    parser.add_argument(
        "--extra-dims",
        type=str,
        default="",
        help="Extra target dims, comma-separated (appended to auto dims)",
    )
    args = parser.parse_args()
    SEED = args.seed

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    # -- Subsample --------------------------------------------------------
    console.rule("[bold]Phase 1: Subsample & Characterize")
    t0 = time.perf_counter()
    x_raw = reservoir_subsample(
        args.path,
        args.n_subsample,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    console.print(
        f"Subsampled {x_raw.shape[0]} embeddings ({x_raw.shape[1]}D) "
        f"in {time.perf_counter() - t0:.1f}s"
    )

    char = characterize(x_raw)
    console.print(
        f"  norm: {char['norm_mean']:.2f} ± {char['norm_std']:.2f}  "
        f"anisotropy: {char['anisotropy_cosine_mean']:.4f}  "
        f"eff_rank: {char['effective_rank']:.1f}  "
        f"isotropy: {char['isotropy']:.6f}"
    )
    console.print(f"  explained var ratios: {char['explained_variance_ratio']}")

    # -- ID estimation ----------------------------------------------------
    console.rule("[bold]Phase 2: Intrinsic Dimension Estimation")
    id_results = estimate_id_sweep(x_raw)
    print_id_table(id_results)

    # Use centered MLE as primary ID estimate
    primary_id = id_results["centered"]["id_mle"]
    console.print(f"\n[bold]Primary ID estimate (centered MLE): {primary_id:.1f}[/bold]")

    report: dict[str, Any] = {
        "seed": args.seed,
        "n_subsample": x_raw.shape[0],
        "embedding_dim": x_raw.shape[1],
        "characterization": char,
        "id_estimation": id_results,
        "primary_id_estimate": primary_id,
    }

    if not args.skip_compression:
        # -- Compression sweep --------------------------------------------
        console.rule("[bold]Phase 3-5: Compression Sweep")
        dims = _target_dims(primary_id)
        if args.extra_dims:
            extra = [int(d) for d in args.extra_dims.split(",") if d.strip()]
            dims = sorted(set(dims) | set(extra))
        console.print(f"Target dims: {dims}")

        # Preprocess (center) for compression
        x_centered = preprocess(x_raw, center=True)
        comp_results = compression_sweep(x_centered, dims)
        print_compression_table(comp_results)

        report["target_dims"] = dims
        report["compression_sweep"] = comp_results

    if not args.skip_quantization:
        # -- Quantization baselines ----------------------------------------
        console.rule("[bold]Phase 6: Quantization Baselines")
        quant_results = quantization_sweep(x_raw)
        print_quantization_table(quant_results)
        report["quantization"] = quant_results

    # -- Save report -------------------------------------------------------
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(_serialize(report), f, indent=2)
    console.print(f"\n[bold green]Report saved to {out_path}[/bold green]")


if __name__ == "__main__":
    main()
