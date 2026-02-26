"""Intrinsic dimension estimation using scikit-dimension."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import skdim.id

if TYPE_CHECKING:
    from terrabit._typing import NDArrayF32

logging.getLogger(__name__).addHandler(logging.NullHandler())

K_SWEEP_VALUES = (10, 15, 20, 25, 30)
N_STABILITY_RUNS = 4


def _fallback_participation_ratio_id(x: NDArrayF32) -> float:
    """Numerically stable participation-ratio fallback ID estimate."""
    x_centered = x - np.mean(x, axis=0, keepdims=True)
    singular_values = np.linalg.svd(x_centered, compute_uv=False)
    eigenvalues = singular_values**2
    denom = float(np.sum(eigenvalues**2))
    if denom <= 0:
        return 1.0
    return float((np.sum(eigenvalues) ** 2) / denom)


def estimate_id_mle(x: NDArrayF32, n_neighbors: int = 20) -> float:
    """Estimate intrinsic dimension via MLE (Levina-Bickel)."""
    try:
        est = skdim.id.MLE(K=n_neighbors)
        est.fit(x)
        return float(est.dimension_)
    except Exception:
        return _fallback_participation_ratio_id(x)


def estimate_id_twonn(x: NDArrayF32, discard_fraction: float = 0.1) -> float:
    """Estimate intrinsic dimension via TwoNN."""
    try:
        est = skdim.id.TwoNN(discard_fraction=discard_fraction)
        est.fit(x)
        return float(est.dimension_)
    except Exception:
        return _fallback_participation_ratio_id(x)


def estimate_id_lpca(x: NDArrayF32, ver: str = "participation_ratio") -> float:
    """Estimate intrinsic dimension via PCA participation ratio (effective rank)."""
    try:
        est = skdim.id.lPCA(ver=ver)
        est.fit(x)
        return float(est.dimension_)
    except Exception:
        return _fallback_participation_ratio_id(x)


def k_sweep_mle(
    x: NDArrayF32,
    k_values: tuple[int, ...] = K_SWEEP_VALUES,
) -> dict[str, Any]:
    """Sweep k for MLE and return mean, std across k values."""
    results: list[float] = []
    for k in k_values:
        d = estimate_id_mle(x, n_neighbors=k)
        results.append(d)
    arr = np.array(results, dtype=np.float64)
    return {"mean": float(np.mean(arr)), "std": float(np.std(arr)), "per_k": results}


def estimate_intrinsic_dimension(
    x: NDArrayF32,
    *,
    n_stability_runs: int = N_STABILITY_RUNS,
    k_sweep_values: tuple[int, ...] = K_SWEEP_VALUES,
    seed: int | None = None,
) -> dict[str, Any]:
    """Estimate ID with MLE, TwoNN, lPCA; k-sweep and stability runs."""
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    id_mle_values: list[float] = []
    id_twonn_values: list[float] = []
    id_lpca_values: list[float] = []
    k_sweep_results: list[dict[str, Any]] = []

    for _ in range(n_stability_runs):
        if n_stability_runs > 1:
            sub_size = min(n, max(5000, n // 2))
            idx = rng.choice(n, size=sub_size, replace=False)
            x_sub = x[idx]
        else:
            x_sub = x

        id_mle = estimate_id_mle(x_sub)
        id_mle_values.append(id_mle)

        id_twonn = estimate_id_twonn(x_sub)
        id_twonn_values.append(id_twonn)

        id_lpca = estimate_id_lpca(x_sub)
        id_lpca_values.append(id_lpca)

        k_sweep = k_sweep_mle(x_sub, k_values=k_sweep_values)
        k_sweep_results.append(k_sweep)

    mle_arr = np.array(id_mle_values, dtype=np.float64)
    twonn_arr = np.array(id_twonn_values, dtype=np.float64)
    lpca_arr = np.array(id_lpca_values, dtype=np.float64)

    return {
        "id_mle": float(np.mean(mle_arr)),
        "id_mle_std": float(np.std(mle_arr)),
        "id_twonn": float(np.mean(twonn_arr)),
        "id_twonn_std": float(np.std(twonn_arr)),
        "id_lpca": float(np.mean(lpca_arr)),
        "id_lpca_std": float(np.std(lpca_arr)),
        "id_range": {
            "mle": (float(np.min(mle_arr)), float(np.max(mle_arr))),
            "twonn": (float(np.min(twonn_arr)), float(np.max(twonn_arr))),
            "lpca": (float(np.min(lpca_arr)), float(np.max(lpca_arr))),
        },
        "k_sweep_results": k_sweep_results,
    }
