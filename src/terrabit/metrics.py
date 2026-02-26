"""Label-free quality metrics for embedding compression.

All metrics operate on numpy arrays and require no task labels.
Designed to measure structural preservation under compression.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from sklearn.neighbors import NearestNeighbors

if TYPE_CHECKING:
    from terrabit._typing import NDArrayF32


def knn_recall(
    x_orig: NDArrayF32,
    x_comp: NDArrayF32,
    k: int = 10,
    metric: Literal["cosine", "euclidean"] = "euclidean",
) -> float:
    """Fraction of k-nearest neighbors preserved after compression.

    For each point, computes overlap between its k-NN set in original space
    vs compressed space. Returns mean recall across all points.
    """
    return knn_recall_multi(x_orig, x_comp, ks=(k,), metric=metric)[k]


def knn_recall_multi(
    x_orig: NDArrayF32,
    x_comp: NDArrayF32,
    ks: tuple[int, ...],
    metric: Literal["cosine", "euclidean"] = "euclidean",
) -> dict[int, float]:
    """Compute kNN recall for multiple k values with shared neighbor search."""
    if not ks:
        return {}

    max_k = max(ks)
    nn_orig = NearestNeighbors(n_neighbors=max_k + 1, algorithm="auto", metric=metric).fit(x_orig)
    nn_comp = NearestNeighbors(n_neighbors=max_k + 1, algorithm="auto", metric=metric).fit(x_comp)
    idx_orig_full = nn_orig.kneighbors(x_orig, return_distance=False)[:, 1:]
    idx_comp_full = nn_comp.kneighbors(x_comp, return_distance=False)[:, 1:]

    out: dict[int, float] = {}
    for k in sorted(set(ks)):
        idx_orig = idx_orig_full[:, :k]
        idx_comp = idx_comp_full[:, :k]
        recalls = []
        for i in range(len(x_orig)):
            overlap = len(set(idx_orig[i]) & set(idx_comp[i]))
            recalls.append(overlap / k)
        out[k] = float(np.mean(recalls))
    return out


def _cosine_sim_matrix(x: NDArrayF32) -> NDArrayF32:
    """Upper-triangle pairwise cosine similarities."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    x_normed = x / norms
    sim = x_normed @ x_normed.T
    return sim


def cosine_similarity_correlation(
    x_orig: NDArrayF32,
    x_comp: NDArrayF32,
    n_sample_pairs: int = 50_000,
    seed: int | None = None,
) -> float:
    """Spearman correlation of pairwise cosine similarities before/after compression.

    Subsamples pairs for scalability.
    """
    from scipy.stats import spearmanr

    n = len(x_orig)
    rng = np.random.default_rng(seed)
    n_pairs = min(n_sample_pairs, n * (n - 1) // 2)
    i_idx = rng.integers(0, n, size=n_pairs)
    j_idx = rng.integers(0, n, size=n_pairs)
    mask = i_idx != j_idx
    i_idx, j_idx = i_idx[mask], j_idx[mask]

    def _pairwise_cosine_chunked(
        x: NDArrayF32,
        ii: NDArrayF32,
        jj: NDArrayF32,
        *,
        chunk_size: int = 8192,
    ) -> NDArrayF32:
        out = np.empty(ii.shape[0], dtype=np.float32)
        for start in range(0, ii.shape[0], chunk_size):
            end = min(start + chunk_size, ii.shape[0])
            a = x[ii[start:end]]
            b = x[jj[start:end]]
            dot = np.sum(a * b, axis=1)
            na = np.linalg.norm(a, axis=1)
            nb = np.linalg.norm(b, axis=1)
            denom = na * nb
            denom = np.where(denom > 0, denom, 1.0)
            out[start:end] = dot / denom
        return out

    sim_orig = _pairwise_cosine_chunked(x_orig, i_idx, j_idx)
    sim_comp = _pairwise_cosine_chunked(x_comp, i_idx, j_idx)
    corr, _ = spearmanr(sim_orig, sim_comp)
    return float(corr)


def reconstruction_mse(x_orig: NDArrayF32, x_recon: NDArrayF32) -> float:
    """Mean squared error of reconstruction."""
    return float(np.mean((x_orig - x_recon) ** 2))


def reconstruction_cosine(x_orig: NDArrayF32, x_recon: NDArrayF32) -> float:
    """Mean cosine similarity between original and reconstructed vectors."""
    dot = np.sum(x_orig * x_recon, axis=1)
    n_orig = np.linalg.norm(x_orig, axis=1)
    n_recon = np.linalg.norm(x_recon, axis=1)
    denom = n_orig * n_recon
    denom = np.where(denom > 0, denom, 1.0)
    return float(np.mean(dot / denom))


def isotropy_score(x: NDArrayF32) -> float:
    """Measure embedding isotropy via partition function ratio.

    Higher = more isotropic (uniformly distributed directions).
    Returns ratio of min/max eigenvalue of the covariance matrix.
    """
    centered = x - x.mean(axis=0)
    cov = centered.T @ centered / len(x)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues[eigenvalues > 0]
    if len(eigenvalues) == 0:
        return 0.0
    return float(eigenvalues.min() / eigenvalues.max())


def effective_rank(x: NDArrayF32) -> float:
    """Effective rank via Shannon entropy of normalized singular values.

    Higher = more dimensions actively used.
    """
    centered = x - x.mean(axis=0)
    s = np.linalg.svd(centered, compute_uv=False)
    s = s[s > 0]
    p = s / s.sum()
    entropy = -np.sum(p * np.log(p))
    return float(np.exp(entropy))


def explained_variance_ratio(x: NDArrayF32, n_components: int) -> float:
    """Fraction of total variance captured by top n_components principal components."""
    centered = x - x.mean(axis=0)
    s = np.linalg.svd(centered, compute_uv=False)
    var = s**2
    return float(var[:n_components].sum() / var.sum())


def evaluate_compression(
    x_orig: NDArrayF32,
    x_comp: NDArrayF32,
    x_recon: NDArrayF32 | None = None,
    *,
    k: int = 10,
    knn_metric: Literal["cosine", "euclidean"] = "euclidean",
    n_sample_pairs: int = 50_000,
    seed: int | None = None,
) -> dict[str, Any]:
    """Run all label-free quality metrics on a compression result.

    Args:
        x_orig: original embeddings (n, d_orig)
        x_comp: compressed embeddings (n, d_comp)
        x_recon: optional reconstruction back to original space
        k: number of neighbors for kNN recall
        knn_metric: distance metric for kNN recall
        n_sample_pairs: pairs for cosine sim correlation
        seed: RNG seed

    Returns:
        dict of metric_name -> value
    """
    results: dict[str, Any] = {}
    results["knn_recall"] = knn_recall(x_orig, x_comp, k=k, metric=knn_metric)
    results["cosine_sim_correlation"] = cosine_similarity_correlation(
        x_orig, x_comp, n_sample_pairs=n_sample_pairs, seed=seed
    )
    results["effective_rank_orig"] = effective_rank(x_orig)
    results["effective_rank_comp"] = effective_rank(x_comp)
    results["isotropy_orig"] = isotropy_score(x_orig)
    results["isotropy_comp"] = isotropy_score(x_comp)
    if x_recon is not None:
        results["reconstruction_mse"] = reconstruction_mse(x_orig, x_recon)
        results["reconstruction_cosine"] = reconstruction_cosine(x_orig, x_recon)
    return results
