"""Orchestration: ID estimation, compression, post-compression analysis."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Literal

from terrabit.compression import CompressionMethod, compress_and_write
from terrabit.id_estimation import estimate_intrinsic_dimension
from terrabit.io import preprocess, reservoir_subsample

logging.getLogger(__name__).addHandler(logging.NullHandler())

DEFAULT_TARGET_DIMS = (64, 128)
MAX_EMBEDDING_DIM = 1024


def _target_dims_from_id(estimated_id: float) -> tuple[int, ...]:
    dims = [
        max(2, int(estimated_id // 2)),
        max(2, int(estimated_id)),
        max(2, int(2 * estimated_id)),
        max(2, int(4 * estimated_id)),
        64,
        128,
    ]
    return tuple(sorted({d for d in dims if d <= MAX_EMBEDDING_DIM}))


def run_estimate_id(  # noqa: PLR0913
    path: str,
    *,
    n_subsample: int = 15_000,
    batch_size: int = 50_000,
    embedding_col: str = "embedding",
    l2_norm: bool = False,
    center: bool = False,
    seed: int | None = None,
) -> dict[str, Any]:
    """Subsample embeddings, estimate ID, return report."""
    x = reservoir_subsample(
        path,
        n_subsample,
        batch_size=batch_size,
        embedding_col=embedding_col,
        seed=seed,
    )
    x = preprocess(x, l2_norm=l2_norm, center=center)
    return estimate_intrinsic_dimension(x, seed=seed)


def run_compress(  # noqa: PLR0913
    path: str,
    output_base: str,
    method: CompressionMethod,
    target_dims: tuple[int, ...],
    *,
    batch_size: int = 50_000,
    embedding_col: str = "embedding",
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Compress with given method at each target dim. Returns list of metrics."""
    results: list[dict[str, Any]] = []
    for dim in target_dims:
        out_path = str(Path(output_base) / f"{method}_{dim}")
        meta = compress_and_write(
            path,
            out_path,
            method,
            dim,
            batch_size=batch_size,
            embedding_col=embedding_col,
            seed=seed,
        )
        meta["output_path"] = out_path
        results.append(meta)
    return results


def run_post_compression_id(
    compressed_path: str,
    *,
    n_subsample: int = 15_000,
    batch_size: int = 50_000,
    embedding_col: str = "embedding",
    seed: int | None = None,
) -> dict[str, Any]:
    """Estimate ID on compressed embeddings (subsample + MLE/TwoNN)."""
    x = reservoir_subsample(
        compressed_path,
        n_subsample,
        batch_size=batch_size,
        embedding_col=embedding_col,
        seed=seed,
    )
    return estimate_intrinsic_dimension(x, n_stability_runs=1, seed=seed)


def run_full_pipeline(  # noqa: PLR0913
    path: str,
    output_base: str,
    *,
    n_subsample: int = 15_000,
    batch_size: int = 50_000,
    embedding_col: str = "embedding",
    methods: tuple[CompressionMethod, ...] = ("ipca", "rp", "pca_whitening"),
    target_dims: tuple[int, ...] | Literal["auto"] = "auto",
    run_post_compression_id_analysis: bool = True,
    seed: int | None = None,
) -> dict[str, Any]:
    """Estimate ID, compress, optionally run post-compression ID analysis."""
    id_report = run_estimate_id(
        path,
        n_subsample=n_subsample,
        batch_size=batch_size,
        embedding_col=embedding_col,
        seed=seed,
    )
    dims = (
        _target_dims_from_id(id_report["id_mle"])
        if target_dims == "auto"
        else target_dims or DEFAULT_TARGET_DIMS
    )
    compression_results: dict[str, list[dict[str, Any]]] = {}
    post_compression: dict[str, dict[str, Any]] = {}
    for method in methods:
        compression_results[method] = run_compress(
            path,
            output_base,
            method,
            dims,
            batch_size=batch_size,
            embedding_col=embedding_col,
            seed=seed,
        )
        if run_post_compression_id_analysis:
            for meta in compression_results[method]:
                pc = run_post_compression_id(
                    meta["output_path"],
                    n_subsample=n_subsample,
                    batch_size=batch_size,
                    embedding_col=embedding_col,
                    seed=seed,
                )
                key = f"{method}_{meta['n_components']}"
                post_compression[key] = pc
    return {
        "original_id": id_report,
        "target_dims": list(dims),
        "compression_results": compression_results,
        "post_compression_id": post_compression,
    }


def save_report(report: dict[str, Any], path: str) -> None:
    """Save pipeline report as JSON."""

    def _serialize(obj: object) -> object:
        if isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        if isinstance(obj, tuple):
            return list(obj)
        if isinstance(obj, dict):
            return {k: _serialize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_serialize(x) for x in obj]
        return str(obj)

    with Path(path).open("w") as f:
        json.dump(_serialize(report), f, indent=2)
