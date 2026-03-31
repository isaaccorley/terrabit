"""Batched embedding compression: IncrementalPCA, Random Projection, PCA+whitening."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.decomposition import IncrementalPCA
from sklearn.random_projection import GaussianRandomProjection

from terrabit.io import iter_embedding_batches, iter_file_batches, iter_parquet_files

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from terrabit._typing import NDArrayF32

logging.getLogger(__name__).addHandler(logging.NullHandler())
log = logging.getLogger(__name__)

CompressionMethod = Literal["ipca", "rp", "pca_whitening"]


# ---------------------------------------------------------------------------
# Fit helpers
# ---------------------------------------------------------------------------


def fit_incremental_pca(
    path: str,
    n_components: int,
    *,
    batch_size: int = 50_000,
    embedding_col: str = "embedding",
) -> IncrementalPCA:
    """Fit IncrementalPCA on parquet stream (all files)."""
    ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    for x_batch, _ in iter_embedding_batches(
        path,
        batch_size=batch_size,
        embedding_col=embedding_col,
    ):
        ipca.partial_fit(x_batch)
    return ipca


def fit_random_projection(
    n_components: int,
    n_features: int,
    *,
    seed: int | None = None,
) -> GaussianRandomProjection:
    """Create fitted GaussianRandomProjection (no data fit required)."""
    rp = GaussianRandomProjection(n_components=n_components, random_state=seed)
    dummy = np.zeros((1, n_features), dtype=np.float32)
    rp.fit(dummy)
    return rp


def transform_pca_whitening(
    x: NDArrayF32,
    mean: NDArrayF32,
    components: NDArrayF32,
    explained_variance: NDArrayF32,
) -> NDArrayF32:
    """Whiten: (X - mean) @ components.T / sqrt(explained_variance)."""
    centered = x - mean
    proj = centered @ components.T
    scale = np.sqrt(explained_variance)
    scale = np.where(scale > 0, scale, 1.0)
    return (proj / scale).astype(np.float32)


# ---------------------------------------------------------------------------
# Parquet write helpers
# ---------------------------------------------------------------------------


def _make_embedding_column(arr: NDArrayF32) -> pa.Array:
    return pa.FixedSizeListArray.from_arrays(
        pa.array(arr.ravel(), type=pa.float32()),
        list_size=arr.shape[1],
    )


def _write_compressed_file(
    batches: Iterator[tuple[NDArrayF32, pa.RecordBatch]],
    output_path: str,
    embedding_col: str,
    transform_fn: Callable[[NDArrayF32], NDArrayF32],
) -> int:
    """Transform each batch, write to a single parquet file. Returns rows written."""
    writer: pq.ParquetWriter | None = None
    schema: pa.Schema | None = None
    n_written = 0
    for x_batch, rb in batches:
        compressed = transform_fn(x_batch).astype(np.float32)
        col = _make_embedding_column(compressed)
        if writer is None:
            new_field = pa.field(embedding_col, col.type)
            schema = rb.schema.set(
                rb.schema.get_field_index(embedding_col),
                new_field,
            )
            writer = pq.ParquetWriter(output_path, schema)
        new_cols = [col if name == embedding_col else rb.column(name) for name in rb.schema.names]
        out_batch = pa.RecordBatch.from_arrays(new_cols, schema=schema)
        writer.write_batch(out_batch)
        n_written += x_batch.shape[0]
    if writer is not None:
        writer.close()
    return n_written


def _write_compressed_dataset(
    input_path: str,
    output_path: str,
    embedding_col: str,
    transform_fn: Callable[[NDArrayF32], NDArrayF32],
    *,
    batch_size: int = 50_000,
) -> int:
    """Transform every file under *input_path*, preserve partition structure."""
    input_root = Path(input_path)
    output_root = Path(output_path)
    n_written = 0

    for file_path in iter_parquet_files(input_path):
        rel = Path(file_path).relative_to(input_root)
        out_file = output_root / rel
        out_file.parent.mkdir(parents=True, exist_ok=True)

        batches = iter_file_batches(
            file_path,
            batch_size=batch_size,
            embedding_col=embedding_col,
        )
        n = _write_compressed_file(
            batches,
            str(out_file),
            embedding_col,
            transform_fn,
        )
        n_written += n
        log.info("compressed %s -> %s  (%d rows)", rel, out_file, n)

    return n_written


# ---------------------------------------------------------------------------
# Method runners
# ---------------------------------------------------------------------------


def _run_ipca(
    path: str,
    output_path: str,
    n_components: int,
    batch_size: int,
    embedding_col: str,
    *,
    whiten: bool = False,
) -> dict[str, Any]:
    ipca = fit_incremental_pca(
        path,
        n_components,
        batch_size=batch_size,
        embedding_col=embedding_col,
    )
    explained_variance_ratio = np.asarray(ipca.explained_variance_ratio_, dtype=np.float32)
    total_var = float(explained_variance_ratio.sum())

    if whiten:
        mean = np.asarray(ipca.mean_, dtype=np.float32)
        components = np.asarray(ipca.components_, dtype=np.float32)
        ev = np.asarray(ipca.explained_variance_, dtype=np.float32)

        def transform(x: NDArrayF32) -> NDArrayF32:
            return transform_pca_whitening(x, mean, components, ev)
    else:

        def transform(x: NDArrayF32) -> NDArrayF32:
            return ipca.transform(x).astype(np.float32, copy=False)

    n_written = _write_compressed_dataset(
        path,
        output_path,
        embedding_col,
        transform,
        batch_size=batch_size,
    )
    method_name = "pca_whitening" if whiten else "ipca"
    return {
        "method": method_name,
        "n_components": n_components,
        "variance_retained": total_var,
        "n_rows": n_written,
    }


def _run_rp(
    path: str,
    output_path: str,
    n_components: int,
    batch_size: int,
    embedding_col: str,
    seed: int | None,
) -> dict[str, Any]:
    # Peek first batch to get n_features
    first_file = next(iter_parquet_files(path))
    first_batch = next(
        iter_file_batches(first_file, batch_size=1, embedding_col=embedding_col),
    )
    n_features = first_batch[0].shape[1]

    rp = fit_random_projection(n_components, n_features, seed=seed)
    n_written = _write_compressed_dataset(
        path,
        output_path,
        embedding_col,
        rp.transform,
        batch_size=batch_size,
    )
    return {"method": "rp", "n_components": n_components, "n_rows": n_written}


def compress_and_write(
    path: str,
    output_path: str,
    method: CompressionMethod,
    n_components: int,
    *,
    batch_size: int = 50_000,
    embedding_col: str = "embedding",
    seed: int | None = None,
) -> dict[str, Any]:
    """Fit compressor, transform stream, write compressed parquet dataset.

    *output_path* mirrors the partition structure of *path*.
    """
    if method == "ipca":
        return _run_ipca(
            path,
            output_path,
            n_components,
            batch_size,
            embedding_col,
        )
    if method == "rp":
        return _run_rp(
            path,
            output_path,
            n_components,
            batch_size,
            embedding_col,
            seed,
        )
    if method == "pca_whitening":
        return _run_ipca(
            path,
            output_path,
            n_components,
            batch_size,
            embedding_col,
            whiten=True,
        )
    msg = f"Unknown method: {method}"
    raise ValueError(msg)
