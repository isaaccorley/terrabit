"""Parquet streaming and reservoir subsampling for embedding datasets."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

if TYPE_CHECKING:
    from collections.abc import Iterator

    from terrabit._typing import NDArrayF32

logging.getLogger(__name__).addHandler(logging.NullHandler())

BATCH_SIZE_DEFAULT = 50_000
N_SUBSAMPLE_DEFAULT = 15_000


def _extract_embeddings(batch: pa.RecordBatch, embedding_col: str) -> NDArrayF32:
    col = batch.column(embedding_col)
    if isinstance(col, pa.FixedSizeListArray):
        flat = col.values.to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
        n_rows = len(col)
        list_size = col.type.list_size
        return flat.reshape((n_rows, list_size))
    if isinstance(col, (pa.ListArray, pa.LargeListArray)):
        flat = col.values.to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
        offsets = col.offsets.to_numpy()
        n_rows = len(col)
        dim = int(offsets[1] - offsets[0])
        lengths = np.diff(offsets)
        if not np.all(lengths == dim):
            msg = "Variable-length embedding lists not supported"
            raise ValueError(msg)
        return flat.reshape(n_rows, dim)
    msg = f"embedding column '{embedding_col}' must be ListArray or FixedSizeListArray"
    raise ValueError(msg)


def iter_parquet_files(path: str) -> Iterator[str]:
    """Yield .parquet file paths under *path* (sorted for determinism)."""
    root = Path(path)
    if root.is_file():
        yield str(root)
    else:
        yield from sorted(str(p) for p in root.rglob("*.parquet"))


def iter_file_batches(
    file_path: str,
    *,
    batch_size: int = BATCH_SIZE_DEFAULT,
    embedding_col: str = "embedding",
) -> Iterator[tuple[NDArrayF32, pa.RecordBatch]]:
    """Stream (embeddings, record_batch) from a single parquet file."""
    pf = pq.ParquetFile(file_path)
    for batch in pf.iter_batches(batch_size=batch_size):
        x = _extract_embeddings(batch, embedding_col)
        yield x, batch


def iter_embedding_batches(
    path: str,
    *,
    batch_size: int = BATCH_SIZE_DEFAULT,
    embedding_col: str = "embedding",
) -> Iterator[tuple[NDArrayF32, pa.RecordBatch]]:
    """Stream batches from all parquet files under *path*."""
    for file_path in iter_parquet_files(path):
        yield from iter_file_batches(
            file_path, batch_size=batch_size, embedding_col=embedding_col,
        )


def reservoir_subsample(
    path: str,
    n_samples: int = N_SUBSAMPLE_DEFAULT,
    *,
    batch_size: int = BATCH_SIZE_DEFAULT,
    embedding_col: str = "embedding",
    seed: int | None = None,
) -> NDArrayF32:
    """Reservoir sample n_samples embeddings from parquet in a single pass."""
    rng = np.random.default_rng(seed)
    reservoir: NDArrayF32 | None = None
    n_seen = 0
    for x_batch, _ in iter_embedding_batches(
        path,
        batch_size=batch_size,
        embedding_col=embedding_col,
    ):
        dim = x_batch.shape[1]
        for i in range(x_batch.shape[0]):
            if reservoir is None:
                reservoir = np.empty((n_samples, dim), dtype=np.float32)
            if n_seen < n_samples:
                reservoir[n_seen] = x_batch[i]
            else:
                u = rng.integers(0, n_seen + 1)
                if u < n_samples:
                    reservoir[u] = x_batch[i]
            n_seen += 1
    if reservoir is None or n_seen == 0:
        msg = "No data found in path"
        raise ValueError(msg)
    return reservoir[: min(n_samples, n_seen)].copy()


def preprocess(
    x: NDArrayF32,
    *,
    l2_norm: bool = False,
    center: bool = False,
) -> NDArrayF32:
    """Apply optional L2 normalization and centering."""
    if not center and not l2_norm:
        return np.asarray(x, dtype=np.float32)

    out = np.asarray(x, dtype=np.float32).copy()
    if center:
        out -= np.mean(out, axis=0, keepdims=True)
    if l2_norm:
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        out /= norms
    return out
