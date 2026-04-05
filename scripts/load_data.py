"""Shared data loading utilities for vector search experiments.

Loads source float32 and quantized embeddings from aligned Hive-partitioned
Parquet files, with early-stop for fast sampling.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

EMBEDDING_DIR = "embeddings"
QUANTIZED_DIR = "results/quantized_dataset_full_allmethods_j8"


def iter_parquet_files(path: str) -> list[str]:
    """Return sorted list of all .parquet files under path."""
    root = Path(path)
    if root.is_file():
        return [str(root)]
    return sorted(str(p) for p in root.rglob("*.parquet"))


def _read_embedding_column(col: pa.Array) -> np.ndarray:
    """Convert an Arrow embedding column to a 2-D numpy array."""
    if isinstance(col, pa.FixedSizeListArray):
        flat = col.values.to_numpy(zero_copy_only=False)
        return flat.reshape(len(col), col.type.list_size)
    if isinstance(col, (pa.ListArray, pa.LargeListArray)):
        flat = col.values.to_numpy(zero_copy_only=False)
        offsets = col.offsets.to_numpy()
        dim = int(offsets[1] - offsets[0])
        return flat.reshape(len(col), dim)
    msg = f"Unexpected column type: {type(col)}"
    raise TypeError(msg)


def load_source_embeddings(
    n_samples: int | None = None,
    *,
    seed: int = 42,
) -> tuple[np.ndarray, list[str], np.ndarray | None]:
    """Load source float32 embeddings with global stratified sampling.

    For n_samples < total corpus, allocates samples per file proportional to
    the file's row count (multinomial) and reads only the required row groups
    from each file. This guarantees coverage of every geohash partition,
    avoiding the single-partition bias of the previous early-stop implementation.

    Returns:
        embeddings: (n_samples, dim) float32 array, shuffled
        files_used: list of parquet file paths that were read (in order)
        chosen_idx: the per-file offset indices for row-aligned loading of
                    quantized counterparts, or None if no subsampling happened.
    """
    files = iter_parquet_files(EMBEDDING_DIR)
    if not files:
        msg = f"No parquet files found under {EMBEDDING_DIR}"
        raise FileNotFoundError(msg)

    rng = np.random.default_rng(seed)

    # Metadata-only pass: get row counts per file.
    file_rows = np.array(
        [pq.ParquetFile(fp).metadata.num_rows for fp in files], dtype=np.int64
    )
    total_rows = int(file_rows.sum())

    # Full-corpus read path (unchanged semantics).
    if n_samples is None or n_samples >= total_rows:
        all_embs: list[np.ndarray] = []
        for fp in files:
            pf = pq.ParquetFile(fp)
            for batch in pf.iter_batches(batch_size=50_000, columns=["embedding"]):
                embs = _read_embedding_column(batch.column("embedding")).astype(
                    np.float32, copy=False
                )
                all_embs.append(embs)
        return np.concatenate(all_embs, axis=0), list(files), None

    # Proportional allocation: draw n_samples via multinomial over files.
    probs = file_rows / total_rows
    allocation = rng.multinomial(n_samples, probs)

    # Concatenation order is fixed (file iteration order), so chosen_idx is
    # well-defined w.r.t. the concatenated quantized-file read in
    # load_quantized_*. For each file we pick `allocation[i]` random row
    # indices (sorted); quantized loaders apply the same indices on the
    # same file order.
    per_file_idx: list[np.ndarray] = []
    all_embs = []
    concat_offset = 0
    chosen_idx_parts: list[np.ndarray] = []
    files_used: list[str] = []

    for fp, nr, k in zip(files, file_rows, allocation, strict=True):
        if k == 0:
            per_file_idx.append(np.empty(0, dtype=np.int64))
            continue
        k = int(k)
        nr = int(nr)
        # Random indices within this file.
        if k >= nr:
            idx = np.arange(nr, dtype=np.int64)
        else:
            idx = rng.choice(nr, size=k, replace=False)
            idx.sort()
        per_file_idx.append(idx)

        # Read only the embedding column for this file, then index.
        pf = pq.ParquetFile(fp)
        batches = []
        for batch in pf.iter_batches(batch_size=50_000, columns=["embedding"]):
            batches.append(
                _read_embedding_column(batch.column("embedding")).astype(
                    np.float32, copy=False
                )
            )
        file_arr = np.concatenate(batches, axis=0) if len(batches) > 1 else batches[0]
        all_embs.append(file_arr[idx])
        chosen_idx_parts.append(idx + concat_offset)
        concat_offset += nr
        files_used.append(fp)

    embeddings = np.concatenate(all_embs, axis=0)
    chosen_idx = np.concatenate(chosen_idx_parts, axis=0)

    # Shuffle to break file-order correlation in downstream consumers.
    perm = rng.permutation(len(embeddings))
    embeddings = embeddings[perm]
    chosen_idx = chosen_idx[perm]

    return embeddings, files_used, chosen_idx


def _source_to_quantized_files(source_files: list[str], method: str) -> list[str]:
    """Map source parquet paths to their quantized counterparts."""
    src_root = Path(EMBEDDING_DIR)
    q_root = Path(QUANTIZED_DIR) / method
    qfiles = []
    for sf in source_files:
        rel = Path(sf).relative_to(src_root)
        qf = q_root / rel
        if qf.exists():
            qfiles.append(str(qf))
    return qfiles


def load_quantized_dequantized(
    method: str,
    source_files: list[str],
    chosen_idx: np.ndarray | None = None,
) -> np.ndarray:
    """Load quantized embeddings from the same files as source, dequantize to float32.

    Reads the same partition files, in the same order, so rows align with source.
    Uses chosen_idx (from load_source_embeddings) for identical row selection.
    """
    from terrabit.quantization import dequantize

    qfiles = _source_to_quantized_files(source_files, method)
    if not qfiles:
        msg = f"No quantized files found for method={method}"
        raise FileNotFoundError(msg)

    all_embs: list[np.ndarray] = []
    for fp in qfiles:
        pf = pq.ParquetFile(fp)
        meta = pf.schema_arrow.metadata
        if meta is None or b"quantization" not in meta:
            msg = f"Missing quantization metadata in {fp}"
            raise ValueError(msg)
        qmeta = json.loads(meta[b"quantization"].decode("utf-8"))

        file_embs: list[np.ndarray] = []
        for batch in pf.iter_batches(batch_size=50_000, columns=["embedding"]):
            raw = _read_embedding_column(batch.column("embedding"))
            file_embs.append(raw)

        raw_all = np.concatenate(file_embs, axis=0)
        result: dict = {"quantized": raw_all, "method": method}
        if "scale" in qmeta:
            result["scale"] = np.array(qmeta["scale"], dtype=np.float32)
        if "zero_point" in qmeta:
            result["zero_point"] = np.array(qmeta["zero_point"], dtype=np.float32)
        if "n_dims" in qmeta:
            result["n_dims"] = qmeta["n_dims"]
        if "turbo_bits" in qmeta:
            result["turbo_bits"] = qmeta["turbo_bits"]
        if "turbo_seed" in qmeta:
            result["turbo_seed"] = qmeta["turbo_seed"]

        all_embs.append(dequantize(result))

    dequantized_all = np.concatenate(all_embs, axis=0)

    if chosen_idx is not None:
        dequantized_all = dequantized_all[chosen_idx]

    return dequantized_all


def load_quantized_binary_raw(
    source_files: list[str],
    chosen_idx: np.ndarray | None = None,
) -> np.ndarray:
    """Load binary-quantized packed uint8 bytes directly (for FAISS IndexBinaryFlat)."""
    qfiles = _source_to_quantized_files(source_files, "binary")
    if not qfiles:
        msg = "No quantized binary files found"
        raise FileNotFoundError(msg)

    all_embs: list[np.ndarray] = []
    for fp in qfiles:
        pf = pq.ParquetFile(fp)
        for batch in pf.iter_batches(batch_size=50_000, columns=["embedding"]):
            raw = _read_embedding_column(batch.column("embedding")).astype(np.uint8)
            all_embs.append(raw)

    raw_all = np.concatenate(all_embs, axis=0)

    if chosen_idx is not None:
        raw_all = raw_all[chosen_idx]

    return raw_all
