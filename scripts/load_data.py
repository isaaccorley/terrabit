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
    """Load source float32 embeddings with early-stop.

    Returns:
        embeddings: (n_samples, dim) float32 array
        files_used: list of parquet file paths that were read (in order)
        chosen_idx: the subsample indices within the concatenated file data,
                    or None if no subsampling was needed.
    """
    files = iter_parquet_files(EMBEDDING_DIR)
    if not files:
        msg = f"No parquet files found under {EMBEDDING_DIR}"
        raise FileNotFoundError(msg)

    rng = np.random.default_rng(seed)
    file_order = rng.permutation(len(files)).tolist()
    target = n_samples * 3 if n_samples else None  # over-read for diversity

    all_embs: list[np.ndarray] = []
    files_used: list[str] = []
    total_rows = 0

    for fi in file_order:
        fp = files[fi]
        pf = pq.ParquetFile(fp)
        for batch in pf.iter_batches(batch_size=50_000, columns=["embedding"]):
            embs = _read_embedding_column(batch.column("embedding")).astype(np.float32, copy=False)
            all_embs.append(embs)
            total_rows += len(embs)
        files_used.append(fp)
        if target and total_rows >= target:
            break

    embeddings = np.concatenate(all_embs, axis=0)
    chosen_idx: np.ndarray | None = None

    if n_samples is not None and n_samples < len(embeddings):
        chosen_idx = rng.choice(len(embeddings), size=n_samples, replace=False)
        chosen_idx.sort()
        embeddings = embeddings[chosen_idx]

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
