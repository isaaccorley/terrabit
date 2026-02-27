"""Integration tests for partitioned compression pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from terrabit.compression import CompressionMethod, compress_and_write
from terrabit.io import iter_embedding_batches, iter_parquet_files


def _write_fake_dataset(root: Path, n_partitions: int = 2, n_rows: int = 200) -> None:
    """Create a minimal Hive-partitioned parquet dataset."""
    rng = np.random.default_rng(42)
    dim = 64
    for i in range(n_partitions):
        part_dir = root / f"geohash_l2=g{i}" / "year=2024" / "month=06"
        part_dir.mkdir(parents=True)
        emb = rng.standard_normal((n_rows, dim)).astype(np.float32)
        col = pa.FixedSizeListArray.from_arrays(
            pa.array(emb.ravel(), type=pa.float32()),
            list_size=dim,
        )
        table = pa.table({"id": list(range(n_rows)), "embedding": col})
        pq.write_table(table, part_dir / "data.parquet")


class TestPartitionedIO:
    def test_iter_parquet_files(self, tmp_path: Path) -> None:
        _write_fake_dataset(tmp_path, n_partitions=3)
        files = list(iter_parquet_files(str(tmp_path)))
        assert len(files) == 3
        assert all(f.endswith(".parquet") for f in files)

    def test_iter_embedding_batches(self, tmp_path: Path) -> None:
        _write_fake_dataset(tmp_path, n_partitions=2, n_rows=100)
        total = 0
        for x, _rb in iter_embedding_batches(str(tmp_path), batch_size=50):
            assert x.shape[1] == 64
            assert x.dtype == np.float32
            total += x.shape[0]
        assert total == 200


class TestCompressAndWrite:
    @pytest.mark.parametrize("method", ["ipca", "rp", "pca_whitening"])
    def test_preserves_partition_structure(
        self,
        tmp_path: Path,
        method: CompressionMethod,
    ) -> None:
        src = tmp_path / "src"
        dst = tmp_path / "dst"
        _write_fake_dataset(src, n_partitions=2, n_rows=150)

        meta = compress_and_write(
            str(src),
            str(dst),
            method,
            n_components=8,
            batch_size=100,
            seed=0,
        )
        assert meta["n_rows"] == 300
        assert meta["n_components"] == 8

        # Output should mirror source partition tree
        src_files = sorted(str(Path(f).relative_to(src)) for f in iter_parquet_files(str(src)))
        dst_files = sorted(str(Path(f).relative_to(dst)) for f in iter_parquet_files(str(dst)))
        assert src_files == dst_files

        # Compressed embeddings should have target dim
        for x, _ in iter_embedding_batches(str(dst), batch_size=100):
            assert x.shape[1] == 8

    def test_non_embedding_columns_preserved(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        dst = tmp_path / "dst"
        _write_fake_dataset(src, n_partitions=1, n_rows=50)

        compress_and_write(
            str(src),
            str(dst),
            "ipca",
            n_components=4,
            batch_size=50,
        )

        # Read back and verify non-embedding cols survive
        out_file = next(iter_parquet_files(str(dst)))
        t = pq.read_table(out_file)
        assert "id" in t.column_names
        assert t.num_rows == 50
