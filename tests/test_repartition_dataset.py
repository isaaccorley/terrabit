from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from tests.test_script_integrations import _fixed_size_list, _load_module


def _bbox_array(rows: list[tuple[float, float, float, float]]) -> pa.Array:
    return pa.array(
        [
            {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}
            for xmin, ymin, xmax, ymax in rows
        ],
        type=pa.struct(
            [
                ("xmin", pa.float64()),
                ("ymin", pa.float64()),
                ("xmax", pa.float64()),
                ("ymax", pa.float64()),
            ]
        ),
    )


def test_repartition_file_writes_grid_layout(tmp_path: Path) -> None:
    repartition = _load_module("repartition_dataset_module", Path("scripts/repartition_dataset.py"))

    src_root = tmp_path / "embeddings"
    src_file = src_root / "geohash_l2=9q" / "year=2024" / "month=06" / "part.parquet"
    out_root = tmp_path / "repartitioned"

    table = pa.table(
        {
            "chips_id": pa.array(["a", "b"]),
            "embedding": _fixed_size_list(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)),
            "bbox": _bbox_array(
                [
                    (-122.43, 37.77, -122.41, 37.79),
                    (-73.99, 40.74, -73.97, 40.76),
                ]
            ),
        }
    )
    meta = {b"quantization": b'{"method":"binary","embedding_col":"embedding","n_dims":16}'}
    table = table.replace_schema_metadata(meta)
    src_file.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, src_file)

    result = repartition.repartition_file(
        str(src_file),
        input_root=str(src_root),
        output_root=str(out_root),
        bbox_col="bbox",
        tile_x_col="tile_x",
        tile_y_col="tile_y",
        tile_width_deg=0.25,
        tile_height_deg=0.25,
        carry_partitions=["year"],
        compression="snappy",
        keep_columns=["chips_id", "embedding", "bbox"],
        max_rows_per_file=5000,
    )
    repartition.write_manifest(
        str(out_root),
        "manifest.parquet",
        input_path=str(src_root),
        tile_x_col="tile_x",
        tile_y_col="tile_y",
        tile_width_deg=0.25,
        tile_height_deg=0.25,
        carry_partitions=["year"],
        compression="snappy",
        entries=result["manifest_entries"],
    )

    assert result["rows"] == 2
    assert result["files"] == 2
    assert len(result["manifest_entries"]) == 2

    out_files = sorted(p for p in out_root.rglob("*.parquet") if p.name != "manifest.parquet")
    assert len(out_files) == 2
    for out_file in out_files:
        rel_parts = out_file.relative_to(out_root).parts
        assert rel_parts[0].startswith("tile_x=")
        assert rel_parts[1].startswith("tile_y=")
        assert rel_parts[2] == "year=2024"
        out_table = pq.ParquetFile(out_file).read()
        assert out_table.num_rows == 1
        assert out_table.schema.metadata == meta
        assert out_table.column("embedding").type == table.column("embedding").type
        assert out_table.column_names == ["chips_id", "embedding", "bbox"]

    manifest = pq.ParquetFile(out_root / "manifest.parquet").read()
    assert manifest.num_rows == 2
    assert manifest.column_names == ["path", "rows", "xmin", "ymin", "xmax", "ymax", "tile_x", "tile_y", "year"]
    schema_meta = manifest.schema.metadata or {}
    assert schema_meta[b"tile_x_col"] == b"tile_x"
    assert schema_meta[b"tile_y_col"] == b"tile_y"
    assert schema_meta[b"tile_width_deg"] == b"0.25"
    assert schema_meta[b"tile_height_deg"] == b"0.25"
    assert schema_meta[b"carry_partitions"] == b"year"
