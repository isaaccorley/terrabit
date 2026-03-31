"""Repartition a Parquet dataset into browser-friendly delivery tiles.

The script reads every Parquet file under an input directory, assigns each row
to a fixed lat/lon delivery tile using the bbox centroid, and rewrites rows
into a new Hive-style directory tree. It is designed for client-side AOI
retrieval, where the physical shard layout should be tuned for fetch size and
manifest intersection rather than inherited source partitions.

Default layout:
    tile_x=<int>/tile_y=<int>/year=<year>/part-00000.parquet

Example:
    uv run python scripts/repartition_dataset.py \
        results/quantized_dataset_full_allmethods_j8/binary \
        -o results/quantized_dataset_binary_grid \
        --tile-size-deg 0.25 \
        --carry-partitions year \
        --keep-columns chips_id,embedding,bbox
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from rich.console import Console
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn

from terrabit.io import iter_parquet_files

console = Console()

LON_MIN = -180.0
LON_MAX = 180.0
LAT_MIN = -90.0
LAT_MAX = 90.0


def _parse_hive_partitions(rel_path: Path) -> dict[str, str]:
    partitions: dict[str, str] = {}
    for part in rel_path.parts[:-1]:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        partitions[key] = value
    return partitions


def _extract_bbox_centroids(table: pa.Table, bbox_col: str) -> tuple[np.ndarray, np.ndarray]:
    col_index = table.schema.get_field_index(bbox_col)
    if col_index < 0:
        raise ValueError(f"Bounding-box column '{bbox_col}' not found")

    bbox_arr = table.column(col_index).combine_chunks()
    if not isinstance(bbox_arr, pa.StructArray):
        raise TypeError(f"Bounding-box column '{bbox_col}' must be a struct array")

    xmin = bbox_arr.field("xmin").to_numpy(zero_copy_only=False)
    ymin = bbox_arr.field("ymin").to_numpy(zero_copy_only=False)
    xmax = bbox_arr.field("xmax").to_numpy(zero_copy_only=False)
    ymax = bbox_arr.field("ymax").to_numpy(zero_copy_only=False)
    lon = (xmin + xmax) / 2.0
    lat = (ymin + ymax) / 2.0
    return lat, lon


def _append_constant_column(table: pa.Table, name: str, value: str) -> pa.Table:
    if table.schema.get_field_index(name) >= 0:
        return table
    arr = pa.array([value] * table.num_rows, type=pa.string())
    return table.append_column(name, arr)


def _select_columns(
    table: pa.Table,
    *,
    keep_columns: list[str] | None,
    required_columns: list[str],
) -> pa.Table:
    if keep_columns is None:
        return table

    requested = list(dict.fromkeys([*keep_columns, *required_columns]))
    missing = [name for name in requested if table.schema.get_field_index(name) < 0]
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Requested columns not found in input table: {missing_text}")
    return table.select(requested)


def _tile_indices(
    values: np.ndarray, *, min_value: float, max_value: float, step: float
) -> np.ndarray:
    clipped = np.clip(values, min_value, np.nextafter(max_value, min_value))
    return np.floor((clipped - min_value) / step).astype(np.int32)


def _group_indices_from_arrays(
    partition_arrays: list[list[str]],
) -> dict[tuple[str, ...], list[int]]:
    groups: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for idx, key in enumerate(zip(*partition_arrays, strict=True)):
        groups[tuple(str(part) for part in key)].append(idx)
    return groups


def _slice_table(table: pa.Table, indices: list[int]) -> pa.Table:
    return table.take(pa.array(indices, type=pa.int64()))


def _write_partition_table(table: pa.Table, out_file: Path, *, compression: str | None) -> None:
    kwargs: dict[str, Any] = {}
    if compression is not None:
        kwargs["compression"] = compression
    pq.write_table(table, out_file, **kwargs)


def _table_bbox(table: pa.Table, bbox_col: str) -> dict[str, float]:
    bbox_arr = table.column(bbox_col).combine_chunks()
    if not isinstance(bbox_arr, pa.StructArray):
        raise TypeError(f"Bounding-box column '{bbox_col}' must be a struct array")
    xmin = bbox_arr.field("xmin").to_numpy(zero_copy_only=False)
    ymin = bbox_arr.field("ymin").to_numpy(zero_copy_only=False)
    xmax = bbox_arr.field("xmax").to_numpy(zero_copy_only=False)
    ymax = bbox_arr.field("ymax").to_numpy(zero_copy_only=False)
    return {
        "xmin": float(np.min(xmin)),
        "ymin": float(np.min(ymin)),
        "xmax": float(np.max(xmax)),
        "ymax": float(np.max(ymax)),
    }


def _merge_bbox(current: dict[str, float] | None, update: dict[str, float]) -> dict[str, float]:
    if current is None:
        return update
    return {
        "xmin": min(current["xmin"], update["xmin"]),
        "ymin": min(current["ymin"], update["ymin"]),
        "xmax": max(current["xmax"], update["xmax"]),
        "ymax": max(current["ymax"], update["ymax"]),
    }


def _tile_bbox(
    tile_x: str, tile_y: str, *, tile_width_deg: float, tile_height_deg: float
) -> dict[str, float]:
    x = int(tile_x)
    y = int(tile_y)
    xmin = LON_MIN + x * tile_width_deg
    ymin = LAT_MIN + y * tile_height_deg
    return {
        "xmin": xmin,
        "ymin": ymin,
        "xmax": xmin + tile_width_deg,
        "ymax": ymin + tile_height_deg,
    }


@dataclass
class OpenShard:
    writer: pq.ParquetWriter
    out_file: Path
    rows: int
    bbox: dict[str, float] | None
    partitions: dict[str, str]


class ShardWriterManager:
    def __init__(
        self,
        *,
        output_root: str,
        bbox_col: str,
        partition_cols: list[str],
        compression: str | None,
        max_rows_per_file: int,
    ) -> None:
        self.output_root = Path(output_root)
        self.bbox_col = bbox_col
        self.partition_cols = partition_cols
        self.compression = compression
        self.max_rows_per_file = max_rows_per_file
        self.open_shards: dict[tuple[str, ...], OpenShard] = {}
        self.next_part_index: dict[tuple[str, ...], int] = defaultdict(int)
        self.manifest_entries: list[dict[str, Any]] = []
        self.files_written = 0

    def _new_file_path(self, key: tuple[str, ...]) -> tuple[Path, dict[str, str]]:
        out_dir = self.output_root
        partitions = dict(zip(self.partition_cols, key, strict=True))
        for name, value in partitions.items():
            out_dir = out_dir / f"{name}={value}"
        out_dir.mkdir(parents=True, exist_ok=True)
        part_index = self.next_part_index[key]
        self.next_part_index[key] += 1
        return out_dir / f"part-{part_index:05d}.parquet", partitions

    def _open_writer(self, key: tuple[str, ...], schema: pa.Schema) -> OpenShard:
        out_file, partitions = self._new_file_path(key)
        kwargs: dict[str, Any] = {}
        if self.compression is not None:
            kwargs["compression"] = self.compression
        writer = pq.ParquetWriter(out_file, schema, **kwargs)
        shard = OpenShard(
            writer=writer, out_file=out_file, rows=0, bbox=None, partitions=partitions
        )
        self.open_shards[key] = shard
        return shard

    def _close_writer(self, key: tuple[str, ...]) -> None:
        shard = self.open_shards.pop(key)
        shard.writer.close()
        self.files_written += 1
        if shard.bbox is None:
            raise ValueError(f"Shard {shard.out_file} was closed without any rows")
        self.manifest_entries.append(
            {
                "path": str(shard.out_file.relative_to(self.output_root)),
                "rows": shard.rows,
                "bbox": shard.bbox,
                "partitions": shard.partitions,
            }
        )

    def write_partition(self, key: tuple[str, ...], table: pa.Table) -> None:
        offset = 0
        while offset < table.num_rows:
            shard = self.open_shards.get(key)
            if shard is None:
                shard = self._open_writer(key, table.schema)

            remaining = self.max_rows_per_file - shard.rows
            if remaining <= 0:
                self._close_writer(key)
                continue

            length = min(remaining, table.num_rows - offset)
            chunk = table.slice(offset, length)
            shard.writer.write_table(chunk)
            shard.rows += chunk.num_rows
            shard.bbox = _merge_bbox(shard.bbox, _table_bbox(chunk, self.bbox_col))
            offset += length

            if shard.rows >= self.max_rows_per_file:
                self._close_writer(key)

    def close(self) -> list[dict[str, Any]]:
        for key in list(self.open_shards):
            self._close_writer(key)
        return sorted(self.manifest_entries, key=lambda entry: entry["path"])


def repartition_file(
    src_file: str,
    *,
    input_root: str,
    output_root: str,
    bbox_col: str,
    tile_x_col: str,
    tile_y_col: str,
    tile_width_deg: float,
    tile_height_deg: float,
    carry_partitions: list[str],
    compression: str | None,
    keep_columns: list[str] | None,
    max_rows_per_file: int = 5000,
) -> dict[str, Any]:
    src_path = Path(src_file)
    rel_path = src_path.relative_to(input_root)
    table = pq.ParquetFile(src_path).read()
    table = _select_columns(table, keep_columns=keep_columns, required_columns=[bbox_col])
    latitudes, longitudes = _extract_bbox_centroids(table, bbox_col)
    tile_x = _tile_indices(longitudes, min_value=LON_MIN, max_value=LON_MAX, step=tile_width_deg)
    tile_y = _tile_indices(latitudes, min_value=LAT_MIN, max_value=LAT_MAX, step=tile_height_deg)
    hive_partitions = _parse_hive_partitions(rel_path)
    partition_cols = [tile_x_col, tile_y_col, *carry_partitions]
    partition_arrays: list[list[str]] = [tile_x.astype(str).tolist(), tile_y.astype(str).tolist()]
    for name in carry_partitions:
        if name not in hive_partitions:
            raise ValueError(f"Partition '{name}' not found in source path {rel_path}")
        partition_arrays.append([hive_partitions[name]] * table.num_rows)

    manager = ShardWriterManager(
        output_root=output_root,
        bbox_col=bbox_col,
        partition_cols=partition_cols,
        compression=compression,
        max_rows_per_file=max_rows_per_file,
    )
    groups = _group_indices_from_arrays(partition_arrays)
    for key, indices in sorted(groups.items()):
        manager.write_partition(key, _slice_table(table, indices))

    manifest_entries = manager.close()
    return {
        "rows": table.num_rows,
        "files": manager.files_written,
        "manifest_entries": manifest_entries,
    }


def write_manifest(
    output_root: str,
    manifest_name: str,
    *,
    input_path: str,
    tile_x_col: str,
    tile_y_col: str,
    tile_width_deg: float,
    tile_height_deg: float,
    carry_partitions: list[str],
    compression: str | None,
    entries: list[dict[str, Any]],
) -> Path:
    manifest_path = Path(output_root) / manifest_name
    ordered_entries = sorted(entries, key=lambda entry: entry["path"])
    payload = pa.table(
        {
            "path": pa.array([entry["path"] for entry in ordered_entries], type=pa.string()),
            "rows": pa.array([entry["rows"] for entry in ordered_entries], type=pa.int64()),
            "xmin": pa.array(
                [entry["bbox"]["xmin"] for entry in ordered_entries], type=pa.float64()
            ),
            "ymin": pa.array(
                [entry["bbox"]["ymin"] for entry in ordered_entries], type=pa.float64()
            ),
            "xmax": pa.array(
                [entry["bbox"]["xmax"] for entry in ordered_entries], type=pa.float64()
            ),
            "ymax": pa.array(
                [entry["bbox"]["ymax"] for entry in ordered_entries], type=pa.float64()
            ),
            "tile_x": pa.array(
                [entry["partitions"][tile_x_col] for entry in ordered_entries], type=pa.string()
            ),
            "tile_y": pa.array(
                [entry["partitions"][tile_y_col] for entry in ordered_entries], type=pa.string()
            ),
            **{
                name: pa.array(
                    [entry["partitions"].get(name, "") for entry in ordered_entries],
                    type=pa.string(),
                )
                for name in carry_partitions
            },
        },
        metadata={
            b"manifest_version": b"1",
            b"input_path": input_path.encode("utf-8"),
            b"output_root": output_root.encode("utf-8"),
            b"tile_x_col": tile_x_col.encode("utf-8"),
            b"tile_y_col": tile_y_col.encode("utf-8"),
            b"tile_width_deg": str(tile_width_deg).encode("utf-8"),
            b"tile_height_deg": str(tile_height_deg).encode("utf-8"),
            b"carry_partitions": ",".join(carry_partitions).encode("utf-8"),
            b"compression": (compression or "none").encode("utf-8"),
        },
    )
    pq.write_table(payload, manifest_path, compression="zstd")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Input parquet dataset root")
    parser.add_argument("--output", "-o", required=True, help="Output dataset root")
    parser.add_argument(
        "--bbox-col",
        default="bbox",
        help="Struct column used to compute centroids (default: bbox)",
    )
    parser.add_argument(
        "--tile-size-deg",
        type=float,
        default=0.25,
        help="Square delivery tile size in degrees (default: 0.25)",
    )
    parser.add_argument(
        "--tile-width-deg",
        type=float,
        default=None,
        help="Optional tile width override in degrees",
    )
    parser.add_argument(
        "--tile-height-deg",
        type=float,
        default=None,
        help="Optional tile height override in degrees",
    )
    parser.add_argument(
        "--tile-x-col",
        default="tile_x",
        help="Tile x partition column name (default: tile_x)",
    )
    parser.add_argument(
        "--tile-y-col",
        default="tile_y",
        help="Tile y partition column name (default: tile_y)",
    )
    parser.add_argument(
        "--carry-partitions",
        default="year",
        help="Comma-separated Hive partition keys to preserve from input paths",
    )
    parser.add_argument(
        "--keep-columns",
        default="chips_id,embedding,bbox",
        help=(
            "Optional comma-separated subset of columns to keep in output shards. "
            "The bbox column and generated partition columns are always preserved."
        ),
    )
    parser.add_argument(
        "--compression",
        choices=("none", "snappy", "zstd"),
        default="snappy",
        help="Compression codec for output Parquet files",
    )
    parser.add_argument(
        "--limit-files",
        type=int,
        default=0,
        help="If >0, only repartition the first N parquet files",
    )
    parser.add_argument(
        "--max-rows-per-file",
        type=int,
        default=5000,
        help="Target maximum rows per output shard file",
    )
    parser.add_argument(
        "--manifest-name",
        default="manifest.parquet",
        help="Manifest file name written under the output root",
    )
    args = parser.parse_args()

    files = list(iter_parquet_files(args.path))
    if args.limit_files > 0:
        files = files[: args.limit_files]
    if not files:
        raise FileNotFoundError(f"No parquet files found under {args.path}")

    tile_width_deg = args.tile_width_deg or args.tile_size_deg
    tile_height_deg = args.tile_height_deg or args.tile_size_deg
    if tile_width_deg <= 0 or tile_height_deg <= 0:
        raise ValueError("Tile width and height must be positive")

    compression = None if args.compression == "none" else args.compression
    carry_partitions = [part.strip() for part in args.carry_partitions.split(",") if part.strip()]
    keep_columns = None
    if args.keep_columns:
        keep_columns = [part.strip() for part in args.keep_columns.split(",") if part.strip()]

    total_rows = 0
    manifest_entries: list[dict[str, Any]] = []

    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    )
    with progress:
        task_id = progress.add_task("Repartitioning", total=len(files))
        for src_file in files:
            result = repartition_file(
                src_file,
                input_root=args.path,
                output_root=args.output,
                bbox_col=args.bbox_col,
                tile_x_col=args.tile_x_col,
                tile_y_col=args.tile_y_col,
                tile_width_deg=tile_width_deg,
                tile_height_deg=tile_height_deg,
                carry_partitions=carry_partitions,
                compression=compression,
                keep_columns=keep_columns,
                max_rows_per_file=args.max_rows_per_file,
            )
            total_rows += result["rows"]
            manifest_entries.extend(result["manifest_entries"])
            progress.advance(task_id)

    manifest_path = write_manifest(
        args.output,
        args.manifest_name,
        input_path=args.path,
        tile_x_col=args.tile_x_col,
        tile_y_col=args.tile_y_col,
        tile_width_deg=tile_width_deg,
        tile_height_deg=tile_height_deg,
        carry_partitions=carry_partitions,
        compression=compression,
        entries=manifest_entries,
    )

    console.print(
        f"Repartitioned {len(files)} source files into {len(manifest_entries)} output files "
        f"covering {total_rows} rows with tile size {tile_width_deg} x {tile_height_deg} deg "
        f"and a shard target of {args.max_rows_per_file} rows/file. "
        f"Wrote manifest to {manifest_path}."
    )


if __name__ == "__main__":
    main()
