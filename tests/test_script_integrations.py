from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from terrabit.quantization import quantize

if TYPE_CHECKING:
    from types import ModuleType

    from terrabit.quantization import QuantizationMethod


def _load_module(module_name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _fixed_size_list(values_2d: np.ndarray) -> pa.FixedSizeListArray:
    flat = pa.array(values_2d.reshape(-1), type=pa.from_numpy_dtype(values_2d.dtype))
    return pa.FixedSizeListArray.from_arrays(flat, values_2d.shape[1])


def _write_source_file(path: Path, values: np.ndarray) -> None:
    table = pa.table({"embedding": _fixed_size_list(values.astype(np.float32))})
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def _write_quantized_file(path: Path, values: np.ndarray, method: str) -> None:
    quantized = quantize(values.astype(np.float32), cast("QuantizationMethod", method))
    q_values = np.asarray(quantized["quantized"])
    table = pa.table({"embedding": _fixed_size_list(q_values)})

    metadata = {
        "method": method,
        "embedding_col": "embedding",
        "n_dims": int(quantized.get("n_dims", values.shape[1])),
    }
    for key, value in quantized.items():
        if key in {"quantized", "method"}:
            continue
        if isinstance(value, np.ndarray):
            metadata[key] = np.asarray(value, dtype=np.float32).tolist()
        elif isinstance(value, (np.integer, int)):
            metadata[key] = int(value)
        elif isinstance(value, (np.floating, float)):
            metadata[key] = float(value)

    schema_meta = dict(table.schema.metadata or {})
    schema_meta[b"quantization"] = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table.replace_schema_metadata(schema_meta), path)


def test_load_quantized_dequantized_uses_per_file_metadata(
    tmp_path: Path,
    monkeypatch,
) -> None:
    load_data = _load_module("load_data_module", Path("scripts/load_data.py"))

    source_root = tmp_path / "embeddings"
    quant_root = tmp_path / "results" / "quantized_dataset_full_allmethods_j8"
    method = "int8"

    file_a = np.array([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]], dtype=np.float32)
    file_b = np.array([[100.0, 101.0, 102.0], [103.0, 104.0, 105.0]], dtype=np.float32)

    src_a = source_root / "part_a.parquet"
    src_b = source_root / "part_b.parquet"
    _write_source_file(src_a, file_a)
    _write_source_file(src_b, file_b)
    _write_quantized_file(quant_root / method / "part_a.parquet", file_a, method)
    _write_quantized_file(quant_root / method / "part_b.parquet", file_b, method)

    monkeypatch.setattr(load_data, "EMBEDDING_DIR", str(source_root))
    monkeypatch.setattr(load_data, "QUANTIZED_DIR", str(quant_root))

    decoded = load_data.load_quantized_dequantized(method, [str(src_a), str(src_b)])

    expected = np.concatenate([file_a, file_b], axis=0)
    np.testing.assert_allclose(decoded, expected, atol=0.05)


def test_sample_reservoir_uses_matching_metadata_for_each_sample(tmp_path: Path) -> None:
    run_metrics_batched = _load_module(
        "run_metrics_batched_module",
        Path("scripts/run_metrics_batched.py"),
    )

    source_root = tmp_path / "embeddings"
    quant_root = tmp_path / "quantized"
    method = "int8"

    file_a = np.array(
        [[0.0, 1.0, 2.0], [1.5, 2.5, 3.5], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]],
        dtype=np.float32,
    )
    file_b = np.array(
        [
            [100.0, 101.0, 102.0],
            [102.0, 103.0, 104.0],
            [104.0, 105.0, 106.0],
            [106.0, 107.0, 108.0],
        ],
        dtype=np.float32,
    )

    src_a = source_root / "part_a.parquet"
    src_b = source_root / "part_b.parquet"
    quant_a = quant_root / "part_a.parquet"
    quant_b = quant_root / "part_b.parquet"
    _write_source_file(src_a, file_a)
    _write_source_file(src_b, file_b)
    _write_quantized_file(quant_a, file_a, method)
    _write_quantized_file(quant_b, file_b, method)

    pairs = [(src_a, quant_a), (src_b, quant_b)]
    sample_orig, sample_recon, sample_n = run_metrics_batched._sample_reservoir(
        pairs=pairs,
        reservoir_size=8,
        batch_size=2,
        seed=0,
        progress_every=10,
        method=method,
    )

    assert sample_n == 8
    np.testing.assert_allclose(sample_orig, np.concatenate([file_a, file_b], axis=0), atol=1e-6)
    np.testing.assert_allclose(sample_recon, sample_orig, atol=0.05)


def test_quantize_script_respects_parquet_write_options(tmp_path: Path) -> None:
    quantize_script = _load_module("quantize_script_module", Path("scripts/quantize.py"))

    source_root = tmp_path / "embeddings"
    out_root = tmp_path / "quantized"
    values = np.arange(24, dtype=np.float32).reshape(6, 4)
    src = source_root / "part.parquet"
    _write_source_file(src, values)

    options = quantize_script.ParquetWriteOptions(
        compression="zstd",
        compression_level=1,
        use_byte_stream_split=True,
        row_group_size=2,
    )
    quantize_script._process_one_file(
        str(src),
        str(source_root),
        str(out_root),
        ("float16",),
        "embedding",
        options,
    )

    out_file = out_root / "float16" / "part.parquet"
    meta = pq.ParquetFile(out_file).metadata.row_group(0).column(0)
    assert meta.compression == "ZSTD"
    assert pq.read_table(out_file).num_rows == 6

    out_file_bss = out_root / "float16" / "part.parquet"
    quantize_script._write_parquet_table(
        pq.read_table(out_file_bss),
        out_file_bss,
        quantize_script.ParquetWriteOptions(
            compression="zstd",
            compression_level=1,
            use_byte_stream_split=True,
        ),
    )
    meta_bss = pq.ParquetFile(out_file_bss).metadata.row_group(0).column(0)
    assert "BYTE_STREAM_SPLIT" in meta_bss.encodings
