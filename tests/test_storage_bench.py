from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

if TYPE_CHECKING:
    from types import ModuleType


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


def _write_dataset(root: Path, values: np.ndarray) -> None:
    part = root / "geohash_l2=aa" / "year=2024"
    part.mkdir(parents=True, exist_ok=True)
    table = pa.table({"embedding": _fixed_size_list(values.astype(np.float32))})
    pq.write_table(table, part / "data.parquet")


def test_storage_bench_helpers_roundtrip(tmp_path: Path) -> None:
    module = _load_module("storage_bench_module", Path("scripts/run_storage_bench.py"))

    sample = np.arange(48, dtype=np.float32).reshape(12, 4)
    quantized = module.quantize(sample, "int8")
    q_values = np.asarray(quantized["quantized"])
    meta = module._make_quant_meta("int8", sample.shape[1], quantized)

    parquet_result = module._benchmark_parquet(
        q_values, sample.shape[1], module.PARQUET_CONFIGS[0], meta
    )
    assert parquet_result["bytes"] > 0
    assert parquet_result["max_abs_error"] == 0.0
    assert parquet_result["dequantize_s"] is not None

    external_result = module._benchmark_external(
        q_values, sample.shape[1], module.EXTERNAL_CONFIGS[-1], meta
    )
    assert external_result["bytes"] > 0
    assert external_result["max_abs_error"] == 0.0
    assert external_result["dequantize_s"] is not None


def test_storage_bench_cli_writes_report(tmp_path: Path) -> None:
    module = _load_module("storage_bench_cli_module", Path("scripts/run_storage_bench.py"))

    data_root = tmp_path / "embeddings"
    out_path = tmp_path / "report.json"
    values = np.arange(80, dtype=np.float32).reshape(20, 4)
    _write_dataset(data_root, values)

    import sys

    argv_prev = sys.argv
    sys.argv = [
        "run_storage_bench.py",
        str(data_root),
        "--output",
        str(out_path),
        "--sample-size",
        "10",
        "--batch-size",
        "5",
        "--methods",
        "float16,int8",
    ]
    try:
        module.main()
    finally:
        sys.argv = argv_prev

    assert out_path.exists()
    report = __import__("json").loads(out_path.read_text())
    assert report["sample_size"] == 10
    assert "float32" in report["methods"]
    assert "float16" in report["methods"]
    assert "int8" in report["methods"]
