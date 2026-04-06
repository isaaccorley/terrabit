#!/usr/bin/env python3
"""Validate a GeoParquet file exported from the TerraBit demo app.

Usage:
    uv run --with geoparquet-io scripts/validate_geoparquet.py <file.parquet>
"""

import sys

import geoparquet_io as gpio


def validate(path: str) -> bool:
    table = gpio.read(path)
    table.info()
    print()

    result = table.validate()
    if result.passed():
        print(f"PASS  valid GeoParquet {table.geoparquet_version}")
    else:
        print("FAIL  GeoParquet validation errors:")
        for f in result.failures():
            print(f"  - {f}")

    best_practice = table.check()
    if not best_practice.passed():
        print("WARN  best-practice checks:")
        for f in best_practice.failures():
            print(f"  - {f}")

    return result.passed()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    sys.exit(0 if validate(sys.argv[1]) else 1)
