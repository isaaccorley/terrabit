"""Compute turbo1 cosine kNN recall under the same protocol as turbo{2,3,4,8}.

The `turbo_vs_int.pdf` figure in `paper/make_figures.py` reads
`knn_recall_10_cosine_sampled` from `report_metrics_batched_turbo_full.json`.
Those numbers come from a 20k reservoir sample of the full 50M corpus,
reconstructed in float32 space, evaluated with sklearn NearestNeighbors
(metric="cosine").

This script matches that protocol as closely as possible without re-streaming
the full 50M corpus (we don't have turbo1-quantized parquet on disk):

  1. Load a 200k random sample from the Clay corpus (seed=42).
  2. Draw a 20k reservoir-like subsample from it (matches the turbo_full run's
     reservoir_size=20000).
  3. Apply turbo1: random orthogonal rotation → sign → {-1,+1} → inverse rotation.
     This gives a float32 reconstruction in the original space, just like
     `dequantize_turbo` does for turbo{2,3,4,8}.
  4. Compute cosine kNN recall @ {10, 25, 50}.
  5. Patch `results/report_metrics_batched_turbo_full.json` with a new
     `turbo1` entry so `make_figures.py` picks it up automatically.

Usage:
    python scripts/run_turbo1_cosine_bench.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

import load_data  # noqa: E402

load_data.EMBEDDING_DIR = "embeddings/clay-v1-5-sentinel-2"
from load_data import load_source_embeddings  # noqa: E402

from terrabit.metrics import knn_recall_multi  # noqa: E402
from terrabit.quantization import _random_orthogonal_matrix  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

TURBO_FULL_PATH = (
    Path(__file__).resolve().parent.parent
    / "results"
    / "report_metrics_batched_turbo_full.json"
)

RESERVOIR_SIZE = 20_000
N_POOL = 200_000  # pool to draw the reservoir from
SEED = 42
ROT_SEED = 0


def turbo1_reconstruct(x: np.ndarray, rot_seed: int = ROT_SEED) -> np.ndarray:
    """Turbo1 forward+inverse: rotate → sign({-1,+1}) → inverse-rotate."""
    d = x.shape[1]
    r = _random_orthogonal_matrix(d, seed=rot_seed)
    x_rot = x @ r
    signs = np.where(x_rot > 0, np.float32(1.0), np.float32(-1.0))
    return signs @ r.T


def main() -> None:
    log.info("Loading %d source embeddings (seed=%d)...", N_POOL, SEED)
    t0 = time.perf_counter()
    pool, _, _ = load_source_embeddings(n_samples=N_POOL, seed=SEED)
    log.info("Loaded %d × %d in %.1fs", *pool.shape, time.perf_counter() - t0)

    rng = np.random.default_rng(SEED)
    sample_idx = rng.choice(len(pool), size=RESERVOIR_SIZE, replace=False)
    sample_idx.sort()
    x_orig = pool[sample_idx].astype(np.float32, copy=False)
    log.info("Drew reservoir sample: %s", x_orig.shape)

    log.info("Computing turbo1 reconstruction (rotate → sign → inverse-rotate)...")
    t0 = time.perf_counter()
    x_recon = turbo1_reconstruct(x_orig, rot_seed=ROT_SEED)
    log.info("Reconstructed in %.2fs", time.perf_counter() - t0)

    log.info("Computing cosine kNN recall @ {10, 25, 50}...")
    t0 = time.perf_counter()
    recalls = knn_recall_multi(x_orig, x_recon, ks=(10, 25, 50), metric="cosine")
    log.info("kNN recall computed in %.1fs", time.perf_counter() - t0)

    # Reconstruction metrics (for parity with existing turbo entries)
    diff = x_orig - x_recon
    mse = float(np.mean(diff * diff))
    orig_norm = np.linalg.norm(x_orig, axis=1)
    recon_norm = np.linalg.norm(x_recon, axis=1)
    cos = float(
        np.mean(
            np.sum(x_orig * x_recon, axis=1)
            / np.where(orig_norm * recon_norm > 0, orig_norm * recon_norm, 1.0)
        )
    )
    log.info("recon MSE=%.6f  recon cosine=%.4f", mse, cos)
    for k, v in recalls.items():
        log.info("  knn_recall_%d_cosine_sampled = %.4f", k, v)

    # Build entry matching existing turbo_full schema.
    entry = {
        "method": "turbo1",
        "n_rows": int(RESERVOIR_SIZE),  # we only evaluated on the sample
        "n_values": int(RESERVOIR_SIZE * x_orig.shape[1]),
        "reconstruction_mse_full": mse,
        "reconstruction_cosine_full": cos,
        "sample_size": int(RESERVOIR_SIZE),
        "workers": 1,
        "exact_pass_elapsed_s": 0.0,
        "sample_pass_elapsed_s": 0.0,
        "knn_recall_10_cosine_sampled": float(recalls[10]),
        "knn_recall_25_cosine_sampled": float(recalls[25]),
        "knn_recall_50_cosine_sampled": float(recalls[50]),
        "knn_recall_mean_cosine_sampled": float(
            (recalls[10] + recalls[25] + recalls[50]) / 3
        ),
        "_note": (
            "turbo1 evaluated on a 20k reservoir from a 200k pool (in-memory), "
            "not the full 50M corpus streaming pass used for turbo2/3/4/8. "
            "Same cosine kNN recall protocol, same reservoir_size=20000. "
            "Random orthogonal rotation (seed=0) + sign coding + inverse rotation."
        ),
    }

    with open(TURBO_FULL_PATH) as f:
        report = json.load(f)

    existing = {r["method"] for r in report["results"]}
    if "turbo1" in existing:
        report["results"] = [r if r["method"] != "turbo1" else entry for r in report["results"]]
    else:
        # Insert at the front so ordering is turbo1, turbo2, turbo3, turbo4, turbo8
        report["results"].insert(0, entry)
    if "turbo1" not in report.get("methods", []):
        report["methods"] = ["turbo1"] + list(report.get("methods", []))

    TURBO_FULL_PATH.write_text(json.dumps(report, indent=2))
    log.info("Patched → %s", TURBO_FULL_PATH)


if __name__ == "__main__":
    main()
