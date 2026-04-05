"""Turbo1 benchmark — random orthogonal rotation + 1-bit sign coding.

Apples-to-apples with naive `binary_hamming`: identical 128 B/vec footprint,
identical Hamming-distance search path (FAISS IndexBinaryFlat), identical
1M sample / 1000 queries / seed. The only difference is that embeddings and
queries are first multiplied by a random orthogonal matrix before sign
packing. This isolates the "does rotation help at 1 bit?" question.

Why it's interesting: the existing `turbo2..turbo8` entries use 2+ bits per
dimension, so they don't answer the 1-bit question. The existing `binary_itq`
entry uses a *learned* PCA+rotation and under-performs naive binary on Clay —
turbo1 uses a *random* Haar-distributed rotation (free, no training).

Patches `results/faiss_results_1M.json` with a `turbo1_hamming` entry so
`paper/make_figures.py` can include it in the search-benchmark bar chart.

Usage:
    python scripts/run_turbo1_bench.py [--seed 42]
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, cast

import faiss
import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

import load_data  # noqa: E402

load_data.EMBEDDING_DIR = "embeddings/clay-v1-5-sentinel-2"
from load_data import load_source_embeddings  # noqa: E402

from terrabit.quantization import _random_orthogonal_matrix  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

RESULTS_PATH = Path(__file__).resolve().parent.parent / "results" / "faiss_results_1M.json"


def recall_at_k(gt: np.ndarray, pred: np.ndarray, k: int) -> float:
    n = gt.shape[0]
    hits = sum(len(set(gt[i, :k].tolist()) & set(pred[i, :k].tolist())) for i in range(n))
    return hits / (n * k)


def sign_pack(x: np.ndarray) -> np.ndarray:
    """Sign-bit quantize (> 0) + packbits → (n, d/8) uint8."""
    signs = (x > 0).astype(np.uint8)
    _, d = signs.shape
    pad = (8 - d % 8) % 8
    if pad > 0:
        signs = np.pad(signs, ((0, 0), (0, pad)))
    return np.packbits(signs, axis=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rot-seed", type=int, default=0)
    args = parser.parse_args()

    with open(RESULTS_PATH) as f:
        results = json.load(f)

    n_samples = results["n_samples"]
    n_queries = results["n_queries"]
    ks = results["ks"]
    max_k = max(ks)

    log.info("Loading source embeddings (n=%d, seed=%d)...", n_samples, args.seed)
    t0 = time.perf_counter()
    source_embs, _, _ = load_source_embeddings(n_samples=n_samples, seed=args.seed)
    log.info("Loaded %d × %d in %.1fs", *source_embs.shape, time.perf_counter() - t0)

    d = source_embs.shape[1]

    # --- Float32 GT (same queries as patch_faiss_binary_and_pq.py) ---
    rng = np.random.default_rng(args.seed + 1)
    n_q = min(n_queries, len(source_embs))
    q_idx = rng.choice(len(source_embs), size=n_q, replace=False)
    queries = source_embs[q_idx]

    log.info("Building float32 GT index + computing GT kNN...")
    gt_index = faiss.IndexFlatL2(d)
    cast("Any", gt_index).add(np.ascontiguousarray(source_embs))
    t0 = time.perf_counter()
    _, gt_knn = cast("Any", gt_index).search(np.ascontiguousarray(queries), max_k)
    log.info("GT search: %.1fs", time.perf_counter() - t0)
    del gt_index
    gc.collect()

    # --- Turbo1: random rotation + sign ---
    log.info("Drawing %d×%d Haar orthogonal matrix (rot_seed=%d)...", d, d, args.rot_seed)
    t0 = time.perf_counter()
    r_matrix = _random_orthogonal_matrix(d, seed=args.rot_seed)
    log.info("Rotation ready in %.1fs", time.perf_counter() - t0)

    log.info("Rotating corpus (%d × %d)...", *source_embs.shape)
    t0 = time.perf_counter()
    # Batched matmul to bound peak memory (1M × 1024 × 4B = 4 GiB per copy).
    batch = 100_000
    rotated = np.empty_like(source_embs)
    for i in range(0, len(source_embs), batch):
        rotated[i : i + batch] = source_embs[i : i + batch] @ r_matrix
    t_rotate = time.perf_counter() - t0
    log.info("Corpus rotated in %.1fs", t_rotate)

    log.info("Sign-packing corpus...")
    t0 = time.perf_counter()
    corpus_bits = sign_pack(rotated)
    t_pack = time.perf_counter() - t0
    log.info("Packed → %s (%d B/vec) in %.1fs", corpus_bits.shape, corpus_bits.shape[1], t_pack)
    del rotated
    gc.collect()

    # Queries: rotate + sign-pack
    queries_rot = queries @ r_matrix
    queries_bits = sign_pack(queries_rot)

    # Build binary index + search
    t0 = time.perf_counter()
    bin_index = faiss.IndexBinaryFlat(corpus_bits.shape[1] * 8)
    cast("Any", bin_index).add(np.ascontiguousarray(corpus_bits))
    t_index = time.perf_counter() - t0
    log.info("Binary index built in %.1fs", t_index)

    t0 = time.perf_counter()
    _, pred_knn = cast("Any", bin_index).search(np.ascontiguousarray(queries_bits), max_k)
    t_search = time.perf_counter() - t0
    qps = n_q / t_search if t_search > 0 else float("inf")
    log.info("Hamming search: %.2fs (%.0f QPS)", t_search, qps)

    exp: dict = {
        "quant_method": "turbo1",
        "index_type": "binary",
        "faiss_qt": None,
        "rotation_seed": args.rot_seed,
        "bytes_per_vector": corpus_bits.shape[1],
        "load_time_s": 0.0,
        "rotate_time_s": round(t_rotate, 3),
        "index_time_s": round(t_index, 3),
        "search_time_s": round(t_search, 3),
        "qps": round(qps, 1),
        "recall": {},
        "_note": (
            "Random orthogonal rotation + sign coding (Hamming search). "
            "Same 128 B/vec footprint as binary_hamming; isolates the "
            "'does rotation help at 1 bit?' question."
        ),
    }
    for k in ks:
        r = recall_at_k(gt_knn, pred_knn, k)
        exp["recall"][f"recall@{k}"] = round(r, 6)
        log.info("  recall@%d = %.4f", k, r)

    results["experiments"]["turbo1_hamming"] = exp
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    log.info("Patched → %s", RESULTS_PATH)


if __name__ == "__main__":
    main()
