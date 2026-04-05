"""Patch `results/faiss_results_1M.json` with two corrections.

1. `binary_hamming` recall is re-measured against the **float32** ground-truth
   neighbor list (the original run used the binary GT as reference, which made
   recall trivially 1.0 — a fairness bug in `scripts/run_faiss.py`).

2. Adds a single **PQ baseline** — `pq_128x8` — matching the 128-byte/vector
   footprint of binary. Trained and searched on the same 1M sample / 1000
   queries, compared against the same float32 GT as all other methods.

Usage:
    python scripts/patch_faiss_binary_and_pq.py
"""

from __future__ import annotations

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

# Pin to the Sentinel-2 Clay dataset — the NAIP dir in embeddings/ has a
# different column name and would break the walker.
load_data.EMBEDDING_DIR = "embeddings/clay-v1-5-sentinel-2"
from load_data import load_quantized_binary_raw, load_source_embeddings  # noqa: E402

from terrabit.quantization import quantize_binary  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

RESULTS_PATH = Path(__file__).resolve().parent.parent / "results" / "faiss_results_1M.json"


def recall_at_k(gt: np.ndarray, pred: np.ndarray, k: int) -> float:
    n = gt.shape[0]
    hits = sum(len(set(gt[i, :k].tolist()) & set(pred[i, :k].tolist())) for i in range(n))
    return hits / (n * k)


def main() -> None:
    with open(RESULTS_PATH) as f:
        results = json.load(f)

    n_samples = results["n_samples"]
    n_queries = results["n_queries"]
    ks = results["ks"]
    max_k = max(ks)
    seed = 42  # matches run_faiss.py default

    log.info("Loading source embeddings (n=%d, seed=%d)...", n_samples, seed)
    t0 = time.perf_counter()
    source_embs, source_files, chosen_idx = load_source_embeddings(n_samples=n_samples, seed=seed)
    log.info("Loaded %d × %d in %.1fs", *source_embs.shape, time.perf_counter() - t0)

    # Float32 GT (same seed-derived queries as run_faiss.py)
    rng = np.random.default_rng(seed + 1)
    n_q = min(n_queries, len(source_embs))
    q_idx = rng.choice(len(source_embs), size=n_q, replace=False)
    queries = source_embs[q_idx]

    log.info("Building float32 ground-truth index...")
    gt_index = faiss.IndexFlatL2(source_embs.shape[1])
    cast("Any", gt_index).add(np.ascontiguousarray(source_embs))
    log.info("Computing float32 GT kNN (k=%d)...", max_k)
    t0 = time.perf_counter()
    _, gt_knn = cast("Any", gt_index).search(np.ascontiguousarray(queries), max_k)
    log.info("GT search: %.1fs", time.perf_counter() - t0)
    del gt_index
    gc.collect()

    # ------------------------------------------------------------------
    # 1. Binary — re-measure recall against float32 GT
    # ------------------------------------------------------------------
    log.info("\n======== [1/2] binary_hamming (recall vs float32 GT) ========")
    queries_bin = quantize_binary(queries)
    source_bin = load_quantized_binary_raw(source_files, chosen_idx)
    d_bits = source_bin.shape[1] * 8

    t0 = time.perf_counter()
    bin_index = faiss.IndexBinaryFlat(d_bits)
    cast("Any", bin_index).add(np.ascontiguousarray(source_bin))
    t_index = time.perf_counter() - t0
    log.info("  Index built in %.1fs", t_index)
    del source_bin
    gc.collect()

    t0 = time.perf_counter()
    _, pred_knn = cast("Any", bin_index).search(np.ascontiguousarray(queries_bin), max_k)
    t_search = time.perf_counter() - t0
    qps = n_q / t_search if t_search > 0 else float("inf")
    log.info("  Search: %.2fs (%.0f QPS)", t_search, qps)
    del bin_index
    gc.collect()

    bin_exp = results["experiments"]["binary_hamming"]
    bin_exp["search_time_s"] = round(t_search, 3)
    bin_exp["qps"] = round(qps, 1)
    bin_exp["index_time_s"] = round(t_index, 3)
    bin_exp["recall"] = {}
    for k in ks:
        r = recall_at_k(gt_knn, pred_knn, k)
        bin_exp["recall"][f"recall@{k}"] = round(r, 6)
        log.info("  recall@%d = %.4f", k, r)
    bin_exp["_note"] = "Recall re-computed vs float32 GT (was vs binary GT in original run)."

    # ------------------------------------------------------------------
    # 2. PQ baseline — 128 bytes/vector to match binary footprint
    #    M=128 subquantizers × nbits=8 → 1 byte/subvector → 128 B total
    #    (1024 dim / 128 M = 8-d subvectors)
    # ------------------------------------------------------------------
    log.info("\n======== [2/2] pq_128x8 (M=128, nbits=8 → 128 B/vec) ========")
    M, nbits = 128, 8
    d = source_embs.shape[1]
    assert d % M == 0, f"dim {d} not divisible by M={M}"

    t0 = time.perf_counter()
    pq_index = faiss.IndexPQ(d, M, nbits)
    log.info("  Training PQ on %d vectors...", len(source_embs))
    cast("Any", pq_index).train(np.ascontiguousarray(source_embs))
    t_train = time.perf_counter() - t0
    log.info("  Trained in %.1fs", t_train)

    t0 = time.perf_counter()
    cast("Any", pq_index).add(np.ascontiguousarray(source_embs))
    t_index = time.perf_counter() - t0 + t_train
    log.info("  Added in %.1fs (index total %.1fs)", t_index - t_train, t_index)

    t0 = time.perf_counter()
    _, pq_pred = cast("Any", pq_index).search(np.ascontiguousarray(queries), max_k)
    t_search = time.perf_counter() - t0
    qps = n_q / t_search if t_search > 0 else float("inf")
    log.info("  Search: %.2fs (%.0f QPS)", t_search, qps)

    pq_exp: dict = {
        "quant_method": "pq",
        "index_type": "pq",
        "faiss_qt": None,
        "pq_M": M,
        "pq_nbits": nbits,
        "bytes_per_vector": M * nbits // 8,
        "load_time_s": 0.0,
        "index_time_s": round(t_index, 3),
        "search_time_s": round(t_search, 3),
        "qps": round(qps, 1),
        "recall": {},
    }
    for k in ks:
        r = recall_at_k(gt_knn, pq_pred, k)
        pq_exp["recall"][f"recall@{k}"] = round(r, 6)
        log.info("  recall@%d = %.4f", k, r)
    results["experiments"]["pq_128x8"] = pq_exp

    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    log.info("\nPatched → %s", RESULTS_PATH)


if __name__ == "__main__":
    main()
