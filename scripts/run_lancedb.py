"""LanceDB vector search experiment for quantized embeddings.

Loads source float32 and quantized (float16, int8, binary) embeddings,
dequantizes them to float32, indexes them in LanceDB, and measures
kNN recall against the source ground truth at varying k values.

Usage:
    python scripts/run_lancedb.py [--n-samples 10000] [--n-queries 1000] [--ks 10,25,50]
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import lancedb
import numpy as np
import pyarrow as pa
from load_data import load_quantized_dequantized, load_source_embeddings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Ordered low → high precision
METHODS = ["binary", "int2", "int3", "int4", "fp8", "int8", "float16"]


# ---------------------------------------------------------------------------
# LanceDB helpers
# ---------------------------------------------------------------------------


def create_table(
    db: lancedb.DBConnection, name: str, embeddings: np.ndarray
) -> lancedb.table.Table:
    n, d = embeddings.shape
    tbl = pa.table(
        {
            "id": pa.array(np.arange(n), type=pa.int64()),
            "vector": pa.FixedSizeListArray.from_arrays(
                pa.array(embeddings.flatten(), type=pa.float32()),
                d,
            ),
        }
    )
    return db.create_table(name, tbl, mode="overwrite")


def knn_search(table: lancedb.table.Table, queries: np.ndarray, k: int) -> np.ndarray:
    """Brute-force kNN via LanceDB. Returns (n_queries, k) ID array."""
    results = np.empty((len(queries), k), dtype=np.int64)
    for i, q in enumerate(queries):
        res = table.search(q.tolist()).limit(k).to_pandas()
        ids = res["id"].values[:k]
        if len(ids) < k:
            ids = np.pad(ids, (0, k - len(ids)), constant_values=-1)
        results[i] = ids
    return results


def recall_at_k(gt: np.ndarray, pred: np.ndarray, k: int) -> float:
    n = gt.shape[0]
    hits = sum(len(set(gt[i, :k].tolist()) & set(pred[i, :k].tolist())) for i in range(n))
    return hits / (n * k)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="LanceDB vector search experiment")
    parser.add_argument("--n-samples", type=int, default=1_000_000)
    parser.add_argument("--n-queries", type=int, default=1_000)
    parser.add_argument("--ks", type=str, default="1,5,10,25,50")
    parser.add_argument("--output", type=str, default="results/lancedb_results.json")
    parser.add_argument("--db-path", type=str, default="/tmp/terrabit_lancedb")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ks = [int(x) for x in args.ks.split(",")]
    max_k = max(ks)

    # --- Source embeddings ---
    log.info("Loading source embeddings (n=%d)...", args.n_samples)
    t0 = time.perf_counter()
    source_embs, source_files, chosen_idx = load_source_embeddings(
        n_samples=args.n_samples, seed=args.seed
    )
    log.info(
        "Loaded %d source embeddings (%d-d) from %d files in %.1fs",
        *source_embs.shape,
        len(source_files),
        time.perf_counter() - t0,
    )

    # --- LanceDB ground-truth table ---
    db = lancedb.connect(args.db_path)

    log.info("Building GT table...")
    t0 = time.perf_counter()
    gt_table = create_table(db, "source_f32", source_embs)
    log.info("GT table: %d rows in %.1fs", len(source_embs), time.perf_counter() - t0)

    # --- Query vectors ---
    rng = np.random.default_rng(args.seed + 1)
    n_q = min(args.n_queries, len(source_embs))
    q_idx = rng.choice(len(source_embs), size=n_q, replace=False)
    queries = source_embs[q_idx]

    log.info("Computing GT kNN (k=%d, n_queries=%d)...", max_k, n_q)
    t0 = time.perf_counter()
    gt_knn = knn_search(gt_table, queries, max_k)
    log.info("GT search: %.1fs", time.perf_counter() - t0)

    results: dict = {"n_samples": len(source_embs), "n_queries": n_q, "ks": ks, "methods": {}}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    # --- Per-method experiments (low → high precision) ---
    for mi, method in enumerate(METHODS):
        log.info("\n======== [%d/%d] %s ========", mi + 1, len(METHODS), method)
        t_exp = time.perf_counter()

        # Load + dequantize
        t0 = time.perf_counter()
        deq = load_quantized_dequantized(method, source_files, chosen_idx)
        t_load = time.perf_counter() - t0
        log.info("  Loaded %d vectors in %.1fs", len(deq), t_load)

        # Build table
        t0 = time.perf_counter()
        tbl = create_table(db, f"quant_{method}", deq)
        t_index = time.perf_counter() - t0
        log.info("  Table built in %.1fs", t_index)

        del deq  # free dequantized array
        import gc

        gc.collect()

        # Search
        t0 = time.perf_counter()
        pred_knn = knn_search(tbl, queries, max_k)
        t_search = time.perf_counter() - t0
        qps = n_q / t_search if t_search > 0 else float("inf")
        log.info("  Search: %.1fs (%.0f QPS)", t_search, qps)

        # Recall
        mrec: dict = {
            "load_time_s": round(t_load, 3),
            "index_time_s": round(t_index, 3),
            "search_time_s": round(t_search, 3),
            "qps": round(qps, 1),
            "recall": {},
        }
        for k in ks:
            r = recall_at_k(gt_knn, pred_knn, k)
            mrec["recall"][f"recall@{k}"] = round(r, 6)
            log.info("  recall@%d = %.4f", k, r)

        results["methods"][method] = mrec
        log.info("  total: %.1fs", time.perf_counter() - t_exp)

        # Incremental save
        out.write_text(json.dumps(results, indent=2))

    log.info("\nAll done. Results → %s", out)


if __name__ == "__main__":
    main()
