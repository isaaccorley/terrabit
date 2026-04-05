"""FAISS vector search experiment for quantized embeddings.

Tests all 7 quantization methods (binary → float16) against brute-force
ground truth using FAISS IndexFlatL2 and IndexBinaryFlat.
Also compares FAISS's own ScalarQuantizer compression.

Ordered low→high precision so we can push memory limits.

Usage:
    python scripts/run_faiss.py [--n-samples 1000000] [--n-queries 1000] [--ks 1,5,10,25,50]
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import time
from pathlib import Path
from typing import Any, cast

import faiss
import numpy as np
from load_data import (
    load_quantized_binary_raw,
    load_quantized_dequantized,
    load_source_embeddings,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FAISS index builders
# ---------------------------------------------------------------------------


def build_flat_index(embeddings: np.ndarray) -> Any:
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    cast("Any", index).add(np.ascontiguousarray(embeddings))
    return index


def build_sq_index(embeddings: np.ndarray, qt: int) -> Any:
    d = embeddings.shape[1]
    index = faiss.IndexScalarQuantizer(d, qt)
    x = np.ascontiguousarray(embeddings)
    cast("Any", index).train(x)
    cast("Any", index).add(x)
    return index


def build_binary_index(packed: np.ndarray) -> Any:
    d_bits = packed.shape[1] * 8
    index = faiss.IndexBinaryFlat(d_bits)
    cast("Any", index).add(np.ascontiguousarray(packed))
    return index


# ---------------------------------------------------------------------------
# Search + metrics
# ---------------------------------------------------------------------------


def search_float(index: Any, queries: np.ndarray, k: int) -> np.ndarray:
    _, ids = cast("Any", index).search(np.ascontiguousarray(queries), k)
    return ids


def search_binary(index: Any, queries: np.ndarray, k: int) -> np.ndarray:
    _, ids = cast("Any", index).search(np.ascontiguousarray(queries), k)
    return ids


def recall_at_k(gt: np.ndarray, pred: np.ndarray, k: int) -> float:
    n = gt.shape[0]
    hits = sum(len(set(gt[i, :k].tolist()) & set(pred[i, :k].tolist())) for i in range(n))
    return hits / (n * k)


def _mem_gib() -> str:
    """Current RSS in GiB."""
    import resource

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # KB on Linux
    return f"{rss / 1024 / 1024:.1f} GiB"


# ---------------------------------------------------------------------------
# Experiment definitions — ordered low → high precision
# ---------------------------------------------------------------------------

# (name, quant_method, index_type, faiss_qt)
# Dequantized methods: load our pre-quantized Parquet, dequantize → f32, build Flat index
# SQ methods: FAISS re-quantizes source f32 internally
EXPERIMENTS = [
    # --- Low precision first ---
    ("binary_hamming", "binary", "binary", None),
    ("int2_flat", "int2", "flat", None),
    ("int3_flat", "int3", "flat", None),
    ("int4_flat", "int4", "flat", None),
    ("fp8_flat", "fp8", "flat", None),
    ("int8_flat", "int8", "flat", None),
    ("float16_flat", "float16", "flat", None),
    # --- FAISS ScalarQuantizer baselines ---
    ("source_sq_4bit", None, "sq", faiss.ScalarQuantizer.QT_4bit),
    ("source_sq_8bit", None, "sq", faiss.ScalarQuantizer.QT_8bit),
    ("source_sq_fp16", None, "sq", faiss.ScalarQuantizer.QT_fp16),
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="FAISS vector search experiment")
    parser.add_argument("--n-samples", type=int, default=1_000_000)
    parser.add_argument("--n-queries", type=int, default=1_000)
    parser.add_argument("--ks", type=str, default="1,5,10,25,50")
    parser.add_argument("--output", type=str, default="results/faiss_results.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ks = [int(x) for x in args.ks.split(",")]
    max_k = max(ks)

    # --- Source embeddings ---
    log.info("Loading source embeddings (n=%d)... [RSS %s]", args.n_samples, _mem_gib())
    t0 = time.perf_counter()
    source_embs, source_files, chosen_idx = load_source_embeddings(
        n_samples=args.n_samples, seed=args.seed
    )
    log.info(
        "Loaded %d source embeddings (%d-d) from %d files in %.1fs [RSS %s]",
        *source_embs.shape,
        len(source_files),
        time.perf_counter() - t0,
        _mem_gib(),
    )

    # --- Ground-truth brute-force index ---
    log.info("Building brute-force GT index...")
    t0 = time.perf_counter()
    gt_index = build_flat_index(source_embs)
    log.info("GT index: %d vectors in %.1fs", gt_index.ntotal, time.perf_counter() - t0)

    # --- Query vectors ---
    rng = np.random.default_rng(args.seed + 1)
    n_q = min(args.n_queries, len(source_embs))
    q_idx = rng.choice(len(source_embs), size=n_q, replace=False)
    queries = source_embs[q_idx]

    log.info("Computing GT kNN (k=%d, n_queries=%d)...", max_k, n_q)
    t0 = time.perf_counter()
    gt_knn = search_float(gt_index, queries, max_k)
    log.info("GT search: %.1fs", time.perf_counter() - t0)

    # Precompute binary GT
    from terrabit.quantization import quantize_binary

    queries_bin = quantize_binary(queries)
    source_bin = quantize_binary(source_embs)
    gt_bin_index = build_binary_index(source_bin)
    search_binary(gt_bin_index, queries_bin, max_k)
    del source_bin, gt_bin_index  # free ~n*128 bytes
    gc.collect()

    results: dict = {
        "n_samples": len(source_embs),
        "n_queries": n_q,
        "dim": int(source_embs.shape[1]),
        "ks": ks,
        "experiments": {},
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    # --- Run experiments low → high ---
    for exp_i, (name, qmethod, idx_type, qt) in enumerate(EXPERIMENTS):
        log.info(
            "\n======== [%d/%d] %s ======== [RSS %s]",
            exp_i + 1,
            len(EXPERIMENTS),
            name,
            _mem_gib(),
        )
        t_exp = time.perf_counter()

        # Load data
        t0 = time.perf_counter()
        data_f32 = None
        data_bin = None
        if qmethod is not None:
            if idx_type == "binary":
                data_bin = load_quantized_binary_raw(source_files, chosen_idx)
            else:
                data_f32 = load_quantized_dequantized(qmethod, source_files, chosen_idx)
        else:
            data_f32 = source_embs
        t_load = time.perf_counter() - t0
        log.info("  Loaded in %.1fs [RSS %s]", t_load, _mem_gib())

        # Build index
        t0 = time.perf_counter()
        if idx_type == "flat":
            assert data_f32 is not None
            idx = build_flat_index(data_f32)
        elif idx_type == "sq":
            assert data_f32 is not None
            assert qt is not None
            idx = build_sq_index(data_f32, qt)
        elif idx_type == "binary":
            assert data_bin is not None
            idx = build_binary_index(data_bin)
        else:
            msg = f"Unknown index type: {idx_type}"
            raise ValueError(msg)
        t_index = time.perf_counter() - t0
        log.info("  Index built in %.1fs", t_index)

        # Free loaded data (index has its own copy)
        del data_f32, data_bin
        gc.collect()

        # Search
        t0 = time.perf_counter()
        if idx_type == "binary":
            pred_knn = search_binary(cast("Any", idx), queries_bin, max_k)
        else:
            pred_knn = search_float(cast("Any", idx), queries, max_k)
        # Recall is always measured against the float32 ground-truth neighbor list —
        # for binary this answers "how well does Hamming search recover the true f32 kNN?",
        # not "does binary search agree with itself" (which is trivially 1.0).
        ref_knn = gt_knn
        t_search = time.perf_counter() - t0
        qps = n_q / t_search if t_search > 0 else float("inf")
        log.info("  Search: %.1fs (%.0f QPS)", t_search, qps)

        # Free index
        del idx
        gc.collect()

        # Recall
        exp: dict = {
            "quant_method": qmethod,
            "index_type": idx_type,
            "faiss_qt": str(qt) if qt is not None else None,
            "load_time_s": round(t_load, 3),
            "index_time_s": round(t_index, 3),
            "search_time_s": round(t_search, 3),
            "qps": round(qps, 1),
            "recall": {},
        }
        for k in ks:
            r = recall_at_k(ref_knn, pred_knn, k)
            exp["recall"][f"recall@{k}"] = round(r, 6)
            log.info("  recall@%d = %.4f", k, r)

        results["experiments"][name] = exp
        log.info("  total: %.1fs [RSS %s]", time.perf_counter() - t_exp, _mem_gib())

        # Incremental save after each experiment
        out.write_text(json.dumps(results, indent=2))

    log.info("\nAll done. Results → %s", out)


if __name__ == "__main__":
    main()
