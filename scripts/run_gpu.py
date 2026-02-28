"""PyTorch GPU brute-force kNN benchmark for quantized embeddings.

Uses torch.cdist on CUDA for exact kNN, matching the same protocol as
run_faiss.py but leveraging the RTX 3090 GPU.

Usage:
    python scripts/run_gpu.py [--n-samples 1000000] [--n-queries 1000]
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from load_data import (
    load_quantized_binary_raw,
    load_quantized_dequantized,
    load_source_embeddings,
)

from terrabit.quantization import quantize_binary

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gpu_knn(
    queries: np.ndarray,
    corpus: np.ndarray,
    k: int,
    batch_size: int = 256,
) -> np.ndarray:
    """Brute-force kNN via torch.cdist on GPU, batched over queries."""
    corpus_t = torch.from_numpy(corpus).to(DEVICE, dtype=torch.float32)
    n_q = len(queries)
    all_ids = np.empty((n_q, k), dtype=np.int64)

    for start in range(0, n_q, batch_size):
        end = min(start + batch_size, n_q)
        q_batch = torch.from_numpy(queries[start:end]).to(DEVICE, dtype=torch.float32)
        dists = torch.cdist(q_batch, corpus_t)  # (batch, n_corpus)
        _, topk_ids = dists.topk(k, largest=False)
        all_ids[start:end] = topk_ids.cpu().numpy()
        del q_batch, dists, topk_ids
    del corpus_t
    torch.cuda.empty_cache()
    return all_ids


def hamming_knn_gpu(
    queries_packed: np.ndarray,
    corpus_packed: np.ndarray,
    k: int,
    batch_size: int = 256,
) -> np.ndarray:
    """Hamming-distance kNN on GPU using bit-unpacking + L2 on {0,1} vectors."""
    # Unpack bits: uint8 → {0,1} float16 for memory efficiency
    corpus_bits = np.unpackbits(corpus_packed, axis=1).astype(np.float16)
    corpus_t = torch.from_numpy(corpus_bits).to(DEVICE, dtype=torch.float16)
    del corpus_bits

    n_q = len(queries_packed)
    all_ids = np.empty((n_q, k), dtype=np.int64)

    for start in range(0, n_q, batch_size):
        end = min(start + batch_size, n_q)
        q_bits = np.unpackbits(queries_packed[start:end], axis=1).astype(np.float16)
        q_t = torch.from_numpy(q_bits).to(DEVICE, dtype=torch.float16)
        dists = torch.cdist(q_t, corpus_t)
        _, topk_ids = dists.topk(k, largest=False)
        all_ids[start:end] = topk_ids.cpu().numpy()
        del q_t, dists, topk_ids
    del corpus_t
    torch.cuda.empty_cache()
    return all_ids


def recall_at_k(gt: np.ndarray, pred: np.ndarray, k: int) -> float:
    n = gt.shape[0]
    hits = sum(len(set(gt[i, :k].tolist()) & set(pred[i, :k].tolist())) for i in range(n))
    return hits / (n * k)


QUANT_METHODS = [
    ("binary", "binary"),
    ("int2", "flat"),
    ("int3", "flat"),
    ("int4", "flat"),
    ("fp8", "flat"),
    ("int8", "flat"),
    ("float16", "flat"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="PyTorch GPU kNN benchmark")
    parser.add_argument("--n-samples", type=int, default=1_000_000)
    parser.add_argument("--n-queries", type=int, default=1_000)
    parser.add_argument("--ks", type=str, default="1,5,10,25,50")
    parser.add_argument("--output", type=str, default="results/gpu_results_1M.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ks = [int(x) for x in args.ks.split(",")]
    max_k = max(ks)

    log.info(
        "Device: %s (%s)", DEVICE, torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "CPU"
    )

    # --- Source embeddings ---
    log.info("Loading source embeddings (n=%d)...", args.n_samples)
    t0 = time.perf_counter()
    source_embs, source_files, chosen_idx = load_source_embeddings(
        n_samples=args.n_samples, seed=args.seed
    )
    log.info(
        "Loaded %d source embeddings (%d-d) in %.1fs", *source_embs.shape, time.perf_counter() - t0
    )

    # --- Query vectors ---
    rng = np.random.default_rng(args.seed + 1)
    n_q = min(args.n_queries, len(source_embs))
    q_idx = rng.choice(len(source_embs), size=n_q, replace=False)
    queries = source_embs[q_idx]

    # --- Ground-truth kNN (GPU) ---
    log.info("Computing GT kNN on GPU (k=%d, n_queries=%d)...", max_k, n_q)
    t0 = time.perf_counter()
    gt_knn = gpu_knn(queries, source_embs, max_k)
    log.info("GT search: %.1fs", time.perf_counter() - t0)

    # Binary queries + GT
    queries_bin = quantize_binary(queries)
    source_bin = quantize_binary(source_embs)
    log.info("Computing binary GT kNN on GPU...")
    t0 = time.perf_counter()
    gt_bin_knn = hamming_knn_gpu(queries_bin, source_bin, max_k)
    log.info("Binary GT: %.1fs", time.perf_counter() - t0)
    del source_bin
    gc.collect()
    torch.cuda.empty_cache()

    results: dict = {
        "n_samples": len(source_embs),
        "n_queries": n_q,
        "dim": int(source_embs.shape[1]),
        "ks": ks,
        "device": torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "cpu",
        "experiments": {},
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    for exp_i, (qmethod, idx_type) in enumerate(QUANT_METHODS):
        name = f"{qmethod}_gpu"
        log.info("\n======== [%d/%d] %s ========", exp_i + 1, len(QUANT_METHODS), name)

        # Load data
        t0 = time.perf_counter()
        if idx_type == "binary":
            data_bin = load_quantized_binary_raw(source_files, chosen_idx)
            data_f32 = None
        else:
            data_f32 = load_quantized_dequantized(qmethod, source_files, chosen_idx)
            data_bin = None
        t_load = time.perf_counter() - t0
        log.info("  Loaded in %.1fs", t_load)

        # Search on GPU
        t0 = time.perf_counter()
        if idx_type == "binary" and data_bin is not None:
            pred_knn = hamming_knn_gpu(queries_bin, data_bin, max_k)
            ref_knn = gt_bin_knn
        elif data_f32 is not None:
            pred_knn = gpu_knn(queries, data_f32, max_k)
            ref_knn = gt_knn
        else:
            log.error("  No data loaded for %s", name)
            continue
        t_search = time.perf_counter() - t0
        qps = n_q / t_search if t_search > 0 else float("inf")
        log.info("  Search: %.3fs (%.0f QPS)", t_search, qps)

        # Free data
        del data_f32, data_bin
        gc.collect()
        torch.cuda.empty_cache()

        # Recall
        exp: dict = {
            "quant_method": qmethod,
            "index_type": idx_type,
            "load_time_s": round(t_load, 3),
            "search_time_s": round(t_search, 3),
            "qps": round(qps, 1),
            "recall": {},
        }
        for k in ks:
            r = recall_at_k(ref_knn, pred_knn, k)
            exp["recall"][f"recall@{k}"] = round(r, 6)
            log.info("  recall@%d = %.4f", k, r)

        results["experiments"][name] = exp

        # Incremental save
        out.write_text(json.dumps(results, indent=2))

    log.info("\nAll done. Results → %s", out)


if __name__ == "__main__":
    main()
