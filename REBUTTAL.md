# Rebuttal & Action Plan

Verified numbers on disk (2026-04-04):

- `./embeddings/clay-v1-5-sentinel-2/` → **182 GiB** (ZSTD parquet, 13 columns incl. `element`=1024-d float32 embedding, geometry, stac metadata, …)
- `./results/quantized_dataset_full_allmethods_j8/binary/` → **11 GiB** (SNAPPY parquet, same 13-column schema, `element` now 128-byte packed bits)
- Ratio: 182 / 11 = **16.5×** — this matches the blog's "16.5× compression" exactly. The number is **on-disk-to-on-disk** including parquet metadata, geometry, stac fields, row-group overhead, etc. It is *not* the theoretical 4096 B → 128 B = 32× the reviewer computed.
- The web-demo corpus (~7 GiB) is smaller because (a) several columns were dropped and (b) parquet codec was retuned — those optimizations are outside the "binary quantization" claim.

---

## Reviewer 1

### Valid — fix in post

| # | Issue | Response |
|---|---|---|
| 1 | "16.5× compression" vs theoretical 32× | **Keep 16.5×** but add a sentence: *"This is disk-to-disk, including geometry / STAC metadata / parquet overhead. The raw embedding payload alone compresses 32× (4096 B → 128 B); the 16.5× figure is what actually lands on S3."* |
| 2 | "183 GiB float32" vs 49.8M × 1024 × 4 = 190 GiB | The 182 GiB figure is the **actual on-disk ZSTD parquet**, not raw float32. Fix prose: *"182 GiB on disk in ZSTD parquet (≈190 GiB raw float32)"* . |
| 3 | "6.4 GiB" GB/GiB | Change to "≈6 GiB" or "6.4 GB". Accept. |
| 4 | int4 "6× compression" | **Fix.** int4 on 1024-d float32 = 4096/512 = **8× theoretical**. On-disk it lands around ~5–6× after parquet overhead — if the 6× came from measured on-disk ratio, label it "measured on-disk"; otherwise correct to 8×. I'll recompute from `results/quantized_dataset_full_allmethods_j8/int4/` and state which. |
| 5 | "7 GiB in browser memory" caveat | **Fix.** Per-query the browser fetches at most ~50k rows × 128 B ≈ 6.4 MB. Rewrite: *"Per query we cap at 50k rows / ~6 MB of embeddings in memory; the 7 GiB is the corpus on S3, fetched in geohash shards on demand."* |
| 6 | `binary_variants.png` referenced but never shown | Insert the figure; it's in the repo. |
| 7 | Clay transition not signposted | Add one line bridging pt.1 (DINOv3) → pt.2 (Clay v1.5). Accept. |
| 8 | Clay under-introduced | One sentence: "Clay v1.5 is a self-supervised ViT trained on multi-modal Sentinel-{1,2}, DEM, …" Accept. |
| 9 | "terrabit paper" links to repo | Change to "terrabit repo". Accept. |
| 10 | "at lake scale" | Either define "embedding lake" on first use or switch to "at corpus scale". Accept. |
| 11 | Unicode math won't render in Quarto | Wrap in `$...$`. Accept. |
| 12 | Title "aka" awkward | Pick one framing. Accept. |
| 13–17 | Tone tightening | Accept all; trivial edits. |
| 18 | Placeholder section (lines 109–120) | Write it or remove it before publishing. Blocker. |
| 19 | "What you could do with this" speculative | Trim to 2 concrete items, reframe as "directions we're exploring". Accept. |
| 20 | Change-monitoring bullet | Cut or expand with concrete example. Accept. |
| 21 | terrabit vs TerraBit casing | Standardize on **TerraBit** (project/title) + `terrabit` (repo/package). Sweep. |
| 22 | Clay vs Clay v1.5 | Use **Clay v1.5** throughout — version matters for reproducibility. |

### Push back (partially)

- **#1 (16.5×)**: Do not "correct" to 32×. The 16.5× is the *honest* disk-to-disk number users actually pay for on S3. We'll clarify that both numbers exist and why they differ, but the headline stays 16.5×.
- **#2 (182 vs 190)**: Similarly — 182 GiB is measured on disk, not a calculation error. Just clarify the label.

---

## Reviewer 1 — Figure bug (NOT in reviewer comments, surfaced by you)

**`paper/figures/search_benchmark.pdf` shows binary recall@10 = 1.0. This is wrong.**

Root cause: [scripts/run_faiss.py:232-237](scripts/run_faiss.py#L232-L237)

```python
if idx_type == "binary":
    pred_knn = search_binary(cast("Any", idx), queries_bin, max_k)
    ref_knn = gt_bin_knn          # <-- bug: compared to BINARY ground truth
else:
    pred_knn = search_float(cast("Any", idx), queries, max_k)
    ref_knn = gt_knn              # float32 ground truth
```

All non-binary methods compute recall against the **float32** GT (`gt_knn`). Binary uniquely uses `gt_bin_knn` as reference — i.e. it asks "does binary hamming search agree with itself?" The answer is trivially 1.0 because `pred_knn == ref_knn` by construction (same index, same queries).

[paper/results/faiss_results_1M.json](results/faiss_results_1M.json) confirms:
```
binary_hamming   qps=270.0   recall@10=1.0     <-- bogus
int2_flat        qps=47.5    recall@10=0.608
int3_flat        qps=48.3    recall@10=0.815
int4_flat        qps=48.5    recall@10=0.901
fp8_flat         qps=48.6    recall@10=0.950
int8_flat        qps=48.6    recall@10=0.992
float16_flat     qps=59.5    recall@10=0.999
```

**Status: FIXED.**

1. [scripts/run_faiss.py:232-240](scripts/run_faiss.py#L232-L240) patched — `ref_knn = gt_knn` unconditionally. The in-line comment explains why binary is no exception.
2. Rather than rerunning the full 7-method sweep (unchanged), wrote [scripts/patch_faiss_binary_and_pq.py](scripts/patch_faiss_binary_and_pq.py) — a targeted re-run that:
   - Reloads the same 1M sample with the same seed (42).
   - Rebuilds the float32 GT index + computes GT kNN.
   - Re-runs **only** the binary experiment, compared against float32 GT.
   - Patches `binary_hamming` in-place in `results/faiss_results_1M.json`.
3. Regenerated [paper/figures/search_benchmark.pdf](paper/figures/search_benchmark.pdf).
4. `scripts/run_gpu.py` should get the same one-line patch before its next run — NOT re-run now (GPU binary is a separate Hamming kernel; the bar in the figure was also bogus, but rerunning needs an RTX 3090). Flag in the "remaining work" section.

**Measured binary recall (1M sample, 1000 queries, vs float32 GT):**

| k | old (bug) | new |
|---|---|---|
| 1  | 1.000 | 1.000 *(query in corpus)* |
| 5  | 1.000 | 0.660 |
| 10 | 1.000 | **0.651** |
| 25 | 1.000 | 0.653 |
| 50 | 1.000 | 0.664 |

So naive sign-binary at 128 B/vec lands at ~0.65 recall@10 — materially worse than the old figure suggested, and exactly the gap that motivates rotated/TurboQuant binary in the rest of the post. The narrative actually gets *stronger* with the correct number: naive binary is bad, TurboQuant rescues it.

**Blog/paper prose that needs updating** — any line quoting binary recall ~1.0 or claiming naive binary is "good enough" without rotation. The correct headline is: naive sign binary ≈ 0.65 recall@10, and the rotation trick is what makes binary viable.

---

## Reviewer 2

> "product quantization is worse than turboquant but if you have time you should include it as a baseline"

**Status: DONE (one configuration).** Added `pq_128x8` in the same [patch script](scripts/patch_faiss_binary_and_pq.py) and [paper/make_figures.py](paper/make_figures.py):
- `faiss.IndexPQ(d=1024, M=128, nbits=8)` → **128 bytes/vector**, matching the naive-binary footprint exactly so recall is an apples-to-apples footprint comparison.
- Trained on the full 1M sample, same seed, same queries.

**Measured PQ vs. binary at 128 B/vec:**

| method | bytes/vec | QPS (CPU) | recall@10 |
|---|---|---|---|
| `binary_hamming` (sign)      | 128 | 800   | 0.651 |
| **`pq_128x8`** (M=128,nbits=8) | 128 | 115   | **0.782** |

Caveats / what this shows:
- At equal footprint, naive sign-binary is *worse* on recall than PQ (0.65 vs 0.78). So "PQ doesn't work" is too strong — plain PQ actually beats naive sign-binary here.
- The real comparison is **PQ vs. TurboQuant** (rotated binary). The `binary_variants` experiment in the repo already shows rotated binary closing most of the gap to float32; if TurboQuant hits ≳0.85 recall@10 at the same 128 B, the PQ baseline is then strictly dominated.
- On QPS, binary (Hamming popcount) is ~7× faster than PQ (asymmetric distance table lookups), which is the other half of the binary pitch.
- Only one PQ config run — no nbits/M ablation, no OPQ, no IVF wrapper. Per the instruction: one solid baseline, not a thorough ablation. If a reviewer wants more, OPQ and IVF-PQ are the obvious next runs.

> "models need strong location encoders, embeddings need to be highly spatially autocorrelated … 2-step query (download geohashes, filter) → 1-step (just query the data)"

**Good framing, worth a paragraph in the discussion / future work.** The current 2-step pipeline (geohash pre-filter → vector search within shard) *assumes* strong spatial locality of semantics, and breaks exactly when the reviewer describes — as the corpus grows spatiotemporally, near-neighbors in embedding space become geographically scattered, so pre-filtering by geohash throws out true positives. We should:

1. Add a paragraph to the blog's "limitations" / "future work" noting this failure mode for global-scale, high-resolution corpora.
2. Reference it as motivation for **location-aware foundation models** / location encoders (SatCLIP, GeoCLIP, CSP, etc.) as a path from 2-step → 1-step.
3. Not a code change — just framing. Accept.

---

## Action Plan (ordered)

### P0 — correctness blockers
1. **Fix binary recall bug** in [scripts/run_faiss.py:234](scripts/run_faiss.py#L234): use `gt_knn` as reference for binary too. Re-run → regenerate `results/faiss_results_1M.json`, `gpu_results_1M.json`, `paper/figures/search_benchmark.pdf`.
2. **Sweep prose** in blog post + `paper/main.tex` for the old binary recall number and update.

### P1 — factual corrections in blog
3. Add disk-to-disk vs theoretical compression-ratio clarification (16.5× on-disk, 32× raw embedding bytes).
4. Fix 182 GiB / 6 GiB unit labels.
5. Fix int4 ratio: 8× theoretical, confirm on-disk by `du`ing `results/quantized_dataset_full_allmethods_j8/int4/`, label accordingly.
6. Rewrite the "7 GiB in browser" caveat to reflect the ~6 MB per-query cap.
7. Insert missing `binary_variants.png`.
8. Fix Quarto math notation (wrap in `$...$`).

### P2 — editorial
9. Add Clay v1.5 one-line intro + signpost the DINOv3 → Clay transition.
10. Fix "terrabit paper" link text → "terrabit repo".
11. Replace "at lake scale" / define "embedding lake".
12. Standardize TerraBit / Clay v1.5 casing throughout.
13. Tone pass per #12–17 (title, "key design decision", "silently destroy recall", "coming whether we like it or not", etc.).
14. Either write or remove the "Interactive labeling" placeholder (lines 109–120).
15. Trim "What you could do with this" to 2 concrete items; cut or expand change-monitoring bullet.

### P3 — extensions (nice-to-have)
16. Add PQ baseline to FAISS sweep per Reviewer 2.
17. Add a "spatial-semantic drift" paragraph in limitations/future-work per Reviewer 2's location-encoder point.

### Not doing
- Not changing the 16.5× headline to 32×. The on-disk number is the honest one and we'll explain why.
- Not restating the web-demo 7 GiB → 11 GiB full-schema distinction as an error — it's a demo-specific optimization (column drop + codec retune) and will be called out in a footnote instead.
