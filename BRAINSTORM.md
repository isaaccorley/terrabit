# Intrinsic-Dimension–Aware Compression of High-Dimensional Embeddings

## Abstract

Modern foundation models routinely emit embeddings with hundreds or thousands of dimensions (e.g., 1024D), imposing non-trivial costs on storage, memory bandwidth, indexing, and downstream compute. Empirically, many such embeddings lie on low-dimensional manifolds, suggesting substantial redundancy.
This project investigates **intrinsic dimension (ID)** as a principled tool for (1) estimating theoretical compressibility limits, (2) guiding the choice of compression methods and target dimensionality, and (3) diagnosing information loss induced by compression independent of downstream task labels.
The goal is not to replace task-based evaluation, but to **reduce the search space** of compression strategies and provide model-agnostic diagnostics for representation quality under dimensionality reduction.

______________________________________________________________________

## Core Questions

- What is the intrinsic dimensionality of the original embedding space?
- How much dimensionality reduction is theoretically possible without collapsing the manifold?
- Which compression methods preserve intrinsic structure most efficiently?
- How does intrinsic-dimension preservation correlate with downstream utility?
- When does variance-based compression (e.g., PCA) fail relative to manifold-aware methods?

______________________________________________________________________

## Project Checklist / To-Do

### 1. Dataset & Embedding Characterization

- [ ] Identify embedding source (model, layer, normalization).
- [ ] Verify embedding distribution (mean, variance, anisotropy).
- [ ] Subsample embeddings for scalable analysis (e.g., 5k–20k points).
- [ ] Apply consistent preprocessing (L2 norm, centering, whitening variants).

______________________________________________________________________

### 2. Baseline Intrinsic Dimension Estimation

- [ ] Measure intrinsic dimension on original embeddings using:
    - [ ] kNN-based MLE (Levina–Bickel)
    - [ ] Correlation dimension (optional sanity check)
    - [ ] PCA participation ratio (effective rank)
- [ ] Sweep k (e.g., k = 10–30) and report stability.
- [ ] Report mean ± std across random subsamples.

**Outcome:** Estimated intrinsic dimension range (lower bound on viable compression).

______________________________________________________________________

### 3. Compression Method Inventory

Evaluate multiple inductive biases rather than a single reducer.

- [ ] PCA (variance-based linear baseline)
- [ ] PCA + whitening
- [ ] Random projection (JL baseline)
- [ ] Autoencoder (nonlinear bottleneck)
- [ ] Product quantization / OPQ (if retrieval-oriented)
- [ ] Hybrid approaches (e.g., PCA → AE)

______________________________________________________________________

### 4. Dimensionality Sweep

For each method, evaluate multiple target dimensions:

- [ ] Below ID (intentional under-compression)
- [ ] ≈ ID
- [ ] 2× ID
- [ ] 4× ID
- [ ] Conventional baselines (e.g., 64, 128)

______________________________________________________________________

### 5. Post-Compression Intrinsic Dimension Analysis

This is the critical diagnostic step.

- [ ] Measure intrinsic dimension **after** compression.
- [ ] Compare:
    - Original ID vs. compressed ID
    - ID collapse vs. dimension reduction ratio
- [ ] Identify regimes where:
    - ID is preserved
    - ID is partially collapsed
    - ID is catastrophically reduced

**Outcome:** Compression methods ranked by structural preservation efficiency.

______________________________________________________________________

### 6. Lightweight Downstream Proxies

Use minimal task signal to validate relevance.

- [ ] Linear probe accuracy (if labels exist)
- [ ] kNN retrieval recall / precision
- [ ] Cluster separability metrics
- [ ] Stability across random seeds

**Note:** These are *validation checks*, not optimization targets.

______________________________________________________________________

### 7. Correlation Analysis

- [ ] Correlate intrinsic-dimension preservation with proxy performance.
- [ ] Identify failure modes:
    - Preserved ID but degraded task performance
    - High variance retention but low ID preservation
- [ ] Analyze curvature and anisotropy effects.

______________________________________________________________________

### 8. Cost–Performance Tradeoff Analysis

- [ ] Memory footprint vs. ID preserved
- [ ] Compression time vs. inference-time savings
- [ ] Retrieval/indexing speedups (if applicable)

______________________________________________________________________

## Expected Outcomes

- A principled lower bound on embedding dimensionality.
- Evidence that intrinsic dimension is a **necessary but insufficient** condition for task preservation.
- Identification of compression regimes where PCA is suboptimal.
- A reusable evaluation framework for future embedding models.

______________________________________________________________________

## Key Takeaway

Intrinsic dimension should be treated as a **structural diagnostic and pruning tool**, not a performance oracle. Used correctly, it dramatically narrows the compression design space and exposes failure modes that variance-based metrics miss.

______________________________________________________________________

## Future Extensions

- Layer-wise ID analysis in deep models
- Spatial / conditional intrinsic dimension (per class or region)
- ID-aware training objectives
- Integration with learned quantization schemes
