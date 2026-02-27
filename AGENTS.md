# AGENTS.md

TerraBit: post-hoc quantization study for geospatial embedding lakes. Python package in `src/terrabit/`, tests in `tests/`, paper in `paper/`, slides in `slides/`.

## Repo Structure

```text
src/terrabit/          # Python package (quantization, metrics, ID estimation, IO, pipeline)
scripts/               # Experiment runners (quantize.py, run_experiment.py, run_metrics_*.py)
tests/                 # pytest tests (compression, quantization, metrics)
paper/                 # ICML workshop paper (NeurIPS style, tex-fmt --nowrap)
  main.tex             # Main manuscript
  references.bib       # Bibliography
  make_figures.py      # Generates figures from results JSON → paper/figures/
  figures/             # Generated PDFs (pareto, storage, knn_recall_k, intrinsic_dim)
slides/                # Beamer presentation
embeddings/            # Source fp32 Parquet embeddings (49.8M rows, d=1024, 183 GiB)
  geohash_l2=*/        # 511 Hive-partitioned dirs by level-2 geohash
    year=*/            # Sub-partitioned by year
results/               # Experiment outputs
  quantized_dataset_full_allmethods_j8/   # Quantized Parquet corpora
    {binary,float16,fp8,int2,int3,int4,int8}/  # One dir per method, same partition layout
  report_metrics_batched_full.json        # Full-corpus kNN recall/correlation metrics
  report_quant_10k.json                   # 10k-sample ID estimation + quick metrics
```

## Stack

- **Python >=3.13**, managed by [uv](https://docs.astral.sh/uv/)
- **Linting/formatting**: ruff, ty (type checker), pre-commit
- **Testing**: pytest + pytest-cov
- **LaTeX**: TeX Live (tlmgr), latexmk, tex-fmt (with `--nowrap`)

## Development

```bash
make install   # uv sync --all-extras
make check     # pre-commit run --all-files (ruff, ty, typos, pyproject-fmt, mdformat, etc.)
make test      # pytest --cov=src tests/
make clean     # remove build artifacts
```

## Paper (`paper/`)

ICML workshop paper (NeurIPS style template). Requires TeX Live + tex-fmt.

```bash
cd paper
make install   # tlmgr install required LaTeX packages
make check     # tex-fmt --check --nowrap on .tex files
make build     # latexmk -pdf (produces main.pdf)
make watch     # latexmk -pvc (live rebuild on save)
make clean     # remove build artifacts
python make_figures.py  # regenerate figures from results/ JSON
```

## Slides (`slides/`)

Beamer presentation template.

```bash
cd slides
make install   # tlmgr install required LaTeX packages
make check     # tex-fmt --check --nowrap on .tex files
make build     # latexmk -pdf (produces main.pdf)
make watch     # latexmk -pvc (live rebuild on save)
make clean     # remove build artifacts
```

## Data

- **Source embeddings**: `embeddings/` — Hive-partitioned Parquet (geohash_l2 × year), 49.8M rows, 1024D float32, ~183 GiB.
- **Quantized embeddings**: `results/quantized_dataset_full_allmethods_j8/` — 7 methods (float16, fp8, int8, int4, int3, int2, binary), same partition layout as source.
- **Metrics**: `results/report_metrics_batched_full.json` (full-corpus), `results/report_quant_10k.json` (10k sample with ID estimation).

## CI

GitHub Actions runs `make check` and `make test` on push/PR to `main`. Release to PyPI on version tags (`v*`).
