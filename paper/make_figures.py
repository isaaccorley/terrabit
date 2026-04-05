"""Generate paper figures from results JSON."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ---------- load data ----------
with open(RESULTS_DIR / "report_metrics_batched_full.json") as f:
    full = json.load(f)

with open(RESULTS_DIR / "report_quant_10k.json") as f:
    quant10k = json.load(f)

binary_baselines_path = RESULTS_DIR / "report_binary_baselines_20k.json"
binary_baselines = None
if binary_baselines_path.exists():
    with open(binary_baselines_path) as f:
        binary_baselines = json.load(f)

binary_full_path = RESULTS_DIR / "report_metrics_batched_binary_variants_full.json"
binary_full = None
if binary_full_path.exists():
    with open(binary_full_path) as f:
        binary_full = json.load(f)

with open(RESULTS_DIR / "report_metrics_batched_turbo_full.json") as f:
    turbo_full = json.load(f)

storage_bench_path = RESULTS_DIR / "report_storage_100k.json"
storage_bench = None
if storage_bench_path.exists():
    with open(storage_bench_path) as f:
        storage_bench = json.load(f)

# ---------- unified paper palette ----------
PRIMARY = "#2d1157"  # dark purple — main series / emphasis
ACCENT = "#0097a7"  # teal — secondary series
LIGHT = "#b998f7"  # light purple — fills, secondary bars
NEUTRAL = "#666666"  # annotations, value labels
ERROR_CLR = "#999999"  # error bar color

# Purple gradient for storage bar chart (darkest → lightest)
_purple_shades = [
    "#1a0533",  # original (fp32) - deepest
    "#2d1157",  # float16
    "#421d7a",  # fp8
    "#572a9e",  # int8
    "#6c37c2",  # int4
    "#8247e0",  # int3
    "#9d6ef0",  # int2
    "#b998f7",  # binary - lightest
]

# ---------- global rcParams ----------
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "grid.alpha": 0.15,
        "grid.linewidth": 0.4,
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "lines.linewidth": 1.6,
        "lines.markersize": 6,
    }
)

# ---------- constants ----------
METHOD_ORDER = ["float16", "fp8", "int8", "int4", "int3", "int2", "binary"]
COMPRESSION = {
    "float16": 1.86,
    "fp8": 3.47,
    "int8": 3.45,
    "int4": 6.29,
    "int3": 7.94,
    "int2": 11.41,
    "binary": 16.53,
}
BITS = {
    "float16": 16,
    "fp8": 8,
    "int8": 8,
    "int4": 4,
    "int3": 3,
    "int2": 2,
    "binary": 1,
}
STORAGE_GIB = {
    "original": 182.88,
    "float16": 98.24,
    "fp8": 52.70,
    "int8": 52.96,
    "int4": 29.08,
    "int3": 23.02,
    "int2": 16.02,
    "binary": 11.07,
}

# Build lookup from full results
metrics = {r["method"]: r for r in full["results"]}


# =====================================================================
# Figure 1: Compression-Quality Pareto (HERO FIGURE)
# =====================================================================
fig, ax = plt.subplots(figsize=(5.5, 3.0))

# Per-method label offsets (dx, dy) in offset-points
LABEL_OFFSETS = {
    "float16": (0, 8),
    "fp8": (0, -12),
    "int8": (10, 7),
    "int4": (10, 5),
    "int3": (10, 5),
    "int2": (0, -12),
    "binary": (-10, 5),
}

xs = [COMPRESSION[m] for m in METHOD_ORDER]
ys_cos = [metrics[m]["knn_recall_10_cosine_sampled"] for m in METHOD_ORDER]
ys_euc = [metrics[m]["knn_recall_10_euclidean_sampled"] for m in METHOD_ORDER]

ax.plot(xs, ys_cos, "o-", color=PRIMARY, label="Cosine", zorder=2, markersize=5, linewidth=1.1)
ax.plot(
    xs,
    ys_euc,
    "s--",
    color=ACCENT,
    label="Euclidean",
    zorder=2,
    markersize=4.5,
    linewidth=1.1,
)

# Annotate methods at the midpoint between curves — all NEUTRAL
for m, x, yc, ye in zip(METHOD_ORDER, xs, ys_cos, ys_euc, strict=False):
    y_mid = (yc + ye) / 2
    dx, dy = LABEL_OFFSETS[m]
    ax.annotate(
        m,
        (x, y_mid),
        textcoords="offset points",
        xytext=(dx, dy),
        fontsize=6.5,
        ha="center",
        color=NEUTRAL,
        fontweight="bold",
    )

# Int8 reference lines
ax.axhline(
    y=metrics["int8"]["knn_recall_10_cosine_sampled"],
    color=PRIMARY,
    linestyle=":",
    alpha=0.30,
    linewidth=0.8,
    zorder=1,
)
ax.axhline(
    y=metrics["int8"]["knn_recall_10_euclidean_sampled"],
    color=ACCENT,
    linestyle=":",
    alpha=0.30,
    linewidth=0.8,
    zorder=1,
)

ax.set_xlabel("Compression Ratio (x)")
ax.set_ylabel("Recall@10")
ax.set_xlim(-0.5, 18)
ax.set_ylim(0.45, 1.05)
ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax.grid(True, alpha=0.15, linewidth=0.4)
ax.legend(fontsize=8, framealpha=0.8, edgecolor="none")

plt.tight_layout()
fig.savefig(FIG_DIR / "pareto.pdf")
plt.close()
print("Saved pareto.pdf")


# =====================================================================
# Figure 2: Storage Footprint Bar Chart
# =====================================================================
fig, ax = plt.subplots(figsize=(5.5, 2.2))
methods_bar = ["original"] + METHOD_ORDER
sizes = [STORAGE_GIB[m] for m in methods_bar]

bars = ax.barh(
    range(len(methods_bar)),
    sizes,
    color=_purple_shades,
    edgecolor="white",
    linewidth=0.8,
    height=0.65,
    alpha=0.88,
)
ax.set_yticks(range(len(methods_bar)))
ax.set_yticklabels(
    [f"{m} ({BITS.get(m, 32)}b)" if m != "original" else "original (fp32)" for m in methods_bar],
    fontsize=8,
)
ax.set_xlabel("Storage (GiB)")
ax.invert_yaxis()
ax.grid(True, axis="x", alpha=0.15, linewidth=0.4)
ax.set_xlim(0, 200)

for bar, size in zip(bars, sizes, strict=False):
    ax.text(
        bar.get_width() + 1.5,
        bar.get_y() + bar.get_height() / 2,
        f"{size:.1f}",
        va="center",
        fontsize=7,
        color=NEUTRAL,
    )

plt.tight_layout()
fig.savefig(FIG_DIR / "storage.pdf")
plt.close()
print("Saved storage.pdf")


# =====================================================================
# Figure 3: Full kNN Recall at k=10,25,50 (cosine)
# =====================================================================
fig, ax = plt.subplots(figsize=(5.5, 2.5))

# Three purple shades + varying markers for colorblind safety
K_STYLE: dict[int, dict] = {
    10: {"color": PRIMARY, "marker": "o", "markersize": 5.5, "linewidth": 1.5},
    25: {"color": "#7c4dff", "marker": "s", "markersize": 4.5, "linewidth": 1.2},
    50: {"color": LIGHT, "marker": "D", "markersize": 4.0, "linewidth": 1.0},
}

ks = [10, 25, 50]
for k in ks:
    key = f"knn_recall_{k}_cosine_sampled"
    vals = [metrics[m][key] for m in METHOD_ORDER]
    compressions = [COMPRESSION[m] for m in METHOD_ORDER]
    style = K_STYLE[k]
    ax.plot(
        compressions,
        vals,
        marker=style["marker"],
        linestyle="-",
        label=f"k={k}",
        markersize=style["markersize"],
        linewidth=style["linewidth"],
        color=style["color"],
        alpha=0.9,
    )

ax.set_xlabel("Compression Ratio (x)")
ax.set_ylabel("kNN Recall (Cosine)")
ax.legend(fontsize=8, framealpha=0.8, edgecolor="none")
ax.set_ylim(0.45, 1.05)
ax.grid(True, alpha=0.15, linewidth=0.4)

# Selective labeling on the k=10 curve — skip int3 to avoid overlap
k10_key = "knn_recall_10_cosine_sampled"
LABEL_METHODS_K = ["float16", "fp8", "int8", "int4", "int2", "binary"]
K_LABEL_OFFSETS = {
    "float16": (0, 7),
    "fp8": (12, -12),
    "int8": (-12, 7),
    "int4": (0, -10),
    "int2": (0, -10),
    "binary": (0, 7),
}

for m in LABEL_METHODS_K:
    cx, cy = COMPRESSION[m], metrics[m][k10_key]
    dx, dy = K_LABEL_OFFSETS[m]
    ax.annotate(
        m,
        (cx, cy),
        textcoords="offset points",
        xytext=(dx, dy),
        fontsize=6.5,
        ha="center",
        color=NEUTRAL,
        fontweight="bold",
    )

plt.tight_layout()
fig.savefig(FIG_DIR / "knn_recall_k.pdf")
plt.close()
print("Saved knn_recall_k.pdf")


# =====================================================================
# Figure 4: Intrinsic dimension context
# =====================================================================
id_data = quant10k["id_estimation"]
fig, ax = plt.subplots(figsize=(3.5, 2.2))

raw = id_data["raw"]
ests = ["MLE", "TwoNN", "LPCA"]
vals = [raw["id_mle"], raw["id_twonn"], raw["id_lpca"]]
stds = [raw["id_mle_std"], raw["id_twonn_std"], raw["id_lpca_std"]]

# Monochrome — all PRIMARY
bars = ax.bar(
    ests,
    vals,
    yerr=stds,
    color=PRIMARY,
    edgecolor="white",
    linewidth=0.8,
    width=0.5,
    capsize=3,
    error_kw={"ecolor": ERROR_CLR, "linewidth": 0.8},
    alpha=0.88,
)
ax.set_ylabel("Intrinsic Dimension")
ax.set_ylim(0, max(vals) * 1.25)

for bar, v in zip(bars, vals, strict=False):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.5,
        f"{v:.1f}",
        ha="center",
        fontsize=7,
        color=NEUTRAL,
    )

ax.text(
    0.98,
    0.95,
    "ambient dim = 1024",
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=7,
    color=ACCENT,
    bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "none", "alpha": 0.9},
)

plt.tight_layout()
fig.savefig(FIG_DIR / "intrinsic_dim.pdf")
plt.close()
print("Saved intrinsic_dim.pdf")


# =====================================================================
# Figure 5: Search Benchmark — QPS (FAISS CPU vs GPU) + Recall@10
# =====================================================================
faiss_path = RESULTS_DIR / "faiss_results_1M.json"
gpu_path = RESULTS_DIR / "gpu_results_1M.json"

if faiss_path.exists() and gpu_path.exists():
    with open(faiss_path) as f:
        faiss_res = json.load(f)
    with open(gpu_path) as f:
        gpu_res = json.load(f)

    _faiss_exp = faiss_res["experiments"]
    _gpu_exp = gpu_res["experiments"]
    SEARCH_METHODS = list(
        reversed(["binary", "int2", "int3", "int4", "fp8", "int8", "float16"])
    )
    FAISS_KEY = {
        "binary": "binary_hamming",
        "int2": "int2_flat",
        "int3": "int3_flat",
        "int4": "int4_flat",
        "fp8": "fp8_flat",
        "int8": "int8_flat",
        "float16": "float16_flat",
    }
    GPU_KEY = {m: f"{m}_gpu" for m in SEARCH_METHODS}

    cpu_qps = [_faiss_exp[FAISS_KEY[m]]["qps"] for m in SEARCH_METHODS]
    gpu_qps = [_gpu_exp[GPU_KEY[m]]["qps"] for m in SEARCH_METHODS]
    recall10 = [_faiss_exp[FAISS_KEY[m]]["recall"]["recall@10"] for m in SEARCH_METHODS]

    fig, ax1 = plt.subplots(figsize=(5.5, 3.6))
    x = np.arange(len(SEARCH_METHODS))
    w = 0.32

    bars_cpu = ax1.bar(
        x - w / 2,
        cpu_qps,
        w,
        label="FAISS CPU",
        color=LIGHT,
        edgecolor="white",
        linewidth=0.6,
        alpha=0.88,
    )
    bars_gpu = ax1.bar(
        x + w / 2,
        gpu_qps,
        w,
        label="GPU (RTX 3090)",
        color=PRIMARY,
        edgecolor="white",
        linewidth=0.6,
        alpha=0.88,
    )

    ax1.set_ylabel("Queries / Second")
    ax1.set_xticks(x)
    ax1.set_xticklabels(SEARCH_METHODS, fontsize=8)
    ax1.set_yscale("log")
    ax1.set_ylim(10, 5000)

    for bars in [bars_cpu, bars_gpu]:
        for bar in bars:
            h = bar.get_height()
            if h <= 0:
                continue  # skip labels on zero-height (missing GPU) bars
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                h * 1.08,
                f"{h:.0f}",
                ha="center",
                fontsize=6,
                color=NEUTRAL,
            )

    # Recall on twin axis — re-enable right spine for it
    ax2 = ax1.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_linewidth(0.6)
    ax2.plot(
        x, recall10, "D-", color=ACCENT, markersize=5, linewidth=1.3, label="Recall@10", zorder=5
    )
    ax2.set_ylabel("Recall@10", color=ACCENT)
    ax2.set_ylim(0.3, 1.08)
    ax2.tick_params(axis="y", labelcolor=ACCENT)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        fontsize=7,
        loc="lower right",
        framealpha=0.85,
        edgecolor="none",
    )

    ax1.grid(True, axis="y", alpha=0.15, linewidth=0.4)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "search_benchmark.pdf")
    plt.close()
    print("Saved search_benchmark.pdf")
else:
    print("Skipping search_benchmark.pdf (results not found)")  # noqa: E501


# =====================================================================
# Figure 6: TurboQuant vs. standard intb — bit-width saturation
# =====================================================================
turbo_metrics = {r["method"]: r for r in turbo_full["results"]}

TURBO_BITS = [2, 3, 4, 8]
int_methods = [f"int{b}" for b in TURBO_BITS]
turbo_methods = [f"turbo{b}" for b in TURBO_BITS]

int_recall_cos = [metrics[m]["knn_recall_10_cosine_sampled"] for m in int_methods]
turbo_recall_cos = [turbo_metrics[m]["knn_recall_10_cosine_sampled"] for m in turbo_methods]

x = np.array(TURBO_BITS)

fig, ax = plt.subplots(figsize=(4.0, 2.8))

ax.fill_between(x, turbo_recall_cos, int_recall_cos, color=PRIMARY, alpha=0.08, zorder=1)
ax.plot(
    x,
    int_recall_cos,
    "o-",
    color=PRIMARY,
    linewidth=1.6,
    markersize=5.5,
    label="int$b$",
    zorder=3,
)
ax.plot(
    x,
    turbo_recall_cos,
    "s--",
    color=ACCENT,
    linewidth=1.6,
    markersize=5.5,
    label="turbo$b$",
    zorder=3,
)

# Per-bit label offsets to avoid overlap at 8-bit
INT_LABEL_OFFSETS = {2: (0, 7), 3: (0, 7), 4: (-16, 0), 8: (-18, 0)}
TURBO_LABEL_OFFSETS = {2: (0, -12), 3: (0, -12), 4: (16, 0), 8: (18, 0)}

for xi, iv, tv in zip(x, int_recall_cos, turbo_recall_cos, strict=False):
    dx_i, dy_i = INT_LABEL_OFFSETS[int(xi)]
    ax.annotate(
        f"{iv:.2f}",
        (xi, iv),
        textcoords="offset points",
        xytext=(dx_i, dy_i),
        fontsize=6.5,
        ha="center",
        color=PRIMARY,
        fontweight="bold",
    )
    dx_t, dy_t = TURBO_LABEL_OFFSETS[int(xi)]
    ax.annotate(
        f"{tv:.2f}",
        (xi, tv),
        textcoords="offset points",
        xytext=(dx_t, dy_t),
        fontsize=6.5,
        ha="center",
        color=ACCENT,
        fontweight="bold",
    )

# BUG FIX: compute turbo - int (turbo is better), find largest absolute gap
# The largest gap is at 2-bit, not 8-bit
turbo_gap = np.array(turbo_recall_cos) - np.array(int_recall_cos)
best_gap_idx = int(np.argmax(np.abs(turbo_gap)))
gap_val = turbo_gap[best_gap_idx]
gap_sign = "+" if gap_val > 0 else ""
gap_label = f"$\\Delta = {gap_sign}{gap_val:.2f}$"
ax.annotate(
    gap_label,
    xy=(x[best_gap_idx], turbo_recall_cos[best_gap_idx]),
    xytext=(x[best_gap_idx] + 2.5, turbo_recall_cos[best_gap_idx] + 0.02),
    fontsize=7,
    color=ACCENT,
    fontweight="bold",
    arrowprops={"arrowstyle": "->", "color": ACCENT, "lw": 0.9},
    ha="center",
)

ax.set_xlabel("Bit-width")
ax.set_ylabel("kNN Recall@10 (Cosine)")
ax.set_xticks(TURBO_BITS)
ax.set_xticklabels([f"{b}b" for b in TURBO_BITS])
ax.set_ylim(0.42, 1.06)
ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax.grid(True, alpha=0.15, linewidth=0.4)
ax.legend(fontsize=8, framealpha=0.85, edgecolor="none")

plt.tight_layout()
fig.savefig(FIG_DIR / "turbo_vs_int.pdf")
plt.close()
print("Saved turbo_vs_int.pdf")


# =====================================================================
# Figure 7: Storage backend comparison
# =====================================================================
if storage_bench is not None:
    methods_storage = ["float32", "float16", "fp8", "int8", "int4", "int2", "binary"]
    labels_storage = ["fp32", "fp16", "fp8", "int8", "int4", "int2", "binary"]

    best_parquet = []
    best_external = []
    for method in methods_storage:
        rows = storage_bench["methods"][method]
        best_parquet.append(min(rows["parquet"], key=lambda r: r["bytes"]))
        best_external.append(min(rows["external"], key=lambda r: r["bytes"]))

    fig, ax = plt.subplots(figsize=(5.8, 2.9))
    x = np.arange(len(methods_storage))
    w = 0.36

    bars_p = ax.bar(
        x - w / 2,
        [r["bits_per_dim"] for r in best_parquet],
        w,
        color=PRIMARY,
        edgecolor="white",
        linewidth=0.7,
        alpha=0.9,
        label="best parquet",
    )
    bars_e = ax.bar(
        x + w / 2,
        [r["bits_per_dim"] for r in best_external],
        w,
        color=ACCENT,
        edgecolor="white",
        linewidth=0.7,
        alpha=0.9,
        label="best external codec",
    )

    for bars in (bars_p, bars_e):
        for bar in bars:
            h = bar.get_height()
            ax.text(
                float(bar.get_x() + bar.get_width() / 2),
                h + 0.18,
                f"{h:.1f}",
                ha="center",
                fontsize=6.5,
                color=NEUTRAL,
                rotation=90,
            )

    ax.annotate(
        "BSS helps\nfp32/fp16",
        xy=(x[0] - w / 2, best_parquet[0]["bits_per_dim"]),
        xytext=(0.7, 31.0),
        fontsize=7,
        color=PRIMARY,
        arrowprops={"arrowstyle": "->", "color": PRIMARY, "lw": 0.9},
        ha="center",
    )
    ax.annotate(
        "dim-major wins\nfor low-bit codes",
        xy=(x[5] + w / 2, best_external[5]["bits_per_dim"]),
        xytext=(4.9, 6.2),
        fontsize=7,
        color=ACCENT,
        arrowprops={"arrowstyle": "->", "color": ACCENT, "lw": 0.9},
        ha="center",
    )

    ax.set_ylabel("Effective Bits / Original Dim")
    ax.set_xlabel("Representation")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_storage)
    ax.set_ylim(0, 33.5)
    ax.grid(True, axis="y", alpha=0.15, linewidth=0.4)
    ax.legend(fontsize=8, framealpha=0.85, edgecolor="none", loc="upper right")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "storage_backends.pdf")
    plt.close()
    print("Saved storage_backends.pdf")
else:
    print("Skipping storage_backends.pdf (storage benchmark report not found)")


# =====================================================================
# Figure 8: Binary variants — sample vs full transfer
# =====================================================================
if binary_baselines is not None and binary_full is not None:
    rows_sample = {r["method"]: r for r in binary_baselines["results"]}
    rows_full = {r["method"]: r for r in binary_full["results"]}
    order = ["binary", "binary_med", "binary_zscore"]
    labels = ["binary", "binary_med", "binary_z"]
    sample_vals = [rows_sample[m]["knn_recall_10_cosine"] for m in order]
    full_vals = [rows_full[m]["knn_recall_10_cosine_sampled"] for m in order]

    fig, ax = plt.subplots(figsize=(5.8, 2.9))
    xb = np.arange(len(order))
    w = 0.34

    bars_sample = ax.bar(
        xb - w / 2,
        sample_vals,
        w,
        color=LIGHT,
        edgecolor="white",
        linewidth=0.7,
        alpha=0.9,
        label="20k subsample",
    )
    bars_full = ax.bar(
        xb + w / 2,
        full_vals,
        w,
        color=PRIMARY,
        edgecolor="white",
        linewidth=0.7,
        alpha=0.9,
        label="full dataset",
    )

    for bars in (bars_sample, bars_full):
        for bar in bars:
            h = bar.get_height()
            ax.text(
                float(bar.get_x() + bar.get_width() / 2),
                h + 0.012,
                f"{h:.3f}",
                ha="center",
                fontsize=7,
                color=NEUTRAL,
            )

    ax.annotate(
        "promising on\nsubsample",
        xy=(xb[1] - w / 2, sample_vals[1]),
        xytext=(0.55, 0.82),
        fontsize=7,
        color=ACCENT,
        arrowprops={"arrowstyle": "->", "color": ACCENT, "lw": 0.9},
        ha="center",
    )
    ax.annotate(
        "calibration drift\nat lake scale",
        xy=(xb[1] + w / 2, full_vals[1]),
        xytext=(1.65, 0.28),
        fontsize=7,
        color=PRIMARY,
        arrowprops={"arrowstyle": "->", "color": PRIMARY, "lw": 0.9},
        ha="center",
    )

    ax.set_ylabel("kNN Recall@10 (Cosine)")
    ax.set_xlabel("Method")
    ax.set_xticks(xb)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.2, 0.9)
    ax.grid(True, axis="y", alpha=0.15, linewidth=0.4)
    ax.legend(fontsize=8, framealpha=0.85, edgecolor="none", loc="upper right")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "binary_variants.pdf")
    plt.close()
    print("Saved binary_variants.pdf")
else:
    print("Skipping binary_variants.pdf (binary baseline reports not found)")


print("\nAll figures generated.")
