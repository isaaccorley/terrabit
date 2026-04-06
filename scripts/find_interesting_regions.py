"""Find interesting exemplar points/regions in the binary embedding dataset.

Analyses:
1. Temporal change: tiles where 2024↔2025 embeddings diverge most
2. Global outliers: embeddings most different from the global centroid
3. Intra-tile diversity: tiles with highest internal embedding variance
4. Cluster centroids: k-means on binary embeddings → representative types
5. High bit-entropy regions
"""

import glob
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from rich.console import Console
from rich.table import Table

console = Console()
DATA_DIR = Path("embeddings/clay-v1_5-binary-sentinel-2")
RNG = np.random.default_rng(42)

# Precompute popcount LUT
_POPCOUNT_LUT = np.array([i.bit_count() for i in range(256)], dtype=np.int32)


def load_tile_fast(tile_x: int, tile_y: int, year: int | None = None) -> dict:
    """Load tile using vectorized pyarrow reads."""
    if year:
        pattern = str(DATA_DIR / f"tile_x={tile_x}/tile_y={tile_y}/year={year}/*.parquet")
    else:
        pattern = str(DATA_DIR / f"tile_x={tile_x}/tile_y={tile_y}/year=*/*.parquet")
    files = glob.glob(pattern)
    if not files:
        return {
            "embeddings": np.empty((0, 128), dtype=np.uint8),
            "lats": np.empty(0),
            "lons": np.empty(0),
        }

    embs_list, lats_list, lons_list, years_list = [], [], [], []
    for f in files:
        t = pq.read_table(f, columns=["embedding", "bbox"])
        n = len(t)
        if n == 0:
            continue
        # Extract embeddings as flat array then reshape
        emb_col = t.column("embedding")
        flat = emb_col.combine_chunks().flatten().to_numpy(zero_copy_only=False)
        embs_list.append(flat.reshape(n, 128))

        # Extract bbox centers
        bbox_col = t.column("bbox").combine_chunks()
        xmin = bbox_col.field("xmin").to_numpy(zero_copy_only=False)
        xmax = bbox_col.field("xmax").to_numpy(zero_copy_only=False)
        ymin = bbox_col.field("ymin").to_numpy(zero_copy_only=False)
        ymax = bbox_col.field("ymax").to_numpy(zero_copy_only=False)
        lons_list.append((xmin + xmax) / 2)
        lats_list.append((ymin + ymax) / 2)

        # Extract year from path
        for part in f.split("/"):
            if part.startswith("year="):
                years_list.append(np.full(n, int(part.split("=")[1])))
                break

    if not embs_list:
        return {
            "embeddings": np.empty((0, 128), dtype=np.uint8),
            "lats": np.empty(0),
            "lons": np.empty(0),
        }

    return {
        "embeddings": np.concatenate(embs_list),
        "lats": np.concatenate(lats_list),
        "lons": np.concatenate(lons_list),
        "years": np.concatenate(years_list) if years_list else np.empty(0),
    }


def hamming_to_ref(embs: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Hamming distance of each row in embs to ref (1D)."""
    return _POPCOUNT_LUT[np.bitwise_xor(embs, ref[np.newaxis, :])].sum(axis=1)


def binary_centroid(embs: np.ndarray) -> np.ndarray:
    bits = np.unpackbits(embs, axis=1)
    majority = (bits.mean(axis=0) >= 0.5).astype(np.uint8)
    return np.packbits(majority)


def get_all_tile_coords() -> list[tuple[int, int]]:
    dirs = glob.glob(str(DATA_DIR / "tile_x=*/tile_y=*"))
    coords = set()
    for d in dirs:
        parts = d.split("/")
        tx = ty = None
        for p in parts:
            if p.startswith("tile_x="):
                tx = int(p.split("=")[1])
            elif p.startswith("tile_y="):
                ty = int(p.split("=")[1])
        if tx is not None and ty is not None:
            coords.add((tx, ty))
    return list(coords)


def find_temporal_change(tile_coords: list, n_sample: int = 3000) -> list:
    console.print("\n[bold cyan]═══ Analysis 1: Temporal Change Detection ═══[/bold cyan]")
    sampled = RNG.choice(len(tile_coords), size=min(n_sample, len(tile_coords)), replace=False)
    results = []
    for idx, i in enumerate(sampled):
        tx, ty = tile_coords[i]
        d24 = load_tile_fast(tx, ty, 2024)
        d25 = load_tile_fast(tx, ty, 2025)
        if len(d24["embeddings"]) < 5 or len(d25["embeddings"]) < 5:
            continue
        c24 = binary_centroid(d24["embeddings"])
        c25 = binary_centroid(d25["embeddings"])
        change = int(_POPCOUNT_LUT[np.bitwise_xor(c24, c25)].sum())
        mid = len(d24["lats"]) // 2
        results.append(
            {
                "tile_x": tx,
                "tile_y": ty,
                "change_score": change,
                "lat": float(d24["lats"][mid]),
                "lon": float(d24["lons"][mid]),
                "n_2024": len(d24["embeddings"]),
                "n_2025": len(d25["embeddings"]),
            }
        )
        if (idx + 1) % 500 == 0:
            console.print(f"  {idx + 1}/{len(sampled)} tiles...")
    results.sort(key=lambda x: x["change_score"], reverse=True)
    return results[:30]


def find_global_outliers(tile_coords: list, n_sample: int = 1500) -> list:
    console.print("\n[bold cyan]═══ Analysis 2: Global Outlier Detection ═══[/bold cyan]")
    sampled = RNG.choice(len(tile_coords), size=min(n_sample, len(tile_coords)), replace=False)

    # Phase 1: global centroid from subsampled embeddings
    console.print("  Computing global centroid...")
    centroid_embs = []
    for idx, i in enumerate(sampled):
        tx, ty = tile_coords[i]
        data = load_tile_fast(tx, ty)
        if len(data["embeddings"]) > 0:
            sel = RNG.choice(
                len(data["embeddings"]), size=min(15, len(data["embeddings"])), replace=False
            )
            centroid_embs.append(data["embeddings"][sel])
        if (idx + 1) % 500 == 0:
            console.print(f"    {idx + 1}/{len(sampled)}...")
    centroid_embs = np.concatenate(centroid_embs)
    centroid = binary_centroid(centroid_embs)
    console.print(f"  Centroid from {len(centroid_embs)} samples")

    # Phase 2: find furthest points
    console.print("  Scanning for outliers...")
    candidates = []
    for idx, i in enumerate(sampled):
        tx, ty = tile_coords[i]
        data = load_tile_fast(tx, ty)
        if len(data["embeddings"]) == 0:
            continue
        dists = hamming_to_ref(data["embeddings"], centroid)
        top3 = np.argsort(dists)[-3:]
        candidates.extend(
            {
                "tile_x": tx,
                "tile_y": ty,
                "distance": int(dists[j]),
                "lat": float(data["lats"][j]),
                "lon": float(data["lons"][j]),
                "year": int(data["years"][j]),
            }
            for j in top3
        )
        if (idx + 1) % 500 == 0:
            console.print(f"    {idx + 1}/{len(sampled)}...")

    candidates.sort(key=lambda x: x["distance"], reverse=True)
    deduped = []
    for c in candidates:
        if not any(
            abs(c["lat"] - d["lat"]) < 0.5 and abs(c["lon"] - d["lon"]) < 0.5 for d in deduped
        ):
            deduped.append(c)
        if len(deduped) >= 30:
            break
    return deduped


def find_diverse_tiles(tile_coords: list, n_sample: int = 1500) -> list:
    console.print("\n[bold cyan]═══ Analysis 3: Intra-tile Diversity ═══[/bold cyan]")
    sampled = RNG.choice(len(tile_coords), size=min(n_sample, len(tile_coords)), replace=False)
    results = []
    for idx, i in enumerate(sampled):
        tx, ty = tile_coords[i]
        data = load_tile_fast(tx, ty)
        if len(data["embeddings"]) < 10:
            continue
        c = binary_centroid(data["embeddings"])
        dists = hamming_to_ref(data["embeddings"], c)
        results.append(
            {
                "tile_x": tx,
                "tile_y": ty,
                "variance": float(np.var(dists)),
                "mean_dist": float(np.mean(dists)),
                "lat": float(np.median(data["lats"])),
                "lon": float(np.median(data["lons"])),
                "n_patches": len(data["embeddings"]),
            }
        )
        if (idx + 1) % 500 == 0:
            console.print(f"  {idx + 1}/{len(sampled)}...")
    results.sort(key=lambda x: x["variance"], reverse=True)
    return results[:30]


def find_cluster_centroids(tile_coords: list, n_clusters: int = 20) -> list:
    console.print("\n[bold cyan]═══ Analysis 4: Cluster Centroids ═══[/bold cyan]")
    from sklearn.cluster import MiniBatchKMeans

    sampled = RNG.choice(len(tile_coords), size=min(2000, len(tile_coords)), replace=False)
    all_embs, all_lats, all_lons, all_years = [], [], [], []
    console.print("  Sampling...")
    for idx, i in enumerate(sampled):
        tx, ty = tile_coords[i]
        data = load_tile_fast(tx, ty)
        if len(data["embeddings"]) == 0:
            continue
        sel = RNG.choice(
            len(data["embeddings"]), size=min(8, len(data["embeddings"])), replace=False
        )
        all_embs.append(data["embeddings"][sel])
        all_lats.append(data["lats"][sel])
        all_lons.append(data["lons"][sel])
        all_years.append(data["years"][sel])
        if (idx + 1) % 500 == 0:
            console.print(f"    {idx + 1}/{len(sampled)}...")

    embs = np.concatenate(all_embs)
    lats = np.concatenate(all_lats)
    lons = np.concatenate(all_lons)
    years = np.concatenate(all_years)
    console.print(f"  Clustering {len(embs)} embeddings...")

    X = np.unpackbits(embs, axis=1).astype(np.float32)
    km = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=2048, n_init=3)
    labels = km.fit_predict(X)

    results = []
    for c in range(n_clusters):
        mask = labels == c
        dists = np.linalg.norm(X[mask] - km.cluster_centers_[c], axis=1)
        nearest = np.argmin(dists)
        orig_idx = np.where(mask)[0][nearest]
        results.append(
            {
                "cluster": c,
                "size": int(mask.sum()),
                "lat": float(lats[orig_idx]),
                "lon": float(lons[orig_idx]),
                "year": int(years[orig_idx]),
                "tile_x": 0,
                "tile_y": 0,  # placeholder
            }
        )
    results.sort(key=lambda x: x["size"], reverse=True)
    return results


def find_high_entropy(tile_coords: list, n_sample: int = 1500) -> list:
    console.print("\n[bold cyan]═══ Analysis 5: High Bit-Entropy Regions ═══[/bold cyan]")
    sampled = RNG.choice(len(tile_coords), size=min(n_sample, len(tile_coords)), replace=False)
    results = []
    for idx, i in enumerate(sampled):
        tx, ty = tile_coords[i]
        data = load_tile_fast(tx, ty)
        if len(data["embeddings"]) < 20:
            continue
        bits = np.unpackbits(data["embeddings"], axis=1).astype(np.float32)
        p = bits.mean(axis=0)
        entropy = float(np.mean(p * (1 - p)))
        results.append(
            {
                "tile_x": tx,
                "tile_y": ty,
                "entropy": entropy,
                "lat": float(np.median(data["lats"])),
                "lon": float(np.median(data["lons"])),
                "n_patches": len(data["embeddings"]),
            }
        )
        if (idx + 1) % 500 == 0:
            console.print(f"  {idx + 1}/{len(sampled)}...")
    results.sort(key=lambda x: x["entropy"], reverse=True)
    return results[:30]


def print_results(title: str, results: list, key_cols: list[str]) -> None:
    table = Table(title=title, show_lines=True)
    table.add_column("#", style="dim")
    table.add_column("Lat", style="green")
    table.add_column("Lon", style="green")
    for col in key_cols:
        table.add_column(col, style="cyan")
    table.add_column("Tile", style="dim")
    for i, r in enumerate(results[:15]):
        table.add_row(
            str(i + 1),
            f"{r['lat']:.3f}",
            f"{r['lon']:.3f}",
            *[str(round(r[c], 4) if isinstance(r[c], float) else r[c]) for c in key_cols],
            f"({r.get('tile_x', '?')},{r.get('tile_y', '?')})",
        )
    console.print(table)


def fmt_bbox(lat: float, lon: float, half: float = 0.3) -> str:
    return (
        f"{{ west: {lon - half:.2f}, south: {lat - half:.2f}, "
        f"east: {lon + half:.2f}, north: {lat + half:.2f} }}"
    )


def main() -> None:
    console.print("[bold]Scanning dataset for interesting regions...[/bold]")
    tile_coords = get_all_tile_coords()
    console.print(f"Found {len(tile_coords)} unique tiles\n")

    temporal = find_temporal_change(tile_coords)
    print_results("Temporal Change (2024→2025)", temporal, ["change_score", "n_2024", "n_2025"])

    outliers = find_global_outliers(tile_coords)
    print_results("Global Outliers", outliers, ["distance", "year"])

    diverse = find_diverse_tiles(tile_coords)
    print_results("Diverse/Mixed Tiles", diverse, ["variance", "mean_dist"])

    clusters = find_cluster_centroids(tile_coords)
    print_results("Cluster Centroids", clusters, ["cluster", "size", "year"])

    entropy = find_high_entropy(tile_coords)
    print_results("High Entropy Regions", entropy, ["entropy"])

    # Print suggested AOI presets
    console.print(
        "\n[bold yellow]═══ Suggested AOI Presets for app/src/main.ts ═══[/bold yellow]\n"
    )

    console.print("// --- Temporal hotspots (most change 2024→2025) ---")
    for r in temporal[:5]:
        console.print(
            f'  {{ name: "Change @({r["lat"]:.1f},{r["lon"]:.1f})", '
            f'tag: "temporal Δ={r["change_score"]}", '
            f"bbox: {fmt_bbox(r['lat'], r['lon'])} }},"
        )

    console.print("\n// --- Global outliers (rarest surfaces) ---")
    for r in outliers[:5]:
        console.print(
            f'  {{ name: "Outlier @({r["lat"]:.1f},{r["lon"]:.1f})", '
            f'tag: "dist={r["distance"]}", '
            f"bbox: {fmt_bbox(r['lat'], r['lon'])} }},"
        )

    console.print("\n// --- Mixed landscapes (highest diversity) ---")
    for r in diverse[:5]:
        console.print(
            f'  {{ name: "Diverse @({r["lat"]:.1f},{r["lon"]:.1f})", '
            f'tag: "var={r["variance"]:.1f}", '
            f"bbox: {fmt_bbox(r['lat'], r['lon'])} }},"
        )

    console.print("\n// --- Cluster centroids (representative surface types) ---")
    for r in clusters[:10]:
        console.print(
            f'  {{ name: "Cluster {r["cluster"]} @({r["lat"]:.1f},{r["lon"]:.1f})", '
            f'tag: "n={r["size"]}", '
            f"bbox: {fmt_bbox(r['lat'], r['lon'])} }},"
        )

    console.print("\n// --- High entropy (information-rich) ---")
    for r in entropy[:5]:
        console.print(
            f'  {{ name: "Entropy @({r["lat"]:.1f},{r["lon"]:.1f})", '
            f'tag: "H={r["entropy"]:.4f}", '
            f"bbox: {fmt_bbox(r['lat'], r['lon'])} }},"
        )


if __name__ == "__main__":
    main()
