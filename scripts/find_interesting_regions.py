"""Find interesting/weird exemplar points in the binary embedding dataset.

Analyses:
1. Non-polar temporal change: tiles where 2024↔2025 embeddings diverge most,
   excluding high-latitude sea-ice churn (|lat| > 62°).
2. k-NN isolation outliers: patches with the highest mean distance to their
   K nearest neighbours globally — truly has-no-close-relatives weird.
3. Intra-tile diversity (inland): tiles with high internal embedding variance,
   excluding obvious coastal land/ocean boundary tiles.
4. Rare cluster types: k-means with many clusters → smallest clusters are the
   rarest surface types; return most isolated point from each.
5. High bit-entropy regions.
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
        emb_col = t.column("embedding")
        flat = emb_col.combine_chunks().flatten().to_numpy(zero_copy_only=False)
        embs_list.append(flat.reshape(n, 128))

        bbox_col = t.column("bbox").combine_chunks()
        xmin = bbox_col.field("xmin").to_numpy(zero_copy_only=False)
        xmax = bbox_col.field("xmax").to_numpy(zero_copy_only=False)
        ymin = bbox_col.field("ymin").to_numpy(zero_copy_only=False)
        ymax = bbox_col.field("ymax").to_numpy(zero_copy_only=False)
        lons_list.append((xmin + xmax) / 2)
        lats_list.append((ymin + ymax) / 2)

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
    """Tiles where embedding centroid changed most 2024→2025, excluding high-latitude ice churn."""
    console.print("\n[bold cyan]═══ Analysis 1: Non-polar Temporal Change ═══[/bold cyan]")
    sampled = RNG.choice(len(tile_coords), size=min(n_sample, len(tile_coords)), replace=False)
    results = []
    for idx, i in enumerate(sampled):
        tx, ty = tile_coords[i]
        d24 = load_tile_fast(tx, ty, 2024)
        d25 = load_tile_fast(tx, ty, 2025)
        if len(d24["embeddings"]) < 5 or len(d25["embeddings"]) < 5:
            continue
        mid_lat = float(d24["lats"][len(d24["lats"]) // 2])
        # Skip polar/subpolar — seasonal ice dominates and is uninteresting
        if abs(mid_lat) > 62.0:
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


def find_knn_outliers(
    tile_coords: list,
    patches_per_tile: int = 3,
    k: int = 15,
    query_batch: int = 1000,
    ref_batch: int = 5000,
) -> list:
    """Patches most isolated in global embedding space — truly has-no-close-relatives weird.

    Exhaustive over the full dataset: loads patches from EVERY tile (no tile sampling),
    then computes exact k-NN isolation scores using a streaming top-K approach that
    avoids materialising the full NxN distance matrix.

    Memory at any point: query_batch x ref_batch x 128 bytes for XOR (~640 MB default).
    """
    console.print(
        "\n[bold cyan]═══ Analysis 2: k-NN Isolation Outliers (full dataset) ═══[/bold cyan]"
    )

    all_embs, all_lats, all_lons, all_years = [], [], [], []
    console.print(
        f"  Loading {patches_per_tile} patches from every tile ({len(tile_coords)} tiles)..."
    )
    for idx, (tx, ty) in enumerate(tile_coords):
        data = load_tile_fast(tx, ty)
        if len(data["embeddings"]) == 0:
            continue
        sel = RNG.choice(
            len(data["embeddings"]),
            size=min(patches_per_tile, len(data["embeddings"])),
            replace=False,
        )
        all_embs.append(data["embeddings"][sel])
        all_lats.append(data["lats"][sel])
        all_lons.append(data["lons"][sel])
        all_years.append(data["years"][sel])
        if (idx + 1) % 2000 == 0:
            console.print(f"    {idx + 1}/{len(tile_coords)} tiles loaded...")

    embs = np.concatenate(all_embs)  # N×128 uint8
    lats = np.concatenate(all_lats)
    lons = np.concatenate(all_lons)
    years = np.concatenate(all_years)
    n = len(embs)
    console.print(f"  {n} embeddings loaded. Computing streaming k-NN isolation scores...")

    # Streaming top-K: for each query batch, accumulate the K+1 smallest distances
    # seen so far across all reference batches. Never stores the full N×N matrix.
    knn_scores = np.zeros(n, dtype=np.float32)

    for qi in range(0, n, query_batch):
        q = embs[qi : qi + query_batch]
        qb = len(q)
        # running_topk[i] holds the (k+1) smallest distances seen for query i so far.
        # Initialised to a value > max possible Hamming (1024 bits).
        running_topk = np.full((qb, k + 1), 1025, dtype=np.int32)

        for ri in range(0, n, ref_batch):
            r = embs[ri : ri + ref_batch]
            xor = np.bitwise_xor(q[:, np.newaxis, :], r[np.newaxis, :, :])  # qb×rb×128
            d = _POPCOUNT_LUT[xor].sum(axis=2)  # qb×rb  int32
            combined = np.concatenate([running_topk, d], axis=1)
            running_topk = np.sort(combined, axis=1)[:, : k + 1]

        # running_topk[:, 0] == 0 (self-distance); skip it, average the rest
        knn_scores[qi : qi + qb] = running_topk[:, 1 : k + 1].mean(axis=1)

        if (qi // query_batch) % 20 == 0:
            console.print(f"    scored {qi}/{n}...")

    order = np.argsort(knn_scores)[::-1]
    results = []
    for idx in order:
        r = {
            "lat": float(lats[idx]),
            "lon": float(lons[idx]),
            "year": int(years[idx]),
            "isolation": float(knn_scores[idx]),
            "tile_x": 0,
            "tile_y": 0,
        }
        if not any(
            abs(r["lat"] - d["lat"]) < 0.5 and abs(r["lon"] - d["lon"]) < 0.5 for d in results
        ):
            results.append(r)
        if len(results) >= 30:
            break
    return results


def find_diverse_tiles(tile_coords: list, n_sample: int = 1500) -> list:
    """Tiles with high internal embedding variance, filtering out coastal land/ocean boundary noise.

    Coastal tiles score high simply because half the patches are ocean and half are land —
    not interesting. We filter by checking whether the intra-tile distance distribution is
    strongly bimodal (two tight groups far apart = coast boundary, not genuine diversity).
    """
    console.print("\n[bold cyan]═══ Analysis 3: Intra-tile Diversity (inland) ═══[/bold cyan]")
    sampled = RNG.choice(len(tile_coords), size=min(n_sample, len(tile_coords)), replace=False)
    results = []
    for idx, i in enumerate(sampled):
        tx, ty = tile_coords[i]
        data = load_tile_fast(tx, ty)
        if len(data["embeddings"]) < 10:
            continue
        c = binary_centroid(data["embeddings"])
        dists = hamming_to_ref(data["embeddings"], c)

        # Bimodality filter: if >35% of patches are within 20 bits of centroid AND
        # >35% are >80 bits from centroid, it's a land/ocean split — skip it.
        frac_near = (dists < 20).mean()
        frac_far = (dists > 80).mean()
        if frac_near > 0.35 and frac_far > 0.35:
            continue

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


def find_rare_clusters(tile_coords: list, n_clusters: int = 50) -> list:
    """Rarest surface types: run k-means with many clusters, return smallest clusters.

    Large clusters = common biomes. Small clusters = rare/unusual surfaces.
    Within each rare cluster we pick the most isolated point (furthest from
    cluster centroid) — the extreme, not the typical.
    """
    console.print("\n[bold cyan]═══ Analysis 4: Rare Cluster Types ═══[/bold cyan]")
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
    console.print(f"  Clustering {len(embs)} embeddings into {n_clusters} clusters...")

    X = np.unpackbits(embs, axis=1).astype(np.float32)
    km = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=2048, n_init=5)
    labels = km.fit_predict(X)

    results = []
    for c in range(n_clusters):
        mask = labels == c
        if mask.sum() == 0:
            continue
        # Pick the point most distant from centroid — most extreme example of this rare type
        dists = np.linalg.norm(X[mask] - km.cluster_centers_[c], axis=1)
        extreme = np.argmax(dists)
        orig_idx = np.where(mask)[0][extreme]
        results.append(
            {
                "cluster": c,
                "size": int(mask.sum()),
                "lat": float(lats[orig_idx]),
                "lon": float(lons[orig_idx]),
                "year": int(years[orig_idx]),
                "tile_x": 0,
                "tile_y": 0,
            }
        )
    # Smallest clusters first — rarest surface types
    results.sort(key=lambda x: x["size"])
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
    print_results(
        "Non-polar Temporal Change (2024→2025)", temporal, ["change_score", "n_2024", "n_2025"]
    )

    outliers = find_knn_outliers(tile_coords)  # exhaustive — all tiles
    print_results("k-NN Isolation Outliers (full dataset)", outliers, ["isolation", "year"])

    diverse = find_diverse_tiles(tile_coords)
    print_results("Diverse Tiles (inland)", diverse, ["variance", "mean_dist"])

    rare = find_rare_clusters(tile_coords)
    print_results("Rare Cluster Types (smallest clusters)", rare, ["cluster", "size", "year"])

    entropy = find_high_entropy(tile_coords)
    print_results("High Entropy Regions", entropy, ["entropy"])

    console.print(
        "\n[bold yellow]═══ Suggested INTERESTING_POINTS for app/src/main.ts ═══[/bold yellow]\n"
    )

    console.print("// --- Non-polar temporal hotspots ---")
    for r in temporal[:5]:
        console.print(
            f'  {{ name: "Change @({r["lat"]:.1f},{r["lon"]:.1f})", '
            f'tag: "temporal Δ={r["change_score"]}", category: "temporal", '
            f"bbox: {fmt_bbox(r['lat'], r['lon'])} }},"
        )

    console.print("\n// --- k-NN isolation outliers (truly weird) ---")
    for r in outliers[:5]:
        console.print(
            f'  {{ name: "Isolated @({r["lat"]:.1f},{r["lon"]:.1f})", '
            f'tag: "knn={r["isolation"]:.1f}", category: "outlier", '
            f"bbox: {fmt_bbox(r['lat'], r['lon'])} }},"
        )

    console.print("\n// --- Rare cluster types ---")
    for r in rare[:8]:
        console.print(
            f'  {{ name: "Rare #{r["cluster"]} @({r["lat"]:.1f},{r["lon"]:.1f})", '
            f'tag: "n={r["size"]}", category: "cluster", '
            f"bbox: {fmt_bbox(r['lat'], r['lon'])} }},"
        )

    console.print("\n// --- Inland diverse tiles ---")
    for r in diverse[:5]:
        console.print(
            f'  {{ name: "Diverse @({r["lat"]:.1f},{r["lon"]:.1f})", '
            f'tag: "var={r["variance"]:.1f}", category: "diverse", '
            f"bbox: {fmt_bbox(r['lat'], r['lon'])} }},"
        )

    console.print("\n// --- High entropy ---")
    for r in entropy[:5]:
        console.print(
            f'  {{ name: "Entropy @({r["lat"]:.1f},{r["lon"]:.1f})", '
            f'tag: "H={r["entropy"]:.4f}", category: "entropy", '
            f"bbox: {fmt_bbox(r['lat'], r['lon'])} }},"
        )


if __name__ == "__main__":
    main()
