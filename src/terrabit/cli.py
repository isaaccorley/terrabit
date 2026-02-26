"""CLI entrypoints for estimate-id and compress."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, cast

from terrabit.pipeline import (
    run_compress,
    run_estimate_id,
    run_full_pipeline,
    save_report,
)

if TYPE_CHECKING:
    from terrabit.compression import CompressionMethod


def _serialize(obj: object) -> object:
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    if isinstance(obj, tuple):
        return list(obj)
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize(x) for x in obj]
    return str(obj)


def cmd_estimate_id(args: argparse.Namespace) -> int:  # noqa: D103
    path = args.path
    report = run_estimate_id(
        path,
        n_subsample=args.n_subsample,
        batch_size=args.batch_size,
        embedding_col=args.embedding_col,
        l2_norm=args.l2_norm,
        center=args.center,
        seed=args.seed,
    )
    if args.output:
        with Path(args.output).open("w") as f:
            json.dump(_serialize(report), f, indent=2)
    else:
        print(json.dumps(_serialize(report), indent=2))  # noqa: T201
    return 0


def cmd_compress(args: argparse.Namespace) -> int:  # noqa: D103
    dims = tuple(int(x) for x in args.target_dim.split(","))
    results = run_compress(
        args.path,
        args.output,
        args.method,
        dims,
        batch_size=args.batch_size,
        embedding_col=args.embedding_col,
        seed=args.seed,
    )
    if args.report:
        with Path(args.report).open("w") as f:
            json.dump(_serialize({"compression_results": results}), f, indent=2)
    for r in results:
        print(f"Wrote {r['output_path']}: n_rows={r['n_rows']}")  # noqa: T201
    return 0


def cmd_full(args: argparse.Namespace) -> int:  # noqa: D103
    target_dims: tuple[int, ...] | str = "auto"
    if args.target_dim:
        target_dims = tuple(int(x) for x in args.target_dim.split(","))
    methods = cast(
        "tuple[CompressionMethod, ...]",
        tuple(m.strip() for m in args.methods.split(",")),
    )
    report = run_full_pipeline(
        args.path,
        args.output,
        n_subsample=args.n_subsample,
        batch_size=args.batch_size,
        embedding_col=args.embedding_col,
        methods=methods,
        target_dims=target_dims,
        run_post_compression_id_analysis=not args.skip_post_id,
        seed=args.seed,
    )
    if args.report:
        save_report(report, args.report)
        print(f"Report saved to {args.report}")  # noqa: T201
    return 0


def main() -> int:  # noqa: D103
    parser = argparse.ArgumentParser(prog="terrabit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    est = subparsers.add_parser("estimate-id", help="Estimate intrinsic dimension")
    est.add_argument("path", help="Path to parquet embeddings directory")
    est.add_argument("--output", "-o", help="Write report JSON to file")
    est.add_argument("--n-subsample", type=int, default=15_000)
    est.add_argument("--batch-size", type=int, default=50_000)
    est.add_argument("--embedding-col", default="embedding")
    est.add_argument("--l2-norm", action="store_true")
    est.add_argument("--center", action="store_true")
    est.add_argument("--seed", type=int, default=None)
    est.set_defaults(func=cmd_estimate_id)

    comp = subparsers.add_parser("compress", help="Compress embeddings")
    comp.add_argument("path", help="Path to parquet embeddings directory")
    comp.add_argument("output", help="Output directory for compressed parquet")
    comp.add_argument(
        "--method",
        choices=("ipca", "rp", "pca_whitening"),
        default="ipca",
    )
    comp.add_argument(
        "--target-dim",
        required=True,
        help="Target dimension(s), comma-separated (e.g. 64,128)",
    )
    comp.add_argument("--batch-size", type=int, default=50_000)
    comp.add_argument("--embedding-col", default="embedding")
    comp.add_argument("--seed", type=int, default=None)
    comp.add_argument("--report", help="Write compression report JSON")
    comp.set_defaults(func=cmd_compress)

    full = subparsers.add_parser(
        "full",
        help="Run full pipeline: estimate-id + compress",
    )
    full.add_argument("path", help="Path to parquet embeddings directory")
    full.add_argument("output", help="Output directory for compressed parquet")
    full.add_argument(
        "--target-dim",
        help="Target dims comma-separated; default: auto from ID",
    )
    full.add_argument("--methods", default="ipca,rp,pca_whitening")
    full.add_argument("--n-subsample", type=int, default=15_000)
    full.add_argument("--batch-size", type=int, default=50_000)
    full.add_argument("--embedding-col", default="embedding")
    full.add_argument(
        "--skip-post-id",
        action="store_true",
        help="Skip post-compression ID",
    )
    full.add_argument("--seed", type=int, default=None)
    full.add_argument("--report", help="Write full report JSON")
    full.set_defaults(func=cmd_full)

    args = parser.parse_args()
    return args.func(args)
