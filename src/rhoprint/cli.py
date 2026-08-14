"""
rhoprint.cli
-------------
Command-line interface.

Usage:
    rhoprint compute CHGCAR_PATH [--full]
    rhoprint batch --data_dir DIR --out_csv results/functionals.csv --n_workers 4 --resume [--full]

With --full, computes the complete descriptor set (grid functionals +
structural metadata + compositional descriptors + derived ratios) via
rhoprint.full.compute_all_features, matching the target CSV schema.
Without --full, computes only the grid-level functionals (zeta,
Laplacian stats, radial moments).

--full requires a material_id of the form "<formula>_<space_group>",
e.g. "NdP_225" -- for `compute`, this is taken from the CHGCAR's
parent directory name; for `batch`, from each subdirectory name.
"""

import argparse
import json
import sys
from pathlib import Path

from .functionals.registry import compute_all_grid_functionals
from .full import compute_all_features
from .batch import run_batch, run_full_batch


def _cmd_compute(args):
    if args.full:
        material_id = Path(args.chgcar_path).resolve().parent.name
        result = compute_all_features(
            material_id, args.chgcar_path,
            laplacian_method=args.laplacian_method)
    else:
        result = compute_all_grid_functionals(
            args.chgcar_path, laplacian_method=args.laplacian_method)
    print(json.dumps(result, indent=2, default=str))


def _cmd_batch(args):
    if args.full:
        run_full_batch(
            data_dir=args.data_dir,
            out_csv=args.out_csv,
            n_workers=args.n_workers,
            resume=args.resume,
            laplacian_method=args.laplacian_method,
        )
    else:
        run_batch(
            data_dir=args.data_dir,
            out_csv=args.out_csv,
            n_workers=args.n_workers,
            resume=args.resume,
            laplacian_method=args.laplacian_method,
        )


def main(argv=None):
    parser = argparse.ArgumentParser(prog="rhoprint")
    sub = parser.add_subparsers(dest="command", required=True)

    p_compute = sub.add_parser(
        "compute", help="Compute functionals for one CHGCAR file")
    p_compute.add_argument("chgcar_path")
    p_compute.add_argument(
        "--full", action="store_true",
        help="Compute the full descriptor set (metadata + composition + "
             "derived ratios), not just grid functionals")
    p_compute.add_argument(
        "--laplacian-method", dest="laplacian_method",
        choices=["metric_tensor", "diagonal"], default="metric_tensor",
        help="Laplacian computation method (default: metric_tensor, exact "
             "for any lattice; diagonal is the legacy approximation)")
    p_compute.set_defaults(func=_cmd_compute)

    p_batch = sub.add_parser(
        "batch", help="Compute functionals for a directory of materials")
    p_batch.add_argument("--data_dir", required=True,
                          help="Root dir; each subdir has a CHGCAR file")
    p_batch.add_argument("--out_csv", default=None,
                          help="Output CSV path (default: results/functionals.csv, "
                               "or results/full_functionals.csv with --full)")
    p_batch.add_argument("--n_workers", type=int, default=4)
    p_batch.add_argument("--resume", action="store_true")
    p_batch.add_argument(
        "--full", action="store_true",
        help="Compute the full descriptor set for every material")
    p_batch.add_argument(
        "--laplacian-method", dest="laplacian_method",
        choices=["metric_tensor", "diagonal"], default="metric_tensor",
        help="Laplacian computation method (default: metric_tensor)")
    p_batch.set_defaults(func=_cmd_batch)

    args = parser.parse_args(argv)
    if args.command == "batch" and args.out_csv is None:
        args.out_csv = "results/full_functionals.csv" if args.full else "results/functionals.csv"
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
