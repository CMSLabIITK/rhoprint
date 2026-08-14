"""
rhoprint.batch
---------------
Batch-process a directory of materials (each a subdirectory containing
a CHGCAR file), computing all grid-level functionals for each and
writing the results to a CSV.

This replaces the original ``run_all.py``. The key functional change
from the original is that each material's CHGCAR is now parsed exactly
once (via :func:`rhoprint.functionals.registry.compute_all_grid_functionals`)
instead of three times (once each for zeta, Laplacian, and moments).
"""

import traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Union

import pandas as pd

from .functionals.registry import compute_all_grid_functionals
from .full import compute_all_features


def _process_one(args) -> dict:
    """Process a single material directory. Returns a result dict."""
    material_id, chgcar_path, laplacian_method = args
    result = {"material_id": material_id}

    try:
        result.update(compute_all_grid_functionals(
            chgcar_path, laplacian_method=laplacian_method))
        result["error"] = None
    except Exception:
        result["error"] = traceback.format_exc()[-300:]

    return result


def find_chgcar_jobs(data_dir: Union[str, Path]):
    """
    Find all (material_id, chgcar_path) pairs under data_dir, where
    material_id is the subdirectory name and each subdirectory is
    expected to contain a file named ``CHGCAR``.
    """
    data_dir = Path(data_dir)
    jobs = []
    for subdir in sorted(data_dir.iterdir()):
        chgcar = subdir / "CHGCAR"
        if chgcar.exists():
            jobs.append((subdir.name, str(chgcar)))
    return jobs


def run_batch(data_dir: Union[str, Path],
              out_csv: Union[str, Path] = "results/functionals.csv",
              n_workers: int = 4,
              resume: bool = False,
              laplacian_method: str = "metric_tensor",
              progress_every: int = 100) -> pd.DataFrame:
    """
    Compute all grid-level functionals for every material under
    ``data_dir`` and write the combined results to ``out_csv``.

    Parameters
    ----------
    data_dir : str or Path
        Root directory; each subdirectory should contain a CHGCAR file.
        Directory name is used as material_id.
    out_csv : str or Path
        Output CSV path.
    n_workers : int
        Number of parallel worker processes.
    resume : bool
        Skip materials already present in ``out_csv``.
    laplacian_method : "metric_tensor" (default) or "diagonal"
        See :mod:`rhoprint.functionals.laplacian`. Use "diagonal" only
        to reproduce results computed before the non-orthogonal-lattice
        fix.
    progress_every : int
        Print a progress line every N completed materials.

    Returns
    -------
    DataFrame of all results (existing + new, if resuming).
    """
    data_dir = Path(data_dir)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    all_jobs = find_chgcar_jobs(data_dir)
    print(f"Found {len(all_jobs)} CHGCAR files in {data_dir}")

    done_ids = set()
    if resume and out_csv.exists():
        existing = pd.read_csv(out_csv)
        done_ids = set(existing["material_id"].tolist())
        print(f"Resuming: {len(done_ids)} already done, "
              f"{len(all_jobs) - len(done_ids)} remaining")

    jobs = [(mid, path, laplacian_method)
            for mid, path in all_jobs if mid not in done_ids]

    if not jobs:
        print("Nothing to do.")
        return pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame()

    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_process_one, job): job for job in jobs}
        for i, future in enumerate(as_completed(futures)):
            res = future.result()
            results.append(res)
            if (i + 1) % progress_every == 0:
                print(f"  Processed {i + 1}/{len(jobs)} ...")

    df_new = pd.DataFrame(results)

    if resume and out_csv.exists():
        df_old = pd.read_csv(out_csv)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new

    df.to_csv(out_csv, index=False)
    n_ok = df["error"].isna().sum()
    n_err = df["error"].notna().sum()
    print(f"\nDone. {n_ok} succeeded, {n_err} failed.")
    print(f"Saved to {out_csv}")

    return df


def _process_one_full(args) -> dict:
    """Process a single material directory with the full descriptor set."""
    material_id, chgcar_path, laplacian_method = args
    try:
        row = compute_all_features(
            material_id, chgcar_path, laplacian_method=laplacian_method)
        row["error"] = None
        return row
    except Exception:
        return {
            "material_id": material_id,
            "error": traceback.format_exc()[-300:],
        }


def run_full_batch(data_dir: Union[str, Path],
                    out_csv: Union[str, Path] = "results/full_functionals.csv",
                    n_workers: int = 4,
                    resume: bool = False,
                    laplacian_method: str = "metric_tensor",
                    progress_every: int = 100) -> pd.DataFrame:
    """
    Like :func:`run_batch`, but computes the FULL descriptor set per
    material (grid functionals + structural metadata + compositional
    descriptors + derived ratios) via
    :func:`rhoprint.full.compute_all_features`, matching your target
    CSV schema (minus "split" -- assign that separately).

    material_id (the subdirectory name) must be of the form
    "<formula>_<space_group_number>", e.g. "NdP_225" -- see
    rhoprint.metadata for how it's parsed.

    Parameters are otherwise identical to :func:`run_batch`.
    """
    data_dir = Path(data_dir)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    all_jobs = find_chgcar_jobs(data_dir)
    print(f"Found {len(all_jobs)} CHGCAR files in {data_dir}")

    done_ids = set()
    if resume and out_csv.exists():
        existing = pd.read_csv(out_csv)
        done_ids = set(existing["material_id"].tolist())
        print(f"Resuming: {len(done_ids)} already done, "
              f"{len(all_jobs) - len(done_ids)} remaining")

    jobs = [(mid, path, laplacian_method)
            for mid, path in all_jobs if mid not in done_ids]

    if not jobs:
        print("Nothing to do.")
        return pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame()

    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_process_one_full, job): job for job in jobs}
        for i, future in enumerate(as_completed(futures)):
            res = future.result()
            results.append(res)
            if (i + 1) % progress_every == 0:
                print(f"  Processed {i + 1}/{len(jobs)} ...")

    df_new = pd.DataFrame(results)

    if resume and out_csv.exists():
        df_old = pd.read_csv(out_csv)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new

    df.to_csv(out_csv, index=False)
    n_ok = df["error"].isna().sum()
    n_err = df["error"].notna().sum()
    print(f"\nDone. {n_ok} succeeded, {n_err} failed.")
    print(f"Saved to {out_csv}")

    return df
