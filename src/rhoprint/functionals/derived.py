"""
rhoprint.functionals.derived
-------------------------------
Derived cross-term descriptors computed from the raw grid-level
functionals (see :mod:`rhoprint.functionals.registry`).

These operate on a pandas DataFrame where each row is one material and
columns already contain the raw functionals (``bond_fraction``,
``interstitial_fraction``, ``radial_variance``, ``moment_1``,
``moment_2``, ``zeta``, ``laplacian_neg_fraction``, ``laplacian_std``,
``mean_vec``, ``mean_elneg``, ``total_charge``) -- typically produced by
running :func:`rhoprint.functionals.registry.compute_all_grid_functionals`
over a dataset and merging in compositional descriptors from
:mod:`rhoprint.functionals.composition`.

A small constant (1e-8) is added to denominators throughout to avoid
division-by-zero on near-zero functional values; this matches the
original script's behavior.
"""

import numpy as np
import pandas as pd

_EPS = 1e-8

REQUIRED_COLUMNS = [
    "bond_fraction", "interstitial_fraction", "radial_variance",
    "moment_1", "moment_2", "zeta", "laplacian_neg_fraction",
    "laplacian_std", "mean_vec", "mean_elneg", "total_charge",
]


def compute_derived_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived cross-term ratio/product columns to a DataFrame of raw
    functionals. Returns a new DataFrame (input is not modified).

    Missing required columns raise a KeyError naming the missing
    column, rather than silently producing NaN columns.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(
            f"compute_derived_ratios: missing required column(s) {missing}. "
            "Run rhoprint.functionals.registry.compute_all_grid_functionals "
            "and rhoprint.functionals.composition.compute_compositional_features "
            "first and merge their outputs into this DataFrame."
        )

    out = df.copy()

    out["bond_int_ratio"] = out["bond_fraction"] / (out["interstitial_fraction"] + _EPS)
    out["radial_cv"] = np.sqrt(out["radial_variance"].clip(lower=0)) / (out["moment_1"] + _EPS)
    out["zeta_over_rvar"] = out["zeta"] / (out["radial_variance"] + _EPS)
    out["lnf_x_m1"] = out["laplacian_neg_fraction"] * out["moment_1"]
    out["fint_over_lnf"] = out["interstitial_fraction"] / (out["laplacian_neg_fraction"] + _EPS)
    out["moment_ratio"] = out["moment_1"] / (out["moment_2"] + _EPS)
    out["vec_x_lnf"] = out["mean_vec"] * out["laplacian_neg_fraction"]
    out["vec_over_rvar"] = out["mean_vec"] / (out["radial_variance"] + _EPS)
    out["bond_over_lnf"] = out["bond_fraction"] / (out["laplacian_neg_fraction"] + _EPS)
    out["elneg_x_lnf"] = out["mean_elneg"] * out["laplacian_neg_fraction"]
    out["charge_per_m1"] = out["total_charge"] / (out["moment_1"] + _EPS)
    out["lap_concentration"] = out["laplacian_neg_fraction"] / (out["laplacian_std"] + _EPS)
    out["sqrt_zeta"] = np.sqrt(out["zeta"].clip(lower=0))
    out["log_lnf"] = np.log(out["laplacian_neg_fraction"].clip(lower=1e-6))

    return out
