"""
rhoprint.full
--------------
Combines every descriptor group -- grid-level functionals, structural
metadata (crystal system / space group), compositional descriptors,
and derived cross-term ratios -- into a single row matching your
target descriptor schema.

Per your confirmation:
    - "split" (train/val/test) is NOT computed here -- assign it
      separately however your training pipeline partitions materials.
    - "is_f_block" and "has_f_block" are the same quantity; both
      columns are populated with the identical value.
"""

from typing import Union

import pandas as pd

from .io.chgcar_reader import ensure_parsed, PathLike
from .functionals.registry import compute_all_grid_functionals
from .functionals.composition import compute_compositional_features
from .functionals.derived import compute_derived_ratios
from .metadata import compute_structural_metadata

# Final column order matching your reference CSV, minus "split"
# (not computed here -- assign separately) and with is_f_block/
# has_f_block both present as identical values.
FULL_SCHEMA_COLUMNS = [
    "material_id",
    "crystal_system", "space_group_number", "is_f_block",
    "zeta", "laplacian_neg_fraction", "laplacian_std",
    "moment_1", "moment_2", "radial_variance",
    "interstitial_fraction", "bond_fraction", "total_charge",
    "mean_mass", "mean_elneg", "mean_vec",
    "bond_int_ratio", "radial_cv", "zeta_over_rvar", "lnf_x_m1",
    "fint_over_lnf", "moment_ratio", "vec_x_lnf", "vec_over_rvar",
    "bond_over_lnf", "elneg_x_lnf", "charge_per_m1", "lap_concentration",
    "sqrt_zeta", "log_lnf",
    "max_mass", "mass_range", "elneg_diff", "max_vec",
    "mean_radius", "radius_diff", "n_elements", "ionicity",
    "mean_period", "has_f_block", "crystal_system_int",
]


def compute_all_features(material_id: str, chgcar: Union[PathLike, dict],
                          laplacian_method: str = "metric_tensor") -> dict:
    """
    Compute every descriptor in the target schema for one material.

    Parameters
    ----------
    material_id : str
        Of the form "<formula>_<space_group_number>", e.g. "NdP_225".
        The formula and space group are parsed from this string (see
        rhoprint.metadata).
    chgcar : str, Path, or dict
        CHGCAR filepath or already-parsed dict.
    laplacian_method : "metric_tensor" (default) or "diagonal"
        See rhoprint.functionals.laplacian.

    Returns
    -------
    dict with keys matching FULL_SCHEMA_COLUMNS (no "split" column --
    assign your train/val/test partition separately).
    """
    data = ensure_parsed(chgcar)

    row = {"material_id": material_id}
    row.update(compute_structural_metadata(material_id))
    row.update(compute_all_grid_functionals(data, laplacian_method=laplacian_method))

    comp = compute_compositional_features(row["formula"])
    row.update(comp)
    row["is_f_block"] = comp["has_f_block"]  # same quantity, two column names

    df = compute_derived_ratios(pd.DataFrame([row]))
    result = df.iloc[0].to_dict()

    # formula was only needed to compute compositional features; drop it
    # unless you want it kept, in which case remove this line
    result.pop("formula", None)

    return result
