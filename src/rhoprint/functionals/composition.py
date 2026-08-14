"""
rhoprint.functionals.composition
-----------------------------------
Compositional descriptors derived from a material's chemical formula
via pymatgen (elemental mass, electronegativity, oxidation state,
atomic radius, periodic-table period, f-block presence).

These are independent of the charge-density grid itself -- they only
need the formula -- but are included here because they were used
alongside the grid-level functionals as inputs to the elastic-property
models in the original analysis.
"""

import re
from typing import Iterable

import numpy as np
import pandas as pd
from pymatgen.core import Composition

from ..metadata import CRYSTAL_SYSTEM_INT

F_BLOCK = {
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy",
    "Ho", "Er", "Tm", "Yb", "Lu", "Ac", "Th", "Pa", "U", "Np",
    "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr",
}

COMPOSITIONAL_COLUMNS = [
    "mean_mass", "max_mass", "mass_range", "mean_elneg", "elneg_diff",
    "mean_vec", "max_vec", "mean_radius", "radius_diff", "n_elements",
    "ionicity", "mean_period", "has_f_block",
]

CRYSTAL_SYSTEM_MAP = CRYSTAL_SYSTEM_INT
"""
Alias for rhoprint.metadata.CRYSTAL_SYSTEM_INT (1=Triclinic ... 7=Cubic,
matching the reference PDF's descriptor #25).

NOTE: earlier versions of this module (and the original functionals.py
script) defined this independently with the OPPOSITE order
(Cubic=1 ... Triclinic=7). That was a bug relative to the documented
definition -- see rhoprint.metadata module docstring. If you have
existing computed CSVs with a crystal_system_int column, check which
convention they used before combining with new results.
"""


def _period_from_atomic_number(z: int) -> int:
    if z <= 2:
        return 1
    if z <= 10:
        return 2
    if z <= 18:
        return 3
    if z <= 36:
        return 4
    if z <= 54:
        return 5
    if z <= 86:
        return 6
    return 7


def compute_compositional_features(formula: str) -> dict:
    """
    Compositional descriptors for a single chemical formula.

    Parameters
    ----------
    formula : str
        A pymatgen-parseable chemical formula, e.g. "Fe2O3".

    Returns
    -------
    dict with keys in :data:`COMPOSITIONAL_COLUMNS`. All values are
    NaN if the formula could not be parsed by pymatgen.
    """
    row = {}
    try:
        comp = Composition(formula)
        els = list(comp.items())
        masses = [float(el.atomic_mass) for el, _ in els]
        amts = [float(a) for _, a in els]
        elneg_v = [float(el.X) for el, _ in els]
        vec_v = [float(el.max_oxidation_state) for el, _ in els]
        radii = [float(el.atomic_radius or 1.5) for el, _ in els]

        row["mean_mass"] = float(np.average(masses, weights=amts))
        row["max_mass"] = float(max(masses))
        row["mass_range"] = float(max(masses) - min(masses))
        row["mean_elneg"] = float(np.average(elneg_v, weights=amts))
        row["elneg_diff"] = float(max(elneg_v) - min(elneg_v))
        row["mean_vec"] = float(np.average(vec_v, weights=amts))
        row["max_vec"] = float(max(vec_v))
        row["mean_radius"] = float(np.average(radii, weights=amts))
        row["radius_diff"] = float(max(radii) - min(radii))
        row["n_elements"] = int(len(els))
        row["ionicity"] = float(1 - np.exp(-0.25 * row["elneg_diff"] ** 2))

        periods = [_period_from_atomic_number(el.Z) for el, _ in els]
        row["mean_period"] = float(np.average(periods, weights=amts))

        elements_in = re.findall(r"[A-Z][a-z]?", formula)
        row["has_f_block"] = int(any(e in F_BLOCK for e in elements_in))

    except Exception:
        for k in COMPOSITIONAL_COLUMNS:
            row[k] = np.nan

    return row


def compute_compositional_features_batch(formulas: Iterable[str],
                                          index=None) -> pd.DataFrame:
    """
    Vectorized (row-per-formula) version of
    :func:`compute_compositional_features`, matching the original
    ``functionals.py`` loop over a DataFrame column.

    Parameters
    ----------
    formulas : iterable of str
    index : optional, index to assign to the returned DataFrame

    Returns
    -------
    DataFrame with one row per formula, columns = COMPOSITIONAL_COLUMNS
    """
    rows = [compute_compositional_features(f) for f in formulas]
    return pd.DataFrame(rows, index=index)


def encode_crystal_system(series: pd.Series) -> pd.Series:
    """Map crystal-system strings (e.g. 'Cubic') to integer codes."""
    return series.map(CRYSTAL_SYSTEM_MAP)
