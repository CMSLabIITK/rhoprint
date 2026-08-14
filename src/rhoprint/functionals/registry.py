"""
rhoprint.functionals.registry
--------------------------------
Combines the grid-level functionals (zeta, Laplacian stats, radial
moments) into a single call per material, parsing the CHGCAR only once.

This replaces the old ``run_all.py`` pattern of calling
``compute_zeta(path)``, ``compute_laplacian_stats(path)``, and
``compute_radial_moments(path)`` independently -- each of which used
to re-parse the same file from disk.
"""

from typing import Callable, Union

from ..io.chgcar_reader import ensure_parsed, PathLike
from .zeta import compute_zeta
from .laplacian import compute_laplacian_stats
from .moments import compute_radial_moments

# name -> callable(parsed_data: dict) -> dict | float
# Each callable receives an already-parsed CHGCAR dict.
GRID_FUNCTIONAL_REGISTRY: dict[str, Callable] = {
    "zeta": compute_zeta,
    "laplacian": compute_laplacian_stats,
    "moments": compute_radial_moments,
}


def compute_all_grid_functionals(chgcar: Union[PathLike, dict],
                                  laplacian_method: str = "metric_tensor") -> dict:
    """
    Compute zeta, Laplacian statistics, and radial moments for one
    material, parsing the CHGCAR file exactly once.

    Parameters
    ----------
    chgcar : str, Path, or dict
        CHGCAR filepath, or an already-parsed dict.
    laplacian_method : "metric_tensor" (default) or "diagonal"
        Passed through to the Laplacian functional. "metric_tensor" is
        exact for any lattice; "diagonal" is the legacy approximation,
        kept only to reproduce results computed before that fix. See
        :mod:`rhoprint.functionals.laplacian`.

    Returns
    -------
    dict, flattened union of all grid-level functional outputs:
        zeta, laplacian_mean, laplacian_std, laplacian_neg_fraction,
        laplacian_pos_fraction, laplacian_min, laplacian_max,
        total_charge, moment_1, moment_2, moment_3, radial_variance,
        interstitial_fraction, bond_fraction
    """
    data = ensure_parsed(chgcar)

    result = {"zeta": compute_zeta(data)}
    result.update(compute_laplacian_stats(data, method=laplacian_method))
    result.update(compute_radial_moments(data))
    return result
