"""
rhoprint
---------
Physically interpretable descriptors ("fingerprints") extracted from
VASP charge-density (CHGCAR) files, for materials-informatics and ML
interpretability studies.

Developed at CMS Lab, IIT Kanpur, alongside work on charge-density-
informed ML models for elastic-property prediction.

Quickstart
----------
>>> from rhoprint import parse_chgcar, compute_all_grid_functionals
>>> data = parse_chgcar("CHGCAR")
>>> compute_all_grid_functionals(data)
{'zeta': ..., 'laplacian_mean': ..., ...}
"""

from .io import parse_chgcar, get_voxel_coords
from .functionals import (
    compute_zeta,
    compute_laplacian_grid,
    compute_laplacian_stats,
    compute_radial_moments,
    compute_all_grid_functionals,
    compute_derived_ratios,
    compute_compositional_features,
    compute_compositional_features_batch,
)
from .metadata import (
    parse_material_id,
    crystal_system_from_space_group,
    compute_structural_metadata,
)
from .full import compute_all_features, FULL_SCHEMA_COLUMNS
from .batch import run_batch, run_full_batch

__version__ = "0.1.0"

__all__ = [
    "parse_chgcar",
    "get_voxel_coords",
    "compute_zeta",
    "compute_laplacian_grid",
    "compute_laplacian_stats",
    "compute_radial_moments",
    "compute_all_grid_functionals",
    "compute_derived_ratios",
    "compute_compositional_features",
    "compute_compositional_features_batch",
    "parse_material_id",
    "crystal_system_from_space_group",
    "compute_structural_metadata",
    "compute_all_features",
    "FULL_SCHEMA_COLUMNS",
    "run_batch",
    "run_full_batch",
    "__version__",
]
