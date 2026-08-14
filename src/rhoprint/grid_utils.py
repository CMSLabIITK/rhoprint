"""
rhoprint.grid_utils
--------------------
Shared geometry helpers used by multiple functionals.

The original scripts computed "distance to nearest atom" (compute_moments.py)
and "unit vector to nearest atom" (compute_zeta.py) with two separate,
near-identical minimum-image-convention loops over atoms. Both quantities
come from the same nearest-atom search, so this module computes them
together in one pass to avoid duplicated O(n_voxels * n_atoms) work.
"""

import numpy as np


def nearest_atom_info(voxel_coords: np.ndarray,
                       positions: np.ndarray,
                       lattice: np.ndarray):
    """
    For each voxel, find the distance to and unit vector toward its
    nearest atom, using the minimum image convention for periodic
    boundary conditions.

    Parameters
    ----------
    voxel_coords : (n_voxels, 3) ndarray, Cartesian Angstrom
    positions    : (n_atoms, 3) ndarray, Cartesian Angstrom
    lattice      : (3, 3) ndarray, Angstrom

    Returns
    -------
    dist : (n_voxels,) ndarray
        Distance (Angstrom) from each voxel to its nearest atom.
    unit_vec : (n_voxels, 3) ndarray
        Unit vector from each voxel to its nearest atom.
    """
    n_voxels = voxel_coords.shape[0]
    lat_inv = np.linalg.inv(lattice)

    min_dist2 = np.full(n_voxels, np.inf)
    nearest_vec = np.zeros((n_voxels, 3))

    for atom_pos in positions:
        diff = voxel_coords - atom_pos[None, :]

        # minimum image convention
        diff_frac = diff @ lat_inv
        diff_frac -= np.round(diff_frac)
        diff_cart = diff_frac @ lattice

        dist2 = np.sum(diff_cart ** 2, axis=1)

        closer = dist2 < min_dist2
        min_dist2[closer] = dist2[closer]
        nearest_vec[closer] = diff_cart[closer]

    dist = np.sqrt(min_dist2)
    norms = np.maximum(dist[:, None], 1e-10)
    unit_vec = nearest_vec / norms

    return dist, unit_vec


def is_orthogonal_lattice(lattice: np.ndarray, tol: float = 1e-6) -> bool:
    """True if lattice vectors are mutually orthogonal (cubic/tetra/ortho)."""
    off_diag = lattice @ lattice.T - np.diag(np.diag(lattice @ lattice.T))
    scale = np.linalg.norm(lattice) ** 2
    return bool(np.max(np.abs(off_diag)) < tol * max(scale, 1.0))
