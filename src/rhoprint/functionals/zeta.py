"""
rhoprint.functionals.zeta
---------------------------
Angular variance metric ζ (ChargE3Net Eq. 3).

    ζ(G) = 1 - [ Σ_{g_k in G} |∇ρ(g_k) . r̂_ki| ]
               / [ Σ_{g_k in G} ||∇ρ(g_k)|| ]

where g_k is a voxel, r̂_ki is the unit vector from voxel g_k to its
nearest atom i, and ∇ρ is the charge-density gradient at g_k.

Interpretation
--------------
ζ -> 0 : density gradient points directly toward/away from the nearest
         atom (purely radial character; ionic/metallic).
ζ -> 1 : density gradient is perpendicular to the nearest-atom direction
         (high angular variance; covalent character).
"""

from typing import Union

import numpy as np

from ..io.chgcar_reader import ensure_parsed, get_voxel_coords, PathLike
from ..grid_utils import nearest_atom_info


def compute_gradient(rho: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    """
    ∇ρ at every voxel via central differences, converted to Cartesian.

    Returns
    -------
    ndarray of shape (nx, ny, nz, 3)
    """
    nx, ny, nz = rho.shape

    drho_di = np.gradient(rho, axis=0)
    drho_dj = np.gradient(rho, axis=1)
    drho_dk = np.gradient(rho, axis=2)

    # J = lattice / grid_size (Jacobian from fractional index to Cartesian)
    J = lattice / np.array([[nx], [ny], [nz]])
    J_inv_T = np.linalg.inv(J).T

    grad_frac = np.stack([drho_di / nx,
                           drho_dj / ny,
                           drho_dk / nz], axis=-1)

    grad_cart = grad_frac @ J_inv_T
    return grad_cart


def compute_zeta(chgcar: Union[PathLike, dict]) -> float:
    """
    Compute the scalar angular-variance metric ζ for one material.

    Parameters
    ----------
    chgcar : str, Path, or dict
        Either a path to a CHGCAR file, or an already-parsed dict from
        :func:`rhoprint.io.parse_chgcar` (use the latter in batch jobs
        that also need other functionals, to avoid re-parsing the file).

    Returns
    -------
    float, ζ in [0, 1], or NaN if the gradient is degenerate everywhere.
    """
    data = ensure_parsed(chgcar)
    rho = data["rho"]
    lattice = data["lattice"]
    positions = data["positions"]

    grad = compute_gradient(rho, lattice)
    grad_flat = grad.reshape(-1, 3)

    voxel_coords = get_voxel_coords(data)
    _, r_hat = nearest_atom_info(voxel_coords, positions, lattice)

    dot = np.abs(np.sum(grad_flat * r_hat, axis=1))
    grad_norm = np.linalg.norm(grad_flat, axis=1)

    mask = grad_norm > 1e-10
    if mask.sum() == 0:
        return float("nan")

    zeta = 1.0 - dot[mask].sum() / grad_norm[mask].sum()
    return float(zeta)


if __name__ == "__main__":
    import sys
    z = compute_zeta(sys.argv[1])
    print(f"zeta = {z:.6f}")
