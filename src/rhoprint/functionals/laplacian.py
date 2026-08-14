"""
rhoprint.functionals.laplacian
--------------------------------
Grid-level Laplacian ∇²ρ of the charge density, used as a cheap proxy
for bonding character (no full QTAIM critical-point search).

Physical interpretation
------------------------
∇²ρ < 0 at a point -> charge concentration -> covalent bond character
∇²ρ > 0 at a point -> charge depletion    -> ionic / metallic character

Two computation methods are available via the ``method`` argument:

``"metric_tensor"`` (default, recommended)
    Exact for any lattice, orthogonal or not. Since the map from
    fractional grid coordinates u to Cartesian coordinates x is linear
    (x = u @ A, A = lattice matrix), the Laplacian in Cartesian space
    is:

        ∇²ρ = Σ_{m,n} G^{mn} * ∂²ρ/∂u_m∂u_n

    where G = A⁻¹ᵀ A⁻¹ is the contravariant metric tensor of the
    lattice. This includes the cross-axis (off-diagonal) second
    derivatives that a purely diagonal approximation drops, and uses
    the correct (possibly non-diagonal) scale factors even on the
    diagonal terms.

``"diagonal"`` (legacy, kept for backward compatibility only)
    The original approximation: sums independent second derivatives
    along each lattice axis, scaled by (|a_i| / n_i)^2. This is exact
    only for orthogonal cells (cubic, tetragonal, orthorhombic) and
    silently wrong for non-orthogonal cells (monoclinic, triclinic,
    some hexagonal settings) because it omits cross-axis terms and
    uses lattice-vector norms rather than the true metric tensor. Use
    this only if you need to exactly reproduce earlier results computed
    before this fix; a warning is raised if used on a non-orthogonal
    cell.
"""

import warnings
from typing import Union

import numpy as np

from ..io.chgcar_reader import ensure_parsed, PathLike
from ..grid_utils import is_orthogonal_lattice

_VALID_METHODS = ("metric_tensor", "diagonal")


def _laplacian_metric_tensor(rho: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    """Exact Laplacian via the full contravariant metric tensor."""
    nx, ny, nz = rho.shape
    n = np.array([nx, ny, nz], dtype=float)

    A_inv = np.linalg.inv(lattice)
    G = A_inv.T @ A_inv  # (3, 3) symmetric metric tensor, fractional-coord space

    # M[m, n] = G[m, n] * n_m * n_n folds the index-to-fractional-coordinate
    # chain-rule scaling (u_m = i_m / n_m) into the metric tensor once.
    M = G * np.outer(n, n)

    d = [np.gradient(rho, axis=ax) for ax in range(3)]

    laplacian = np.zeros_like(rho)
    for m in range(3):
        # diagonal term
        laplacian += M[m, m] * np.gradient(d[m], axis=m)
        for nidx in range(m + 1, 3):
            # symmetrize the two finite-difference orders of the mixed
            # partial so the result doesn't depend on differentiation order
            d2_mn = np.gradient(d[m], axis=nidx)
            d2_nm = np.gradient(d[nidx], axis=m)
            d2_sym = 0.5 * (d2_mn + d2_nm)
            laplacian += 2.0 * M[m, nidx] * d2_sym

    return laplacian


def _laplacian_diagonal(rho: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    """Legacy diagonal approximation. Exact only for orthogonal lattices."""
    nx, ny, nz = rho.shape

    a1_len = np.linalg.norm(lattice[0]) / nx
    a2_len = np.linalg.norm(lattice[1]) / ny
    a3_len = np.linalg.norm(lattice[2]) / nz

    d2_di2 = np.gradient(np.gradient(rho, axis=0), axis=0) / (a1_len ** 2)
    d2_dj2 = np.gradient(np.gradient(rho, axis=1), axis=1) / (a2_len ** 2)
    d2_dk2 = np.gradient(np.gradient(rho, axis=2), axis=2) / (a3_len ** 2)

    return d2_di2 + d2_dj2 + d2_dk2


def compute_laplacian_grid(rho: np.ndarray, lattice: np.ndarray,
                            method: str = "metric_tensor") -> np.ndarray:
    """
    ∇²ρ at every voxel. See module docstring for ``method`` options.

    Returns
    -------
    ndarray of shape (nx, ny, nz)
    """
    if method not in _VALID_METHODS:
        raise ValueError(f"method must be one of {_VALID_METHODS}, got {method!r}")

    if method == "diagonal":
        if not is_orthogonal_lattice(lattice):
            warnings.warn(
                "compute_laplacian_grid(method='diagonal'): lattice is "
                "non-orthogonal. This legacy method is only approximate "
                "for this cell -- use method='metric_tensor' (the default) "
                "for an exact result. See rhoprint.functionals.laplacian "
                "module docstring.",
                stacklevel=2,
            )
        return _laplacian_diagonal(rho, lattice)

    return _laplacian_metric_tensor(rho, lattice)


def compute_laplacian_stats(chgcar: Union[PathLike, dict],
                             method: str = "metric_tensor") -> dict:
    """
    Summary statistics of ∇²ρ for one material.

    Parameters
    ----------
    chgcar : str, Path, or dict
        CHGCAR filepath, or an already-parsed dict.
    method : "metric_tensor" (default) or "diagonal"
        See module docstring. Use "diagonal" only to reproduce results
        computed before this fix.

    Returns
    -------
    dict with keys: laplacian_mean, laplacian_std, laplacian_neg_fraction,
    laplacian_pos_fraction, laplacian_min, laplacian_max
    """
    data = ensure_parsed(chgcar)
    rho = data["rho"]
    lattice = data["lattice"]

    lap = compute_laplacian_grid(rho, lattice, method=method)
    lap_flat = lap.flatten()

    return {
        "laplacian_mean": float(np.mean(lap_flat)),
        "laplacian_std": float(np.std(lap_flat)),
        "laplacian_neg_fraction": float(np.mean(lap_flat < 0)),
        "laplacian_pos_fraction": float(np.mean(lap_flat > 0)),
        "laplacian_min": float(np.min(lap_flat)),
        "laplacian_max": float(np.max(lap_flat)),
    }


if __name__ == "__main__":
    import sys
    import json
    stats = compute_laplacian_stats(sys.argv[1])
    print(json.dumps(stats, indent=2))
