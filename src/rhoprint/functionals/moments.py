"""
rhoprint.functionals.moments
-------------------------------
Radial charge-density moments and bonding-region fractions.

For each voxel g_k, let r_k be its distance to the nearest atom. We
compute weighted moments of rho as a function of r:

    moment_n = Σ_k rho(g_k) * r_k^n * dV / Σ_k rho(g_k) * dV

where dV = volume / (nx*ny*nz) is the voxel volume.

    moment_0 : total electron count (sanity check ~= n_electrons)
    moment_1 : mean distance of electron density from nearest atom
               (small -> density concentrated near atoms -> ionic/metallic)
               (large -> density spread between atoms -> covalent)
    moment_2 : second moment (variance proxy of radial distribution)
    moment_3 : third moment (skewness proxy)

We additionally report:
    interstitial_fraction : fraction of electron density farther than
                             1.5 Angstrom from the nearest atom
    bond_fraction          : fraction of electron density between 0.8
                              and 1.5 Angstrom from the nearest atom
                              (bonding region)
"""

from typing import Union

import numpy as np

from ..io.chgcar_reader import ensure_parsed, get_voxel_coords, PathLike
from ..grid_utils import nearest_atom_info

INTERSTITIAL_CUTOFF_ANGSTROM = 1.5
BOND_REGION_ANGSTROM = (0.8, 1.5)


def compute_radial_moments(chgcar: Union[PathLike, dict]) -> dict:
    """
    Radial moment statistics for one material.

    Parameters
    ----------
    chgcar : str, Path, or dict
        CHGCAR filepath, or an already-parsed dict.

    Returns
    -------
    dict with keys: total_charge, moment_1, moment_2, moment_3,
    radial_variance, interstitial_fraction, bond_fraction
    """
    data = ensure_parsed(chgcar)
    rho = data["rho"]
    lattice = data["lattice"]
    positions = data["positions"]
    volume = data["volume"]
    nx, ny, nz = data["nx"], data["ny"], data["nz"]

    n_voxels = nx * ny * nz
    dV = volume / n_voxels

    rho_flat = rho.flatten()
    voxel_coords = get_voxel_coords(data)
    r, _ = nearest_atom_info(voxel_coords, positions, lattice)

    total_charge = float(np.sum(rho_flat) * dV)

    w = rho_flat * dV
    moment_1 = float(np.sum(w * r) / total_charge)
    moment_2 = float(np.sum(w * r ** 2) / total_charge)
    moment_3 = float(np.sum(w * r ** 3) / total_charge)

    radial_variance = moment_2 - moment_1 ** 2

    interstitial_mask = r > INTERSTITIAL_CUTOFF_ANGSTROM
    interstitial_fraction = float(
        np.sum(rho_flat[interstitial_mask]) / np.sum(rho_flat)
    )

    lo, hi = BOND_REGION_ANGSTROM
    bond_mask = (r > lo) & (r <= hi)
    bond_fraction = float(np.sum(rho_flat[bond_mask]) / np.sum(rho_flat))

    return {
        "total_charge": total_charge,
        "moment_1": moment_1,
        "moment_2": moment_2,
        "moment_3": moment_3,
        "radial_variance": radial_variance,
        "interstitial_fraction": interstitial_fraction,
        "bond_fraction": bond_fraction,
    }


if __name__ == "__main__":
    import sys
    import json
    stats = compute_radial_moments(sys.argv[1])
    print(json.dumps(stats, indent=2))
