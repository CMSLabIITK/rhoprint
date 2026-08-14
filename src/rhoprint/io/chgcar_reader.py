"""
rhoprint.io.chgcar_reader
--------------------------
Reader for VASP CHGCAR files.

A CHGCAR file stores the self-consistent charge density on a real-space
grid, along with the lattice and atomic positions used to generate it.
This module parses that file into plain numpy arrays that the rest of
``rhoprint`` operates on.

Note on units: VASP stores rho * Volume in CHGCAR, not rho itself. This
reader divides by the cell volume so that ``rho`` is returned in
electrons / Angstrom^3, matching physical charge density.
"""

from pathlib import Path
from typing import Union

import numpy as np

PathLike = Union[str, Path]


def parse_chgcar(filepath: PathLike) -> dict:
    """
    Read a VASP CHGCAR file.

    Parameters
    ----------
    filepath : str or Path
        Path to a CHGCAR file (VASP 5 format: element symbols on their
        own line before the ion counts).

    Returns
    -------
    dict with keys:
        rho       : (nx, ny, nz) ndarray, electrons / Angstrom^3
        lattice   : (3, 3) ndarray, Angstrom
        positions : (n_atoms, 3) ndarray, Cartesian Angstrom
        volume    : float, Angstrom^3
        nx, ny, nz: int, grid dimensions
        n_atoms   : int
    """
    filepath = Path(filepath)
    with open(filepath, "r") as f:
        lines = f.readlines()

    # --- lattice vectors (lines 2-4, 0-indexed 1-3) ---
    scale = float(lines[1].strip())
    lattice = np.array([
        [float(x) for x in lines[2].split()],
        [float(x) for x in lines[3].split()],
        [float(x) for x in lines[4].split()],
    ]) * scale  # shape (3, 3)

    volume = abs(np.linalg.det(lattice))

    # --- atom types and counts ---
    species = lines[5].split()
    counts = [int(x) for x in lines[6].split()]
    n_atoms = sum(counts)

    # --- coordinate type and positions ---
    coord_line = lines[7].strip().lower()
    is_direct = coord_line.startswith("d")

    pos_frac = []
    for i in range(8, 8 + n_atoms):
        pos_frac.append([float(x) for x in lines[i].split()[:3]])
    pos_frac = np.array(pos_frac)  # (n_atoms, 3) fractional

    if is_direct:
        positions = pos_frac @ lattice  # Cartesian Angstrom
    else:
        positions = pos_frac  # already Cartesian

    # --- grid size ---
    grid_line_idx = 8 + n_atoms
    while lines[grid_line_idx].strip() == "":
        grid_line_idx += 1

    nx, ny, nz = [int(x) for x in lines[grid_line_idx].split()]

    # --- density values ---
    data_lines = lines[grid_line_idx + 1:]
    values = []
    for line in data_lines:
        if "augmentation" in line.lower():
            break
        values.extend([float(x) for x in line.split()])
        if len(values) >= nx * ny * nz:
            break

    values = np.array(values[:nx * ny * nz])

    # VASP stores rho * Volume -- divide to get actual charge density
    rho = values.reshape((nx, ny, nz), order="F") / volume

    return {
        "rho": rho,                 # (nx, ny, nz) electrons/Angstrom^3
        "lattice": lattice,         # (3, 3) Angstrom
        "positions": positions,     # (n_atoms, 3) Cartesian Angstrom
        "volume": volume,           # float Angstrom^3
        "nx": nx, "ny": ny, "nz": nz,
        "n_atoms": n_atoms,
        "species": species,
        "counts": counts,
        "source_path": str(filepath),
    }


def get_voxel_coords(data: dict) -> np.ndarray:
    """
    Cartesian coordinates of every voxel center.

    Parameters
    ----------
    data : dict
        Output of :func:`parse_chgcar`.

    Returns
    -------
    ndarray of shape (nx*ny*nz, 3)
    """
    nx, ny, nz = data["nx"], data["ny"], data["nz"]
    lattice = data["lattice"]

    ix = np.arange(nx) / nx
    iy = np.arange(ny) / ny
    iz = np.arange(nz) / nz

    gx, gy, gz = np.meshgrid(ix, iy, iz, indexing="ij")
    frac = np.stack([gx, gy, gz], axis=-1)  # (nx, ny, nz, 3)

    cart = frac @ lattice  # (nx, ny, nz, 3)
    return cart.reshape(-1, 3)


def ensure_parsed(chgcar: Union[PathLike, dict]) -> dict:
    """
    Accept either a CHGCAR filepath or an already-parsed dict and return
    a parsed dict either way.

    This lets every functional in ``rhoprint`` be called either
    standalone (``compute_zeta("CHGCAR")``) or as part of a batch where
    the file has already been parsed once upstream (avoiding redundant
    I/O across multiple functionals on the same material).
    """
    if isinstance(chgcar, dict):
        return chgcar
    return parse_chgcar(chgcar)
