"""
Shared pytest fixtures.

Real CHGCAR files are large (tens of MB) and require VASP/DFT to
generate, so tests use a small synthetic CHGCAR: a single-atom cubic
cell with a Gaussian charge density centered on the atom. This is
enough to sanity-check parsing and the functionals' basic behavior
(e.g. zeta should be small for a spherically symmetric density, since
the gradient points radially toward the atom everywhere).
"""

import numpy as np
import pytest


def _write_synthetic_chgcar(path, n_grid=12, cell_length=5.0, n_electrons=8.0):
    """
    Write a minimal VASP5-format CHGCAR: single atom at the origin of a
    cubic cell, with an isotropic Gaussian charge density around it.
    """
    lattice = np.eye(3) * cell_length

    lines = []
    lines.append("synthetic single-atom test cell")
    lines.append("1.0")
    for row in lattice:
        lines.append(f"  {row[0]:.6f} {row[1]:.6f} {row[2]:.6f}")
    lines.append("X")
    lines.append("1")
    lines.append("Direct")
    lines.append("  0.000000 0.000000 0.000000")
    lines.append("")
    lines.append(f"  {n_grid} {n_grid} {n_grid}")

    # Build an isotropic Gaussian density (in fractional-index space,
    # using minimum image convention so it's periodic).
    idx = np.arange(n_grid)
    gx, gy, gz = np.meshgrid(idx, idx, idx, indexing="ij")
    frac = np.stack([gx, gy, gz], axis=-1) / n_grid  # (n,n,n,3)
    frac -= np.round(frac)  # wrap to [-0.5, 0.5)
    cart = frac * cell_length
    r2 = np.sum(cart ** 2, axis=-1)

    sigma = 0.6
    density_shape = np.exp(-r2 / (2 * sigma ** 2))

    volume = cell_length ** 3
    # VASP stores rho * volume; normalize so integral ~= n_electrons
    raw = density_shape / density_shape.sum() * n_electrons * (n_grid ** 3)

    flat = raw.flatten(order="F")  # CHGCAR is Fortran-ordered

    # write 5 values per line, matching typical VASP formatting
    data_lines = []
    for i in range(0, len(flat), 5):
        chunk = flat[i:i + 5]
        data_lines.append(" ".join(f"{v:.6E}" for v in chunk))

    content = "\n".join(lines) + "\n" + "\n".join(data_lines) + "\n"
    path.write_text(content)
    return {
        "lattice": lattice,
        "volume": volume,
        "n_grid": n_grid,
        "n_electrons": n_electrons,
    }


@pytest.fixture
def synthetic_chgcar(tmp_path):
    """Path to a small synthetic single-atom CHGCAR, plus known params."""
    chgcar_path = tmp_path / "CHGCAR"
    params = _write_synthetic_chgcar(chgcar_path)
    return chgcar_path, params
