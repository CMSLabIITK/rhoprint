import numpy as np

from rhoprint.io import parse_chgcar, get_voxel_coords


def test_parse_chgcar_shapes(synthetic_chgcar):
    path, params = synthetic_chgcar
    data = parse_chgcar(path)

    n = params["n_grid"]
    assert data["rho"].shape == (n, n, n)
    assert data["nx"] == data["ny"] == data["nz"] == n
    assert data["n_atoms"] == 1
    assert np.allclose(data["lattice"], params["lattice"])
    assert np.isclose(data["volume"], params["volume"])


def test_parse_chgcar_electron_count_sanity(synthetic_chgcar):
    """moment_0 (total integrated charge) should recover n_electrons."""
    path, params = synthetic_chgcar
    data = parse_chgcar(path)

    dV = data["volume"] / (data["nx"] * data["ny"] * data["nz"])
    total_charge = float(np.sum(data["rho"]) * dV)

    assert np.isclose(total_charge, params["n_electrons"], rtol=1e-3)


def test_get_voxel_coords_shape(synthetic_chgcar):
    path, _ = synthetic_chgcar
    data = parse_chgcar(path)
    coords = get_voxel_coords(data)

    n_voxels = data["nx"] * data["ny"] * data["nz"]
    assert coords.shape == (n_voxels, 3)
