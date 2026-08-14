import warnings

import numpy as np
import pytest

from rhoprint.io import parse_chgcar
from rhoprint.functionals import (
    compute_zeta,
    compute_laplacian_stats,
    compute_radial_moments,
    compute_all_grid_functionals,
)
from rhoprint.functionals.derived import compute_derived_ratios
from rhoprint.functionals.composition import compute_compositional_features
import pandas as pd


def test_compute_zeta_isotropic_density_is_small(synthetic_chgcar):
    """
    For a spherically symmetric (isotropic Gaussian) density centered
    on the single atom, the gradient points radially toward/away from
    that atom everywhere, so zeta should be close to 0.
    """
    path, _ = synthetic_chgcar
    zeta = compute_zeta(path)
    assert 0.0 <= zeta < 0.15


def test_compute_zeta_accepts_parsed_dict(synthetic_chgcar):
    path, _ = synthetic_chgcar
    data = parse_chgcar(path)
    zeta_from_path = compute_zeta(path)
    zeta_from_dict = compute_zeta(data)
    assert zeta_from_path == zeta_from_dict


def test_laplacian_metric_tensor_exact_on_nonorthogonal_lattice():
    """
    Analytic benchmark: rho(x) = x1 * x2 is a pure cross term, whose
    Hessian is off-diagonal only, so its Laplacian is exactly 0
    everywhere for ANY lattice (Laplacian is coordinate-invariant).

    (An isotropic function like |x|^2 is a poor test here: its Hessian
    is a multiple of the identity, so the diagonal-only legacy method
    accidentally reproduces the right trace even with cross terms
    dropped. x1*x2 has no such accident -- the cross-derivative terms
    the diagonal method omits are exactly what's needed to cancel to
    zero, so it's a clean discriminator between the two methods.)
    """
    from rhoprint.functionals.laplacian import compute_laplacian_grid

    n = 40
    # sheared / non-orthogonal lattice (monoclinic-like: a2 has an x-component)
    lattice = np.array([
        [6.0, 0.0, 0.0],
        [2.5, 6.0, 0.0],
        [0.0, 0.0, 6.0],
    ])

    idx = np.arange(n) / n
    gx, gy, gz = np.meshgrid(idx, idx, idx, indexing="ij")
    frac = np.stack([gx, gy, gz], axis=-1)
    cart = frac @ lattice
    rho = cart[..., 0] * cart[..., 1]  # true Laplacian = 0 everywhere

    lap_exact = compute_laplacian_grid(rho, lattice, method="metric_tensor")
    with pytest.warns(UserWarning, match="non-orthogonal"):
        lap_diag = compute_laplacian_grid(rho, lattice, method="diagonal")

    # check interior points only, away from finite-difference edge effects
    interior = lap_exact[5:-5, 5:-5, 5:-5]
    interior_diag = lap_diag[5:-5, 5:-5, 5:-5]

    assert np.allclose(interior, 0.0, atol=1e-8), (
        f"metric_tensor method should recover the analytic Laplacian of 0, "
        f"got mean={interior.mean():.6g}"
    )
    # the legacy diagonal method drops the cross-axis term needed to
    # cancel to zero on this sheared lattice -- this demonstrates the
    # bug that method="metric_tensor" fixes
    assert not np.allclose(interior_diag, 0.0, atol=0.05), (
        "diagonal method unexpectedly matched the analytic answer on a "
        "sheared lattice -- test lattice may not be non-orthogonal enough"
    )


def test_laplacian_methods_agree_on_orthogonal_lattice(synthetic_chgcar):
    """For an orthogonal lattice, metric_tensor and diagonal should agree
    (the metric tensor is diagonal in this case, so both formulas coincide)."""
    path, _ = synthetic_chgcar
    stats_exact = compute_laplacian_stats(path, method="metric_tensor")
    stats_diag = compute_laplacian_stats(path, method="diagonal")

    assert np.isclose(stats_exact["laplacian_mean"], stats_diag["laplacian_mean"])
    assert np.isclose(stats_exact["laplacian_std"], stats_diag["laplacian_std"])


def test_laplacian_diagonal_method_warns_on_nonorthogonal_lattice():
    from rhoprint.functionals.laplacian import compute_laplacian_grid

    rho = np.random.rand(8, 8, 8)
    skewed_lattice = np.array([[5, 0, 0], [1.5, 5, 0], [0, 0, 5]], dtype=float)

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        compute_laplacian_grid(rho, skewed_lattice, method="diagonal")
    assert any("non-orthogonal" in str(w.message) for w in record)


def test_laplacian_metric_tensor_method_does_not_warn_on_nonorthogonal_lattice():
    """The whole point of the fix: metric_tensor needs no warning, ever."""
    from rhoprint.functionals.laplacian import compute_laplacian_grid

    rho = np.random.rand(8, 8, 8)
    skewed_lattice = np.array([[5, 0, 0], [1.5, 5, 0], [0, 0, 5]], dtype=float)

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        compute_laplacian_grid(rho, skewed_lattice, method="metric_tensor")
    assert len(record) == 0


def test_compute_laplacian_stats_orthogonal_no_warning(synthetic_chgcar):
    path, _ = synthetic_chgcar
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        compute_laplacian_stats(path)
    non_orthogonal_warnings = [
        w for w in record if "non-orthogonal" in str(w.message)
    ]
    assert len(non_orthogonal_warnings) == 0


def test_compute_radial_moments_keys_and_ranges(synthetic_chgcar):
    path, _ = synthetic_chgcar
    stats = compute_radial_moments(path)

    expected_keys = {
        "total_charge", "moment_1", "moment_2", "moment_3",
        "radial_variance", "interstitial_fraction", "bond_fraction",
    }
    assert expected_keys.issubset(stats.keys())
    assert stats["moment_1"] > 0
    assert 0.0 <= stats["interstitial_fraction"] <= 1.0
    assert 0.0 <= stats["bond_fraction"] <= 1.0


def test_compute_all_grid_functionals_parses_once(synthetic_chgcar, monkeypatch):
    """compute_all_grid_functionals should call parse_chgcar exactly once."""
    path, _ = synthetic_chgcar

    call_count = {"n": 0}
    import rhoprint.io.chgcar_reader as reader_mod
    original = reader_mod.parse_chgcar

    def counting_parse(*args, **kwargs):
        call_count["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(reader_mod, "parse_chgcar", counting_parse)

    result = compute_all_grid_functionals(path)

    assert call_count["n"] == 1
    for key in ("zeta", "laplacian_mean", "moment_1", "total_charge"):
        assert key in result


def test_compute_derived_ratios_requires_columns():
    df = pd.DataFrame({"bond_fraction": [0.1]})
    with pytest.raises(KeyError):
        compute_derived_ratios(df)


def test_compute_compositional_features_valid_formula():
    row = compute_compositional_features("Fe2O3")
    assert row["n_elements"] == 2
    assert 0.0 <= row["ionicity"] <= 1.0
    assert not np.isnan(row["mean_mass"])


def test_compute_compositional_features_invalid_formula_returns_nan():
    row = compute_compositional_features("!!!not a formula!!!")
    assert np.isnan(row["mean_mass"])
