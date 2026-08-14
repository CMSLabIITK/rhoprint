# rhoprint

Physically interpretable descriptors ("fingerprints") extracted from VASP
charge-density (`CHGCAR`) files, for materials-informatics and ML
interpretability studies.

Developed at **CMS Lab, IIT Kanpur** (Department of Materials Science and
Engineering) alongside work on charge-density-informed ML models for
elastic-property prediction (bulk modulus, shear modulus, Young's modulus,
formation energy, Debye temperature).

## What it computes

For a single `CHGCAR` file, `rhoprint` extracts:

| Functional | What it measures |
|---|---|
| `zeta` | Angular variance of the density gradient relative to nearest-atom direction (0 = radial/ionic, 1 = angular/covalent) |
| `laplacian_mean`, `laplacian_std`, `laplacian_neg_fraction`, `laplacian_pos_fraction`, `laplacian_min`, `laplacian_max` | Grid-level ∇²ρ statistics — a cheap proxy for covalent (∇²ρ<0) vs. ionic/metallic (∇²ρ>0) bonding character |
| `total_charge`, `moment_1`, `moment_2`, `moment_3`, `radial_variance` | Radial moments of ρ(r) about the nearest atom |
| `interstitial_fraction`, `bond_fraction` | Fraction of electron density in the interstitial / bonding region |

A second layer of **derived cross-term descriptors** (ratios and products of
the above, e.g. `fint_over_lnf`) and **compositional descriptors** (via
`pymatgen`: mean electronegativity, oxidation state, atomic radius, etc.) can
be computed on top of a dataset of these raw functionals.

## Install

```bash
pip install rhoprint          # once published to PyPI
# or, for development:
git clone https://github.com/CMSLab-IITK/rhoprint
cd rhoprint
pip install -e ".[dev]"
```

## Quickstart

```python
from rhoprint import parse_chgcar, compute_all_grid_functionals

data = parse_chgcar("CHGCAR")
features = compute_all_grid_functionals(data)
print(features)
# {'zeta': 0.31, 'laplacian_mean': ..., 'moment_1': ..., ...}
```

### Batch processing a dataset

```python
from rhoprint import run_batch

df = run_batch(
    data_dir="/path/to/materials",   # each subdir has a CHGCAR file
    out_csv="results/functionals.csv",
    n_workers=4,
    resume=True,
)
```

or from the command line:

```bash
rhoprint compute path/to/CHGCAR
rhoprint batch --data_dir /path/to/materials --out_csv results/functionals.csv --n_workers 4 --resume
```

### Derived + compositional descriptors on a dataset

```python
import pandas as pd
from rhoprint.functionals.derived import compute_derived_ratios
from rhoprint.functionals.composition import compute_compositional_features_batch

df = pd.read_csv("results/functionals.csv")
comp = compute_compositional_features_batch(df["formula"])
df = pd.concat([df, comp], axis=1)
df = compute_derived_ratios(df)
```

## Known limitations

- **Non-orthogonal lattices**: `compute_laplacian_grid` defaults to
  `method="metric_tensor"`, which is exact for any lattice (orthogonal or
  not) — it uses the full contravariant metric tensor rather than assuming
  independent, perpendicular axes. A legacy `method="diagonal"` is also
  available, which reproduces the original approximation (exact only for
  orthogonal cells; silently wrong on monoclinic/triclinic/some hexagonal
  cells) — use it only if you need to exactly reproduce numbers computed
  before this fix, and note it will raise a warning on non-orthogonal
  lattices. See `rhoprint.functionals.laplacian` for the derivation and
  `tests/test_functionals.py` for an analytic benchmark demonstrating the
  difference between the two methods.
- QTAIM-style critical-point analysis is **not** performed; `laplacian_*`
  are grid-level summary statistics, used as a cheap proxy for bonding
  character rather than a rigorous topological classification.
- `parse_chgcar` assumes VASP 5 format (element symbols on their own line).
  VASP 4-format `CHGCAR` files (no element-symbol line) are not supported.

## Citing

See [`CITATION.cff`](CITATION.cff).

## License

MIT — see [`LICENSE`](LICENSE).
