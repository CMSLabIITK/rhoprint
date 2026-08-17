# rhoprint

[![PyPI](https://img.shields.io/pypi/v/rhoprint.svg)](https://pypi.org/project/rhoprint/)

Physically interpretable descriptors ("fingerprints") extracted from VASP
charge-density (`CHGCAR`) files, for materials-informatics and ML
interpretability studies.

Developed at **CMS Lab, IIT Kanpur** (Department of Materials Science and
Engineering) alongside work on charge-density-informed ML models for
elastic-property prediction (bulk modulus, shear modulus, Young's modulus,
formation energy, Debye temperature).

## What it computes

`rhoprint compute --full` (or `rhoprint.compute_all_features`) extracts every
descriptor below for one material, given its CHGCAR file and a material_id of
the form `<formula>_<space_group_number>` (e.g. `NdP_225`).

### Charge-density grid descriptors
Computed directly from the CHGCAR voxel grid.

| Descriptor | What it measures |
|---|---|
| `zeta` | Angular variance of the density gradient relative to the nearest-atom direction. 0 = purely radial (ionic/metallic), 1 = highly directional (covalent). |
| `laplacian_neg_fraction` | Fraction of voxels where ∇²ρ < 0 (local charge concentration) — a proxy for covalent character. |
| `laplacian_std` | Standard deviation of ∇²ρ across the grid — heterogeneity of charge concentration/depletion. |
| `laplacian_mean`, `laplacian_pos_fraction`, `laplacian_min`, `laplacian_max` | Additional Laplacian summary statistics beyond the core three above. |
| `moment_1` | Density-weighted mean distance ⟨r⟩ of electrons from the nearest atom. |
| `moment_2` | Density-weighted second radial moment ⟨r²⟩. |
| `moment_3` | Density-weighted third radial moment (skewness proxy). |
| `radial_variance` | σ²_r = moment_2 − moment_1², the spread of the radial charge distribution. |
| `interstitial_fraction` | Fraction of total charge farther than 1.5 Å from any atom — free-electron/metallic signal. |
| `bond_fraction` | Fraction of total charge in the 0.8–1.5 Å bonding region. |
| `total_charge` | Integrated electron count over the grid (sanity check against valence electron count). |

### Structural & compositional descriptors
Derived from the material_id (crystal symmetry) and chemical formula (via `pymatgen`).

| Descriptor | What it measures |
|---|---|
| `crystal_system`, `crystal_system_int` | Crystal system (Triclinic...Cubic) and its integer code (1–7), derived from `space_group_number`. |
| `space_group_number` | Parsed directly from the material_id, e.g. `225` from `NdP_225`. |
| `is_f_block`, `has_f_block` | 1 if the formula contains a lanthanide/actinide element, else 0 (both columns hold the same value). |
| `mean_mass`, `max_mass`, `mass_range` | Stoichiometric mean, max, and range of constituent atomic masses. |
| `mean_elneg`, `elneg_diff` | Mean and max−min Pauling electronegativity across constituent species. |
| `mean_vec`, `max_vec` | Mean and max valence electron count across constituent species. |
| `mean_radius`, `radius_diff` | Mean and max−min atomic radius across constituent species. |
| `n_elements` | Number of unique chemical species (1=unary, 2=binary, ...). |
| `ionicity` | Pauling ionicity estimate from `elneg_diff`. |
| `mean_period` | Stoichiometric mean periodic-table period (1–7). |

### Derived cross-term descriptors
Ratios and products combining the above, capturing non-linear structure-electronic interactions.

| Descriptor | What it measures |
|---|---|
| `bond_int_ratio` | `bond_fraction / interstitial_fraction` — directional bonding vs. delocalized charge. |
| `radial_cv` | Coefficient of variation of the radial charge distribution. |
| `zeta_over_rvar` | Angular anisotropy relative to radial spread. |
| `lnf_x_m1` | `laplacian_neg_fraction × moment_1` — covalent volume fraction coupled with orbital radius. |
| `fint_over_lnf` | `interstitial_fraction / laplacian_neg_fraction` — interstitial delocalization vs. local covalent concentration. |
| `moment_ratio` | `moment_2 / moment_1` — spatial tailing of charge away from cores. |
| `vec_x_lnf` | `mean_vec × laplacian_neg_fraction` — valence electron availability coupled with covalency. |
| `vec_over_rvar` | Valence electron density relative to radial dispersion. |
| `bond_over_lnf` | `bond_fraction / laplacian_neg_fraction` — bond density vs. local covalent concentration. |
| `elneg_x_lnf` | `mean_elneg × laplacian_neg_fraction`. |
| `charge_per_m1` | `total_charge / moment_1`. |
| `lap_concentration` | Ratio of total charge-concentration magnitude to charge-depletion magnitude. |
| `sqrt_zeta` | √ζ — enhances sensitivity at low anisotropy. |
| `log_lnf` | ln(`laplacian_neg_fraction`) — reduces skew for ML regression. |

## Install

```bash
pip install rhoprint
# or, for development:
git clone https://github.com/CMSLabIITK/rhoprint
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

### Full descriptor set in one call

`compute_all_features` combines all three descriptor groups above (grid,
structural/compositional, derived) into a single row, given a material_id of
the form `<formula>_<space_group_number>`:

```python
from rhoprint import compute_all_features

row = compute_all_features("NdP_225", "path/to/CHGCAR")
```

or for a whole dataset (each subdirectory named `<formula>_<space_group_number>`,
containing a CHGCAR file):

```python
from rhoprint import run_full_batch

df = run_full_batch(
    data_dir="/path/to/materials",
    out_csv="results/full_functionals.csv",
    n_workers=4,
    resume=True,
)
```

or from the command line:

```bash
rhoprint compute path/to/NdP_225/CHGCAR --full
rhoprint batch --data_dir /path/to/materials --out_csv results/full_functionals.csv --n_workers 4 --full --resume
```

## Notes

- `parse_chgcar` assumes VASP 5 format (element symbols on their own line).
  VASP 4-format `CHGCAR` files (no element-symbol line) are not supported.

## Citing

See [`CITATION.cff`](CITATION.cff).

> **Zenodo DOI: pending.** The repo is owned by the CMS Lab GitHub org; a
> permanent Zenodo DOI will be minted once someone with org access enables
> the GitHub→Zenodo integration and cuts a tagged release. Until then, cite
> the GitHub repository and version tag directly.

## License

MIT — see [`LICENSE`](LICENSE).
