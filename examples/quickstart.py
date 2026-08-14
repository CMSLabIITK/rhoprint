"""
Quickstart: compute all grid-level functionals for a single CHGCAR
file, then (optionally) run a small batch and layer on derived +
compositional descriptors.

Usage:
    python quickstart.py path/to/CHGCAR
"""

import sys

from rhoprint import parse_chgcar, compute_all_grid_functionals


def main():
    if len(sys.argv) < 2:
        print("Usage: python quickstart.py path/to/CHGCAR")
        sys.exit(1)

    chgcar_path = sys.argv[1]

    # Parse once, reuse for all functionals.
    data = parse_chgcar(chgcar_path)
    print(f"Grid: {data['nx']}x{data['ny']}x{data['nz']}, "
          f"{data['n_atoms']} atoms, volume={data['volume']:.2f} A^3")

    features = compute_all_grid_functionals(data)
    print("\nGrid-level functionals:")
    for k, v in features.items():
        print(f"  {k:25s} = {v}")

    print(
        "\nFor a full dataset: use rhoprint.run_batch(...) to process many "
        "materials into a CSV, then rhoprint.functionals.composition and "
        "rhoprint.functionals.derived to add compositional + cross-term "
        "descriptors on top."
    )


if __name__ == "__main__":
    main()
