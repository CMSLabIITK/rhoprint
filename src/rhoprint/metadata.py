"""
rhoprint.metadata
-------------------
Structural metadata (chemical formula, space group number, crystal
system) parsed from a material identifier of the form
``"<formula>_<space_group_number>"``, e.g. ``"NdP_225"``.

The trailing ``_<int>`` is NOT part of the chemical formula -- it is
the space group number, confirmed against your dataset's directory
naming convention. Crystal system is derived from the space group
number via the standard International Tables for Crystallography
ranges; it is not re-computed via symmetry analysis.

.. important::
    ``CRYSTAL_SYSTEM_INT`` here (1=Triclinic ... 7=Cubic) matches your
    reference PDF (descriptor #25) exactly. This is the OPPOSITE order
    from the original ``functionals.py`` script's ``cs_map``, which had
    Cubic=1 ... Triclinic=7. That script's encoding was backwards
    relative to the documented definition -- if ``crystal_system_int``
    was ever used as a numeric/ordinal feature in a trained model, the
    encoding was inverted. This module uses the PDF's definition as the
    source of truth.
"""

import re

# (min_space_group, max_space_group, crystal_system_name), inclusive.
# Standard International Tables for Crystallography space-group ranges.
_SPACE_GROUP_RANGES = [
    (1, 2, "Triclinic"),
    (3, 15, "Monoclinic"),
    (16, 74, "Orthorhombic"),
    (75, 142, "Tetragonal"),
    (143, 167, "Trigonal"),
    (168, 194, "Hexagonal"),
    (195, 230, "Cubic"),
]

# Matches the reference PDF's descriptor #25 definition exactly:
# 1=Triclinic, 2=Monoclinic, 3=Orthorhombic, 4=Tetragonal, 5=Trigonal,
# 6=Hexagonal, 7=Cubic.
CRYSTAL_SYSTEM_INT = {
    "Triclinic": 1,
    "Monoclinic": 2,
    "Orthorhombic": 3,
    "Tetragonal": 4,
    "Trigonal": 5,
    "Hexagonal": 6,
    "Cubic": 7,
}

_MATERIAL_ID_PATTERN = re.compile(r"^(?P<formula>.+)_(?P<space_group>\d+)$")


def parse_material_id(material_id: str) -> dict:
    """
    Split a material_id like "NdP_225" into its formula and space
    group number.

    Parameters
    ----------
    material_id : str
        Of the form "<formula>_<space_group_number>". Matches the
        directory-naming convention used under
        /data/sai/new_charge/6000_data_aug13.

    Returns
    -------
    dict with keys: formula (str), space_group_number (int)

    Raises
    ------
    ValueError if material_id doesn't match the expected pattern.
    """
    match = _MATERIAL_ID_PATTERN.match(material_id)
    if not match:
        raise ValueError(
            f"Could not parse material_id {material_id!r}: expected format "
            f"'<formula>_<space_group_number>', e.g. 'NdP_225'."
        )
    return {
        "formula": match.group("formula"),
        "space_group_number": int(match.group("space_group")),
    }


def crystal_system_from_space_group(space_group_number: int) -> str:
    """
    Map a space group number (1-230) to its crystal system name.

    Raises
    ------
    ValueError if space_group_number is out of the valid [1, 230] range.
    """
    if not (1 <= space_group_number <= 230):
        raise ValueError(
            f"space_group_number {space_group_number} out of valid range [1, 230]"
        )
    for lo, hi, name in _SPACE_GROUP_RANGES:
        if lo <= space_group_number <= hi:
            return name
    raise AssertionError("unreachable: space group ranges should cover 1-230")


def compute_structural_metadata(material_id: str) -> dict:
    """
    Full structural metadata for one material, parsed from its
    material_id.

    Returns
    -------
    dict with keys: formula, space_group_number, crystal_system,
    crystal_system_int
    """
    parsed = parse_material_id(material_id)
    crystal_system = crystal_system_from_space_group(parsed["space_group_number"])
    return {
        "formula": parsed["formula"],
        "space_group_number": parsed["space_group_number"],
        "crystal_system": crystal_system,
        "crystal_system_int": CRYSTAL_SYSTEM_INT[crystal_system],
    }
