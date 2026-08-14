import pytest

from rhoprint.metadata import (
    parse_material_id,
    crystal_system_from_space_group,
    compute_structural_metadata,
    CRYSTAL_SYSTEM_INT,
)


def test_parse_material_id_basic():
    result = parse_material_id("NdP_225")
    assert result == {"formula": "NdP", "space_group_number": 225}


def test_parse_material_id_formula_with_numbers():
    # formula itself may contain digits (e.g. "Ta6Sn2"); rsplit on the
    # LAST underscore must still isolate the trailing space group.
    result = parse_material_id("Ta6Sn2_221")
    assert result == {"formula": "Ta6Sn2", "space_group_number": 221}


def test_parse_material_id_invalid_raises():
    with pytest.raises(ValueError):
        parse_material_id("no_underscore_but_not_numeric")
    with pytest.raises(ValueError):
        parse_material_id("NoUnderscoreHere")


@pytest.mark.parametrize("sg,expected", [
    (1, "Triclinic"), (2, "Triclinic"),
    (3, "Monoclinic"), (15, "Monoclinic"),
    (16, "Orthorhombic"), (74, "Orthorhombic"),
    (75, "Tetragonal"), (142, "Tetragonal"),
    (143, "Trigonal"), (167, "Trigonal"),
    (168, "Hexagonal"), (194, "Hexagonal"),
    (195, "Cubic"), (225, "Cubic"), (230, "Cubic"),
])
def test_crystal_system_from_space_group(sg, expected):
    assert crystal_system_from_space_group(sg) == expected


def test_crystal_system_from_space_group_out_of_range():
    with pytest.raises(ValueError):
        crystal_system_from_space_group(0)
    with pytest.raises(ValueError):
        crystal_system_from_space_group(231)


def test_crystal_system_int_matches_reference_pdf():
    """
    Descriptor #25 in the reference PDF: 1=Triclinic, 2=Monoclinic,
    3=Orthorhombic, 4=Tetragonal, 5=Trigonal, 6=Hexagonal, 7=Cubic.
    """
    assert CRYSTAL_SYSTEM_INT == {
        "Triclinic": 1, "Monoclinic": 2, "Orthorhombic": 3,
        "Tetragonal": 4, "Trigonal": 5, "Hexagonal": 6, "Cubic": 7,
    }


def test_compute_structural_metadata():
    meta = compute_structural_metadata("NdP_225")
    assert meta["formula"] == "NdP"
    assert meta["space_group_number"] == 225
    assert meta["crystal_system"] == "Cubic"
    assert meta["crystal_system_int"] == 7
