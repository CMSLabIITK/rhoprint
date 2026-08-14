from rhoprint.full import compute_all_features, FULL_SCHEMA_COLUMNS


def test_compute_all_features_has_all_schema_columns(synthetic_chgcar):
    path, _ = synthetic_chgcar
    # material_id's formula doesn't need to match the synthetic CHGCAR's
    # placeholder atom -- composition is derived purely from the formula
    # string, independent of the grid computation.
    row = compute_all_features("Fe2O3_225", path)

    expected = set(FULL_SCHEMA_COLUMNS) - {"material_id"}  # already checked separately
    missing = expected - set(row.keys())
    assert not missing, f"compute_all_features is missing columns: {missing}"

    assert row["material_id"] == "Fe2O3_225"
    assert "split" not in row  # explicitly not computed here


def test_compute_all_features_is_f_block_matches_has_f_block(synthetic_chgcar):
    path, _ = synthetic_chgcar
    row = compute_all_features("Nd2O3_225", path)  # Nd is f-block (lanthanide)
    assert row["is_f_block"] == row["has_f_block"] == 1


def test_compute_all_features_crystal_system_from_space_group(synthetic_chgcar):
    path, _ = synthetic_chgcar
    row = compute_all_features("NdP_225", path)
    assert row["space_group_number"] == 225
    assert row["crystal_system"] == "Cubic"
    assert row["crystal_system_int"] == 7
