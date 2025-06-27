import pytest
from uncertimety.dataprep import (
    normalize_vintage_label,
    clean_vintage,
    clean_name,
    normalize_column_name,
    clean_col_names,
    import_census_dataset,
    check_year,
    round_to_next_5,
    infer_last_full_year,
    get_monthly_activity,
    get_vintage_shares,
    extract_year_from_token,
    parse_single_vintage,
)


@pytest.fixture
def vintage_replacements():
    return {"or": "", "to": "", "house": ""}


@pytest.fixture
def type_replacements():
    return {
        "total": "total",
        "movable": "mobile",
        "apartment": "apartments",
        "fewer": "apartment<5",
        "more": "apartment>5",
        "semi-": "semi_detached",
        "duplex": "apartment_duplex",
    }


@pytest.fixture
def mock_csv():
    cs_1961 = """
Period of const,Total occupied private dwellings,  Single-detached house,  Single-attached house,  Apartment,Mobile
Total occupied private dwellings,1198368,467716,138373,583983,1296
  1945 or before,,,,,
    1920 or before,357568,132101,55621,169841,5
    1921 - 1945,301937,101448,37692,162766,31
  1946 - 1960,,,,,
    1946 - 1959,484486,211256,42351,230088,791
    1960 - 1961 (1),47377,22911,2709,21288,469
"""
    return cs_1961


@pytest.fixture
def data_dir_with_mock_csv(tmp_path, mock_csv):
    census_dir = tmp_path / "census"
    census_dir.mkdir(parents=True, exist_ok=True)
    csv_file = census_dir / "qc_s_c_g_mock_1961.csv"
    csv_file.write_text(mock_csv.strip(), encoding="utf-8")
    return tmp_path


@pytest.fixture
def data_dir_with_bad_filename(tmp_path, mock_csv):
    census_dir = tmp_path / "census"
    census_dir.mkdir(parents=True)
    bad_file = census_dir / "qc_s_c_g_mock_no_year.csv"
    bad_file.write_text(mock_csv.strip(), encoding="utf-8")
    return tmp_path


# === Unit Tests ===


def test_normalize_vintage_label():
    cases = [
        ("Total", "total"),
        ("1945 or before", "<1945"),
        ("1946-1960", "1946-1960"),
        ("1986 or after", "1986+"),
        ("1986 (1)", "1986-1"),
        ("1996(1)", "1996-1"),
        ("1961-1971(1)", "1961-1971-1"),
        ("2011 to 2015", "2011-2015"),
    ]
    # TODO test with other sep values
    # TODO: check for 1960 - 1961 (1) : how is it treated?
    for raw, expected in cases:
        assert normalize_vintage_label(raw) == expected


def test_clean_vintage():
    # TODO test with other sep values
    input_list = [
        "Total",
        "Total occupied private dwellings",
        "1945 or before",
        "1986 or after",
        "1996(1)",
        "1981 to 1991",
        "1961-1971(1)",
    ]
    expected = [
        "total",
        "total",
        "<1945",
        "1986+",
        "1996-1",
        "1981-1991",
        "1961-1971-1",
    ]
    assert clean_vintage(input_list) == expected


def test_clean_name(vintage_replacements):
    cases = [
        ("Total", "total"),
        (" !-:_ _Total --- _ ", "total"),
        ("1956-1961", "1956_1961"),
        ("1945-or-before", "1945_before"),
        ("  1945  or  before   ", "1945_before"),
        ("  1945  to  1961   ", "1945_1961"),
        ("1996(1)", "1996_1"),
        ("1986 or after", "1986_after"),
        ("   1986  (1)  ", "1986_1"),
        ("2011 to 2015", "2011_2015"),
        # TODO: check for 1960 - 1961 (1) : how is it treated?
    ]

    for raw, expected in cases:
        # TODO Test using different separators?
        assert clean_name(raw, replacements=vintage_replacements) == expected


def test_normalize_column_name(type_replacements):
    cases = [
        ("Total", "total"),
        ("Apartment", "apartments"),
        ("Single attached", "single_attached"),
        ("Other single-attached house", "other_single_attached"),
        ("Other single-attached house 3 (42)", "other_single_attached"),
        ("Single Detached", "single_detached"),
        ("Other dwelling (274)", "other_dwelling"),
        ("  Apartment: five or more storeys", "apartment>5"),
        ("  Apartment, detached duplex", "apartment_duplex"),
        ("Single-detached house", "single_detached"),
        ("Apartment in a building that has five or more storeys", "apartment>5"),
        ("Other attached dwelling", "other_attached_dwelling"),
        ("  Apartment or flat in a duplex", "apartment_duplex"),
        ("Apartment in a building that has fewer than five storeys", "apartment<5"),
        ("  Other single-attached house", "other_single_attached"),
        ("  Row house", "row"),
        ("  Semi-detached house", "semi_detached"),
        ("Movable dwelling", "mobile"),
    ]

    for raw, expected in cases:
        assert normalize_column_name(raw, type_replacements) == expected


def test_clean_col_names(type_replacements):
    # TODO: move some names from test_normalize_column_names here
    input_cols = [
        "Total",
        "Movable house",
        "Apartment",
        "  Other single-attached house",
        "  Apartment in a building that has fewer than five storeys (76)",
    ]
    expected = [
        "total",
        "mobile",
        "apartments",
        "other_single_attached",
        "apartment<5",
    ]
    # TODO: test different separators?
    assert (
        clean_col_names(input_cols, replacements=type_replacements, sep="_") == expected
    )


def test_import_valid_file(data_dir_with_mock_csv, type_replacements):
    result = import_census_dataset(
        data_dir=data_dir_with_mock_csv, replacements=type_replacements
    )
    # FIXME unit="dw" seems useless?

    assert isinstance(result, dict)
    assert "1961" in result
    df = result["1961"]

    assert "census_year" in df.columns
    assert "vintage" in df.columns
    assert df["census_year"].iloc[0] == "1961"
    assert df["vintage"].iloc[2].strip() == "<1920"
    assert "apartments" in df.columns
    # TODO: add tests for dataframe content?


def test_missing_replacements_raises(data_dir_with_mock_csv):
    with pytest.raises(ValueError, match="replacements"):
        import_census_dataset(data_dir=data_dir_with_mock_csv, replacements=None)


def test_no_files_found_raises(tmp_path, type_replacements):
    empty_dir = tmp_path / "census"
    empty_dir.mkdir(parents=True, exist_ok=True)
    with pytest.raises(ValueError, match="No census CSV files"):
        import_census_dataset(data_dir=tmp_path, replacements=type_replacements)


def test_bad_filename_raises(data_dir_with_bad_filename, type_replacements):
    with pytest.raises(ValueError, match="valid 4-digit census year"):
        import_census_dataset(
            data_dir=data_dir_with_bad_filename,
            replacements=type_replacements,
        )


def test_check_year():
    with pytest.raises(TypeError, match="Expected int or str convertible"):
        check_year("nineteen")

    with pytest.raises(ValueError, match="unexpected year"):
        check_year(True)

    with pytest.raises(ValueError, match="non-negative"):
        check_year(-9)

    with pytest.raises(ValueError, match="unexpected year"):
        # gets converted to 0
        check_year(0.34)

    with pytest.raises(ValueError, match="unexpected year"):
        check_year("2222")


def test_round_to_next_5():
    cases = [
        (1975, 1980),
        (1976, 1980),
        (1980, 1985),
        (1981, 1985),
        (1986, 1990),
        (1988, 1990),
    ]
    for raw, expected in cases:
        assert round_to_next_5(raw) == expected


def test_infer_last_full_year():
    cases = [
        (1976, 1975),
        (1981, 1980),
        ("1986", 1985),
        ("1999", 1995),
    ]
    for raw, expected in cases:
        assert infer_last_full_year(raw) == expected


def test_get_monthly_activity():
    cases = [0, 11, 7, None]

    for value in cases:
        assert isinstance(get_monthly_activity(inactive_months=value), list)
        assert len(get_monthly_activity(inactive_months=value)) == 12
        assert all(
            activity >= 0 for activity in get_monthly_activity(inactive_months=value)
        )
        assert abs(sum(get_monthly_activity(inactive_months=value)) - 1.0) < 1e-6

    with pytest.raises(ValueError, match="an int between"):
        get_monthly_activity(inactive_months=-5)

    with pytest.raises(ValueError, match="an int between"):
        get_monthly_activity(inactive_months=89)


def test_get_vintage_shares():
    cases = [
        ("1960-1961-1", None, [12 / 17, 5 / 17]),
        ("1966-1971-1", None, [60 / 65, 5 / 65]),
        ("1960-1961-1", 4, [8 / 9, 1 / 9]),
        ("1966-1971-1", 4, [40 / 41, 1 / 41]),
        ("1960-1961-1", 5, [1, 0]),
        ("1966-1971-1", 5, [1, 0]),
    ]
    # FIXME these tests all assume census_month = 5, test for different census_month cutoffs
    # TODO test different separators

    for label, inactive_months, expected in cases:
        assert get_vintage_shares(
            label, sep="-", inactive_months=inactive_months
        ) == pytest.approx(expected)

    with pytest.raises(ValueError, match="Invalid vintage label"):
        get_vintage_shares("1986-1")


def test_extract_year_from_token_valid():
    assert extract_year_from_token("<1920", "<") == 1920
    assert extract_year_from_token("1986+", "+") == 1986

    with pytest.raises(ValueError):
        extract_year_from_token("hello", "<")

    with pytest.raises(ValueError):
        extract_year_from_token("<abc", "<")


def test_parse_single_vintage():
    cases = [
        ("total", 1991, [(1608, 1995)], [1.0]),
        ("<1920", 2001, [(1608, 1920)], [1.0]),
        ("1986+", 1996, [(1986, 2000)], [1.0]),
        ("1960-1961-1", 1961, [(1960, 1960), (1961, 1965)], [12 / 17, 5 / 17]),
        ("1986-1", 1986, [(1986, 1990)], [1.0]),
        ("1966-1971-1", 1971, [(1966, 1970), (1971, 1975)], [60 / 65, 5 / 65]),
    ]  # NOTE years are inclusive - stock is measured at the end of year (consistent with ODYM definitions)
    # FIXME the cases might need to be changed if the behaviour of "total" is modified to stop at census year.

    for label, census_year, expected_labels, expected_shares in cases:
        new_labels, shares = parse_single_vintage(label, census_year)
        assert new_labels == expected_labels
        assert shares == pytest.approx(expected_shares)
