import pytest
import pandas as pd
import numpy as np
from uncertimety.dataprep import (
    _check_sums,
    _extract_year_from_token,
    _validate_data_preservation,
    add_missing_vintages_types,
    check_year,
    clean_col_names,
    clean_name,
    clean_vintage,
    drop_duplicate_rows,
    get_monthly_activity,
    get_vintage_shares,
    import_census_dataset,
    infer_last_full_year,
    normalize_column_name,
    normalize_vintage_label,
    parse_single_vintage,
    round_to_next_5,
    validate_vintage_interval,
    vintage_label_to_tuple,
    calculate_missing_types,
    _validate_frame_preservation,
    check_series_sum,
)
# FIXME: split/rename tests (see, e.g., drop_duplicate_rows)
# FIXME: add _ to helper functions


@pytest.fixture
def vintage_replacements():
    return {"or": "", "to": "", "house": ""}


@pytest.fixture
def type_replacements():
    return {
        "total": "total",
        "movable": "mobile",
        "fewer": "apartment_lt_5",
        "more": "apartment_ge_5",
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


@pytest.fixture
def data_columns():
    return [
        "total",
        "single_detached",
        "single_attached",
        "apartments",
        "mobile",
    ]


@pytest.fixture
def original_row():
    return pd.Series(
        {
            "vintage": "1960 - 1961 (1)",
            "census_year": 1961,
            "total": 47377,
            "single_detached": 22911,
            "single_attached": 2709,
            "apartments": 21288,
            "mobile": 469,
        }
    )


@pytest.fixture
def valid_split_rows():
    return [
        {
            "vintage": "1960-1960",
            "census_year": 1961,
            "total": 33443,
            "single_detached": 16172,
            "single_attached": 1912,
            "apartments": 15027,
            "mobile": 331,
        },
        {
            "vintage": "1961-1961",
            "census_year": 1961,
            "total": 13934,
            "single_detached": 6739,
            "single_attached": 797,
            "apartments": 6261,
            "mobile": 138,
        },
    ]


@pytest.fixture
def invalid_split_rows():
    # Slightly off (Total = 47392 instead of 47377)
    # FIXME check for different values?
    return [
        {
            "vintage": "1960-1960",
            "census_year": 1961,
            "total": 33458,  # <-- Incremented by 15
            "single_detached": 16172,
            "single_attached": 1912,
            "apartments": 15027,
            "mobile": 331,
        },
        {
            "vintage": "1961-1961",
            "census_year": 1961,
            "total": 13934,
            "single_detached": 6739,
            "single_attached": 797,
            "apartments": 6261,
            "mobile": 138,
        },
    ]


@pytest.fixture
def total_dwelling_types():
    return [
        "single_detached",
        "semi_detached",
        "row",
        "apartment_duplex",
        "apartment_ge_5",
        "apartment_lt_5",
        "other_single_attached",
        "mobile",
    ]


@pytest.fixture
def historic_vintages():
    return [
        "1608-2025",
        "1608-1945",
        "1608-1920",
        "1921-1945",
        "1946-1960",
        "1961-1970",
        "1971-1980",
        "1981-1990",
        "1991-1995",
        "1996-2000",
        "2001-2005",
        "2006-2010",
        "2011-2015",
        "2016-2020",
        "2021-2025",
    ]


@pytest.fixture
def df_1991():
    data = {
        "census_year": [1991] * 13,
        "vintage": [
            "total",
            "le-1945",
            "le-1920",
            "1921-1945",
            "1946-1970",
            "1946-1960",
            "1961-1970",
            "1971-1985",
            "1971-1980",
            "1981-1985",
            "ge-1986",
            "1986-1990",
            "1991-1",
        ],
        "total": [
            2634300,
            474795,
            204285,
            270510,
            994930,
            495950,
            498975,
            825455,
            600180,
            225270,
            339125,
            328310,
            10820,
        ],
        "single_detached": [
            1174795,
            184310,
            96875,
            87435,
            402970,
            196440,
            206530,
            430270,
            321925,
            108345,
            157245,
            152450,
            4795,
        ],
        "apartment_ge_5": [
            137120,
            6020,
            1680,
            4340,
            47635,
            11785,
            35855,
            60460,
            44755,
            15700,
            23005,
            22360,
            650,
        ],
        "mobile": [
            24605,
            180,
            70,
            110,
            2505,
            385,
            2130,
            19410,
            16970,
            2440,
            2505,
            2330,
            180,
        ],
        "other_dwelling": [
            1297785,
            284285,
            105660,
            178620,
            541815,
            287345,
            254470,
            315320,
            216530,
            98785,
            156370,
            151170,
            5200,
        ],
    }
    return pd.DataFrame(data)


@pytest.fixture
def base_duplicate_df():
    data = {
        "vintage": ["1946-1970", "1946-1970", "1991-1995", "1991-1995"],
        "single_detached": [np.nan, np.nan, 10, np.nan],
        "mobile": [np.nan, np.nan, 0, np.nan],
    }
    return pd.DataFrame(data)


@pytest.fixture
def duplicate_with_conflict_df():
    data = {
        "vintage": ["2001-2005", "2001-2005"],
        "single_detached": [5, 6],
        "mobile": [1, np.nan],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_df_to_sum():
    data = {
        "census_year": [2021, 2021],
        "vintage": ["2001-2005", "2006-20010"],
        "source": ["(9, 2001-2005)", "10, 2006-2010"],
        "split": [False, False],
        "apartment_duplex": [3, 2],
        "apartment_ge_5": [7, 8],
        "apartment_lt_5": [5, 1],
        "single_detached": [100, 10],
        "apartments": [None, np.nan],
    }
    return pd.DataFrame(data)


@pytest.fixture
def agg_types():
    return {
        "apartments": ["apartment_duplex", "apartment_ge_5", "apartment_lt_5"],
        "single_attached": ["semi_detached", "other_single_attached", "row"],
        "other_attached_dwelling": [
            "apartment_duplex",
            "apartment_lt_5",
            "other_single_attached",
            "row",
            "semi_detached",
        ],
    }


@pytest.fixture
def original_df():
    data = {
        "census_year": [1981] * 3,
        "vintage": ["1608-2025", "1946-1960", "1961-1970"],
        "total": [2172855, None, None],
        "apartment_duplex": [239190, None, None],
        "apartment_ge_5": [115515, None, None],
        "apartment_lt_5": [597995, None, None],
        "apartments": [952700, None, None],
        "single_attached": [228545, None, None],
        "single_detached": [954455, None, None],
    }
    return pd.DataFrame(data)


@pytest.fixture
def matching_df(original_df):
    return original_df.copy()


@pytest.fixture
def slightly_modified_df(original_df):
    df = original_df.copy()
    df.loc[0, "total"] += 4  # within default atol=5
    return df


@pytest.fixture
def significantly_modified_df(original_df):
    df = original_df.copy()
    df.loc[0, "total"] += 400  # outside default atol + rtol for ~2M dwellings
    return df


@pytest.fixture
def df_for_check_sums():
    df = pd.DataFrame(
        {
            "vintage": ["1608-2025", "1608-1920", "1921-1945", "1946-1960"],
            "total": [1500, 500, 200, 800],
            "single_detached": [500, 300, 192, 8],
            "other_attached_dwelling": [800, np.nan, 8, 792],
            "other_dwelling": [200, np.nan, np.nan, 0],
        }
    )
    return df


# === Unit Tests ===


def test_normalize_vintage_label():
    cases = [
        ("Total", "total"),
        ("1945 or before", "le-1945"),
        ("1946-1960", "1946-1960"),
        ("1986 or after", "ge-1986"),
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
        "le-1945",
        "ge-1986",
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
        ("  Apartment: five or more storeys", "apartment_ge_5"),
        ("  Apartment, detached duplex", "apartment_duplex"),
        ("Single-detached house", "single_detached"),
        ("Apartment in a building that has five or more storeys", "apartment_ge_5"),
        ("Other attached dwelling", "other_attached_dwelling"),
        ("  Apartment or flat in a duplex", "apartment_duplex"),
        ("Apartment in a building that has fewer than five storeys", "apartment_lt_5"),
        ("  Other single-attached house", "other_single_attached"),
        ("  Row house", "row"),
        ("  Semi-detached house", "semi_detached"),
        ("Total structural type of dwelling", "total"),
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
        "apartment_lt_5",
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
    assert df["vintage"].iloc[2].strip() == "le-1920"
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
    assert _extract_year_from_token("<1920", "<") == "le-1920"
    assert _extract_year_from_token("1986+", "+") == "ge-1986"

    with pytest.raises(ValueError):
        _extract_year_from_token("hello", "<")

    with pytest.raises(ValueError):
        _extract_year_from_token("<abc", "<")


def test_parse_single_vintage():
    cases = [
        ("total", 1991, True, [(1608, 2025)], [1.0]),
        ("le-1920", 2001, True, [(1608, 1920)], [1.0]),
        ("<1920", 2001, True, [(1608, 1920)], [1.0]),
        ("1986+", 1986, True, [(1986, 1990)], [1.0]),
        ("1986-1990", 1986, False, [(1986, 1986)], [1.0]),
        ("1986-1990", 1986, True, [(1986, 1990)], [1.0]),
        ("1986+", 1996, True, [(1986, 2000)], [1.0]),
        # FIXME: for cases in -1, should add has a single year, not a block of years. Otherwise, it 'opens' the bounds of total too much. in 1986, total ends in 1986, not in 1990!
        ("1960-1961-1", 1961, True, [(1960, 1960), (1961, 1965)], [12 / 17, 5 / 17]),
        ("1960-1961-1", 1961, False, [(1960, 1960), (1961, 1961)], [12 / 17, 5 / 17]),
        ("1986-1", 1986, False, [(1986, 1986)], [1.0]),
        ("1986-1", 1986, True, [(1986, 1990)], [1.0]),
        ("1966-1971-1", 1971, True, [(1966, 1970), (1971, 1975)], [60 / 65, 5 / 65]),
        ("1966-1971-1", 1971, False, [(1966, 1970), (1971, 1971)], [60 / 65, 5 / 65]),
    ]  # NOTE years are inclusive - stock is measured at the end of year (consistent with ODYM definitions)
    # FIXME the cases might need to be changed if the behaviour of "total" is modified to stop at census year.

    for label, census_year, round_flag, expected_labels, expected_shares in cases:
        new_labels, shares = parse_single_vintage(
            label, census_year, round_last_vintage=round_flag
        )
        assert new_labels == expected_labels
        assert shares == pytest.approx(expected_shares)


def test_check_sums():
    assert _check_sums([0.4, 0.6])
    assert _check_sums(0.8, target=0.8)  # Check it accepts floats
    assert _check_sums(0.8) is False

    assert _check_sums(np.array([0.1, 0.2, 0.7]))
    assert _check_sums(pd.Series([0.25, 0.75]))
    assert _check_sums([0.3333, 0.6667], atol=1e-4)

    with pytest.raises(TypeError):
        _check_sums([0.1, "not a float", 0.9])

    with pytest.raises(TypeError):
        _check_sums("0.1,0.9")

    with pytest.raises(ValueError):
        _check_sums([0.1, 0.1, 0.5], raise_error=True)


def test_valid_data_preservation(original_row, valid_split_rows, data_columns):
    result = _validate_data_preservation(
        original_row=original_row,
        new_rows=valid_split_rows,
        data_columns=data_columns,
        idx=42,
        original_label="1960-1961-1",
    )
    assert result == valid_split_rows


def test_invalid_data_preservation(original_row, invalid_split_rows, data_columns):
    with pytest.raises(ValueError, match="Value mismatch after splitting row"):
        _validate_data_preservation(
            original_row=original_row,
            new_rows=invalid_split_rows,
            data_columns=data_columns,
            idx=42,
            original_label="1960-1961-1",
        )


def test_add_missing_vintages_types(
    df_1991, historic_vintages, total_dwelling_types, tmp_path
):
    # Setup
    result = add_missing_vintages_types(
        dataframe=df_1991.copy(),
        target_vintages=historic_vintages,
        target_types=total_dwelling_types,
        config_dir=tmp_path,  # ignored in this test
        sep="-",
    )
    # === 1. Check all vintages are present ===
    vintages_out = set(result["vintage"])
    assert set(historic_vintages).issubset(vintages_out)

    # === 2. Check all expected dwelling types exist as columns ===
    for col in total_dwelling_types:
        assert col in result.columns

    # === 3. Check column order starts with census_year, vintage ===
    assert result.columns[:2].tolist() == ["census_year", "vintage"]


def test_missing_vintage_filling_logic(total_dwelling_types, tmp_path):
    # Only a subset of vintages is present
    df = pd.DataFrame(
        {
            "census_year": [2001],
            "vintage": ["1946-1970"],
            "total": [100],
            "single_detached": [50],
            "apartment_ge_5": [25],
            "mobile": [25],
        }
    )

    target_vintages = [
        "1946-1970",
        "2001-2005",
        "2011-2015",
    ]  # One before, one containing, and one after census_year
    target_types = total_dwelling_types

    result = add_missing_vintages_types(
        dataframe=df.copy(),
        target_vintages=target_vintages,
        target_types=target_types,
        config_dir=tmp_path,  # dummy path; not used
        sep="-",
    )

    row_2011 = result[result["vintage"] == "2011-2015"].iloc[0]
    row_2001 = result[result["vintage"] == "2001-2005"].iloc[0]
    row_1946 = result[result["vintage"] == "1946-1970"].iloc[0]

    # === Check zero fill for vintage after census year ===
    for col in target_types:
        assert row_2011[col] == 0 or np.isnan(row_2011[col]) is False
        assert np.isnan(row_2001[col])

    # === Check NaN preservation for original row ===
    # Columns not originally present should be NaN in this row
    missing_cols = set(target_types) - set(df.columns)
    for col in missing_cols:
        assert pd.isna(row_1946[col])


def test_drop_duplicate_rows_only_keeps_one_nan(base_duplicate_df):
    df = base_duplicate_df
    meta_cols = ["vintage"]
    data_cols = base_duplicate_df.columns.difference(meta_cols)

    result = drop_duplicate_rows(df, meta_cols)

    assert len(result) == 2  # one for 1946-1970, one for 1991-1995
    assert result["vintage"].value_counts().max() == 1
    assert result.loc[result["vintage"] == "1946-1970"].iloc[0][data_cols].isna().all()


def test_drop_duplicate_rows_keeps_valid_row(base_duplicate_df):
    df = base_duplicate_df
    meta_cols = ["vintage"]

    result = drop_duplicate_rows(df, meta_cols)

    row = result[result["vintage"] == "1991-1995"].iloc[0]
    assert row["single_detached"] == 10
    assert row["mobile"] == 0


def test_drop_duplicate_rows_raises_on_multiple_valid_rows(duplicate_with_conflict_df):
    df = duplicate_with_conflict_df
    meta_cols = ["vintage"]

    with pytest.raises(
        ValueError, match="Multiple non-NaN rows found for vintage '2001-2005'"
    ):
        drop_duplicate_rows(df, meta_cols)


def test_validate_vintage_interval():
    valid_tuple = (1987, 1989)
    invalid_tuple = (1986, 1985)
    assert validate_vintage_interval(valid_tuple) is None

    with pytest.raises(ValueError, match="Invalid vintage interval:"):
        validate_vintage_interval(invalid_tuple)


def test_vintage_label_to_tuple():
    # FIXME add other tests for 'wrong' inputs; see normalize_vintage_labels
    assert vintage_label_to_tuple("1986-1990") == (1986, 1990)

    with pytest.raises(ValueError):
        vintage_label_to_tuple(1982)

    with pytest.raises(ValueError):
        vintage_label_to_tuple("19829801")


def test_calculate_missing_types_values(sample_df_to_sum, agg_types):
    result = calculate_missing_types(sample_df_to_sum, agg_types)
    assert result["apartments"].tolist() == [15, 11]


def test_calculate_missing_types_preserves_original(sample_df_to_sum, agg_types):
    # FIXME split tests in different files (e.g., one per function) then rename, preserves_original_if_not_inplace
    _ = calculate_missing_types(sample_df_to_sum, agg_types)
    assert sample_df_to_sum["apartments"].isna().all()


def test_calculate_missing_types_modifies_if_inplace(sample_df_to_sum, agg_types):
    calculate_missing_types(sample_df_to_sum, agg_types, inplace=True)
    assert sample_df_to_sum["apartments"].tolist() == [15, 11]


def test_validate_matching_frames(
    matching_df, historic_vintages, data_columns, original_df
):
    assert (
        _validate_frame_preservation(
            original_df, matching_df, historic_vintages, data_columns
        )
        is None
    )


def test_validate_within_tolerance(
    slightly_modified_df, historic_vintages, data_columns, original_df
):
    assert (
        _validate_frame_preservation(
            original_df, slightly_modified_df, historic_vintages, data_columns
        )
        is None
    )


def test_validate_outside_tolerance(
    significantly_modified_df, historic_vintages, data_columns, original_df
):
    with pytest.raises(ValueError, match="Frame mismatch"):
        _validate_frame_preservation(
            original_df, significantly_modified_df, historic_vintages, data_columns
        )


class TestCheckSeriesSum:  # FIXME rename _check_sums?
    def test_exact_match(self, df_for_check_sums):
        """Test when component values sum exactly to total."""
        passed, details = check_series_sum(df_for_check_sums, "1608-2025")
        assert passed
        assert details["total"] == 1500
        assert details["component_sum"] == 1500
        assert details["difference"] == 0

    def test_within_tolerance(self, df_for_check_sums):
        """Test when sum is within tolerance."""
        df = df_for_check_sums.copy()
        df.loc[1, "total"] = 502  # +2
        df.loc[2, "total"] = 203  # +3

        passed, details = check_series_sum(df, "total", atol=6, rtol=0, axis=1)
        assert passed
        assert details["difference"] == -5

        # Test with relative tolerance
        df.loc[1, "total"] = 490  # -10
        df.loc[2, "total"] = 203  # +3

        passed, details = check_series_sum(df, "total", atol=0, rtol=0.01, axis=1)
        assert passed
        assert details["difference"] == 7

    def test_outside_tolerance(self, df_for_check_sums):
        """Test when sum is outside tolerance."""
        df = df_for_check_sums.copy()
        df.loc[1, "total"] = 510

        passed, details = check_series_sum(df, "total", atol=5, rtol=1e-5, axis=1)
        assert not passed
        assert details["difference"] == -10

    def test_wrong_target_axis(self, df_for_check_sums):
        """Test when target axis is not 0 or 1."""
        with pytest.raises(KeyError, match="not found in DataFrame"):
            check_series_sum(df_for_check_sums, "total", axis=0)

    def test_invalid_axis(self, df_for_check_sums):
        """Test when target axis is not 0 or 1."""
        with pytest.raises(ValueError, match="Invalid axis"):
            check_series_sum(df_for_check_sums, "total", axis=3)
