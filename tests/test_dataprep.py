from uncertimety.dataprep import (
    normalize_vintage_label,
    clean_vintage,
    clean_name,
    normalize_column_name,
    clean_col_names,
)


def test_normalize_vintage_label():
    # TODO test with other sep values
    assert normalize_vintage_label("Total") == "total"
    assert normalize_vintage_label("1945 or before") == "<1945"
    assert normalize_vintage_label("1946-1960") == "1946-1960"
    assert normalize_vintage_label("1986 or after") == "1986+"
    assert normalize_vintage_label("1986 (1)") == "1986"
    assert normalize_vintage_label("1996(1)") == "1996"
    assert normalize_vintage_label("2011 to 2015") == "2011-2015"


def test_clean_vintage():
    # TODO test with other sep values
    input_list = ["Total", "1945 or before", "1986 or after", "1996(1)", "1981 to 1991"]
    expected = ["total", "<1945", "1986+", "1996", "1981-1991"]
    assert clean_vintage(input_list) == expected


def test_clean_name():
    mock_replacements = {"or": "", "to": "", "house": ""}
    # TODO Test using different separators?
    assert clean_name("Total", replacements=mock_replacements) == "total"
    assert clean_name(" !-:_ _Total --- _ ", replacements=mock_replacements) == "total"
    assert clean_name("1956-1961", replacements=mock_replacements) == "1956_1961"
    assert clean_name("1945-or-before", replacements=mock_replacements) == "1945_before"
    assert (
        clean_name("  1945  or  before   ", replacements=mock_replacements)
        == "1945_before"
    )
    assert (
        clean_name("  1945  to  1961   ", replacements=mock_replacements) == "1945_1961"
    )
    assert clean_name("1945 or before", replacements=mock_replacements) == "1945_before"
    assert clean_name("1946-1960", replacements=mock_replacements) == "1946_1960"
    assert clean_name("1986 or after", replacements=mock_replacements) == "1986_after"
    assert clean_name("   1986  (1)  ", replacements=mock_replacements) == "1986_1"
    assert clean_name("1996(1)", replacements=mock_replacements) == "1996_1"
    assert clean_name("2011 to 2015", replacements=mock_replacements) == "2011_2015"


def test_normalize_column_name():
    # FIXME upgrade actual replacements based on this dict
    mock_replacements = {
        # "total": "total",
        "movable": "mobile",
        # "mobile": "mobile",
        "apartment": "apartments",
        # "single-attached": "single_attached",
        # "Other single-attached house": "other_single_attached",
        # "Other single-attached house 3 (42)": "other_single_attached",
        # "single-detached": "single_detached",
        # "Other dwelling (38)": "other_dwelling",
        "fewer": "apartment<5",
        "more": "apartment>5",
        "semi-": "semi_detached",
        "duplex": "apartment_duplex",
    }

    assert normalize_column_name("Total", mock_replacements) == "total"
    assert normalize_column_name("Apartment", mock_replacements) == "apartments"
    assert (
        normalize_column_name("Single attached", mock_replacements) == "single_attached"
    )
    assert (
        normalize_column_name("Other single-attached house", mock_replacements)
        == "other_single_attached"
    )
    assert (
        normalize_column_name("Other single-attached house 3 (42)", mock_replacements)
        == "other_single_attached"
    )
    assert (
        normalize_column_name("Single Detached", mock_replacements) == "single_detached"
    )
    assert (
        normalize_column_name("Other dwelling (274)", mock_replacements)
        == "other_dwelling"
    )

    assert (
        normalize_column_name("  Apartment: five or more storeys", mock_replacements)
        == "apartment>5"
    )

    assert (
        normalize_column_name("  Apartment, detached duplex", mock_replacements)
        == "apartment_duplex"
    )
    assert (
        normalize_column_name("Single-detached house", mock_replacements)
        == "single_detached"
    )
    assert (
        normalize_column_name(
            "Apartment in a building that has five or more storeys", mock_replacements
        )
        == "apartment>5"
    )
    assert (
        normalize_column_name("Other attached dwelling", mock_replacements)
        == "other_attached_dwelling"
    )
    assert (
        normalize_column_name("  Apartment or flat in a duplex", mock_replacements)
        == "apartment_duplex"
    )
    assert (
        normalize_column_name(
            "  Apartment in a building that has fewer than five storeys",
            mock_replacements,
        )
        == "apartment<5"
    )
    assert (
        normalize_column_name("  Other single-attached house", mock_replacements)
        == "other_single_attached"
    )
    assert normalize_column_name("  Row house", mock_replacements) == "row"
    assert (
        normalize_column_name("  Semi-detached house", mock_replacements)
        == "semi_detached"
    )
    assert normalize_column_name("Movable dwelling", mock_replacements) == "mobile"


def test_clean_col_names():
    mock_replacements = {
        # "total": "total",
        "movable": "mobile",
        # "mobile": "mobile",
        "apartment": "apartments",
        # "single-detached": "single_detached",
        "fewer": "apartment<5",
        "more": "apartment>5",
        "semi-": "semi_detached",
        "duplex": "apartment_duplex",
    }

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
    assert clean_col_names(input_cols, mock_replacements) == expected
