import pytest
import pandas as pd
import numpy as np
from uncertimety.interpolate import (
    fill_missing_dwellings,
    round_consistent_sum,
    apply_ipfn,
    fix_marginals,
    reconcile_data_with_marginals,
    pivot_with_mask,
)


@pytest.fixture
def sample_tidy_df():
    """Create a sample DataFrame with NaN values for testing."""
    return pd.DataFrame(
        {
            "census_year": [1981, 1991, 2001, 1981, 1991, 2001, 1981, 1991, 2001],
            "type": [
                "total",
                "total",
                "total",
                "single_detached",
                "single_detached",
                "single_detached",
                "apartments",
                "apartments",
                "apartments",
            ],
            "vintage": [
                "1608-2025",
                "1608-2025",
                "1608-2025",
                "1608-1920",
                "1608-1920",
                "1608-1920",
                "1921-1945",
                "1921-1945",
                "1921-1945",
            ],
            "dwellings": [1000, np.nan, 1500, 500, np.nan, 450, np.nan, 200, np.nan],
        }
    )


@pytest.fixture
def sample_ipfn_array():
    arr = np.array(
        [
            [40, 30, 20, 10],
            [35, 50, 100, 75],
            [30, 80, 70, 120],
            [20, 30, 40, 50],
        ]
    )
    return arr


@pytest.fixture
def expected_ipfn_results():
    # Adapted from Wikipedia example: https://en.wikipedia.org/wiki/Iterative_proportional_fitting#Example
    target_sum_cols = np.array([150, 300, 400, 150])  # axis=1
    target_sum_rows = np.array([200, 300, 400, 100])  # axis=0
    expected = np.array(
        [
            [64.61, 46.28, 35.42, 3.83],
            [49.95, 68.15, 156.49, 25.37],
            [56.70, 144.40, 145.06, 53.76],
            [28.74, 41.18, 63.03, 17.03],
        ]
    )
    return target_sum_cols, target_sum_rows, expected


@pytest.fixture
def sample_marginals_df():
    data = {
        "vintage": ["1608-2025", "1608-1920", "1921-1945", "1946-1960", "2020-2025"],
        "total": [800, 260, 300, 240, 0],
        "single_attached": [100, 20, 50, 30, 0],
        "apartments": [220, 80, 100, 40, 0],
        "single_detached": [280, 100, 120, 60, 0],
        "mobile": [200, 80, 80, 40, 0],
        "some_other_col": [None, None, None, None, None],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_marginals_mask(sample_marginals_df):
    mask = pd.DataFrame(
        True, index=sample_marginals_df.index, columns=sample_marginals_df.columns
    )
    mask["some_other_col"] = False  # TODO test with a 'false' at 0?
    mask.loc[2:3, "mobile"] = False
    mask.loc[1:2, "single_attached"] = False
    mask.loc[4, "apartments"] = False
    return mask


@pytest.fixture
def simple_df():
    return pd.DataFrame(
        {
            "vintage": ["1608-1920", "1608-1920", "1921-1945", "1921-1945"],
            "type": ["apartments", "single_detached", "apartments", "single_detached"],
            "dwellings": [100, 200, 300, 400],
        }
    )


@pytest.fixture
def simple_mask():
    # Trusted: first apartments (100) and last single_detached (400)
    """FIXME should be created directly from simple_df"""
    mask = pd.DataFrame(
        {
            "vintage": ["1608-1920", "1608-1920", "1921-1945", "1921-1945"],
            "type": ["apartments", "single_detached", "apartments", "single_detached"],
            "dwellings": [True, False, False, True],
        }
    )
    return mask


class TestFillMissingDwellings:
    def test_no_nans(self, sample_tidy_df):
        """Test that the function removes all NaN values in the dwellings column."""
        # Count NaNs before
        nan_count_before = sample_tidy_df["dwellings"].isna().sum()
        assert nan_count_before > 0, "Test data should contain NaNs to start with"

        # Apply function
        filled_df = fill_missing_dwellings(sample_tidy_df)
        print(filled_df["dwellings"])

        # Verify no NaNs remain
        assert filled_df["dwellings"].isna().sum() == 0, "All NaNs should be filled"

    def test_preserves_structure(self, sample_tidy_df):
        """Test that the function preserves DataFrame structure."""
        filled_df = fill_missing_dwellings(sample_tidy_df)

        # Check shape and columns
        assert filled_df.shape[0] == sample_tidy_df.shape[0], (
            "Row count should be preserved"
        )
        assert list(filled_df.columns) == list(sample_tidy_df.columns), (
            "Column structure should be preserved"
        )

        # Check that groups are preserved
        original_groups = sample_tidy_df.groupby(["type", "vintage"]).size()
        filled_groups = filled_df.groupby(["type", "vintage"]).size()
        pd.testing.assert_series_equal(original_groups, filled_groups)

    def test_correct_values(self, sample_tidy_df):
        """Test that values are filled correctly within each group."""
        filled_df = fill_missing_dwellings(sample_tidy_df)

        # Check specific known cases
        # For 'total' in '1608-2025', the NaN in 1991 should be filled with 1500
        total_row = filled_df[
            (filled_df["type"] == "total")
            & (filled_df["vintage"] == "1608-2025")
            & (filled_df["census_year"] == 1991)
        ]
        assert total_row["dwellings"].values[0] == 1500

        # For 'single_detached' in '1608-1920', the NaN in 1991 should be filled with 450
        sd_row = filled_df[
            (filled_df["type"] == "single_detached")
            & (filled_df["vintage"] == "1608-1920")
            & (filled_df["census_year"] == 1991)
        ]
        assert sd_row["dwellings"].values[0] == 450

        # For 'apartments' in '1921-1945', the NaN in 1981 should be filled with 200
        apt_row_1981 = filled_df[
            (filled_df["type"] == "apartments")
            & (filled_df["vintage"] == "1921-1945")
            & (filled_df["census_year"] == 1981)
        ]
        assert apt_row_1981["dwellings"].values[0] == 200

    def test_missing_columns(self):
        """Test that the function raises an error when required columns are missing."""
        incomplete_df = pd.DataFrame(
            {
                "census_year": [1981, 1991],
                "type": ["total", "total"],
                # Missing 'vintage' column
                "dwellings": [1000, np.nan],
            }
        )

        with pytest.raises(ValueError, match="missing required columns"):
            fill_missing_dwellings(incomplete_df)

    def test_all_nan_group_fails(self):
        """Test behavior when a group has all NaN values for dwellings."""
        df_with_all_nan_group = pd.DataFrame(
            {
                "census_year": [1981, 1991, 2001, 1981, 1991, 2001],
                "type": ["total", "total", "total", "missing", "missing", "missing"],
                "vintage": [
                    "1608-2025",
                    "1608-2025",
                    "1608-2025",
                    "1608-1920",
                    "1608-1920",
                    "1608-1920",
                ],
                "dwellings": [1000, np.nan, 1500, np.nan, np.nan, np.nan],
            }
        )
        with pytest.raises(ValueError, match="NaN values remain"):
            fill_missing_dwellings(df_with_all_nan_group)


class TestRoundConsistentSum:
    def test_sum_preserved(self):
        arr = [1.2, 2.3, 3.9]
        desired_sum = 8
        result = round_consistent_sum(arr, desired_sum)
        assert sum(result) == desired_sum

    def test_order_preserved(self):
        arr = [1.7, 2.2, 3.1, 4.9]
        desired_sum = 13
        result = round_consistent_sum(arr, desired_sum)
        # Check that the length and order (only values rounded) is preserved.
        assert isinstance(result, list)
        assert len(result) == len(arr)
        # FIXME check order, maybe using all() and zip()?

    def test_add_by_weight(self):
        arr = [10, 20, 30, 40]
        desired_sums = [102, 107, 112]
        expected = [
            [10, 20, 31, 41],
            [11, 21, 32, 43],
            [11, 22, 34, 45],
        ]

        for i, desired_sum in enumerate(desired_sums):
            result = round_consistent_sum(arr, desired_sum)
            assert sum(result) == desired_sum
            assert result == expected[i]


class TestApplyIpfn:
    def test_expected_results(self, sample_ipfn_array, expected_ipfn_results):
        arr = np.array(sample_ipfn_array)
        col_target, row_target, expected = expected_ipfn_results

        # here, col_target represents the desired sum of each row (sum over columns, axis=1), and row_target represents the desired sum of each column (sum over rows, axis=0)

        result = apply_ipfn(
            arr,
            aggregates=[col_target, row_target],
            dimensions=[[0], [1]],  # FIXME notation is confusing
            convergence_rate=1e-6,
            max_iter=10,
        )
        print(result)
        # check sums are close
        assert all(np.isclose(result.sum(axis=0), row_target))
        assert all(np.isclose(result.sum(axis=1), col_target))

        # check marginals sum to same total
        assert sum(col_target) == sum(row_target)

        # check result sums to total
        assert np.isclose(result.sum(), sum(col_target))

        # check individual values are relatively close to expected values
        assert all(
            [all(row) for row in np.isclose(result, expected, rtol=1e-2)]
        )  # first row fails at rtol=1e-3

    def test_simple_ipfn(self):
        target_row = np.array([11, 9, 8])  # sum of each column
        target_col = np.array([5, 15, 8])  # sum of each row
        arr = np.array(
            [
                [1, 2, 1],
                [3, 5, 5],
                [6, 2, 2],
            ]
        )
        expected = np.array(
            [
                [1.51, 2.31, 1.18],
                [4.20, 5.35, 5.45],
                [5.28, 1.34, 1.37],
            ]
        )

        result = apply_ipfn(
            arr, aggregates=[target_col, target_row], dimensions=[[0], [1]]
        )
        # check sums are close
        assert all(np.isclose(result.sum(axis=0), target_row))
        assert all(np.isclose(result.sum(axis=1), target_col))

        # check marginals sum to same total
        assert sum(target_col) == sum(target_row)

        # check result sums to total
        assert np.isclose(result.sum(), sum(target_col))

        # check individual values are relatively close to expected values
        assert all(
            [all(row) for row in np.isclose(result, expected, rtol=1e-2)]
        )  # first and last row fails at rtol=1e-3


class TestFixMarginals:
    def test_correct_shape(self, sample_marginals_df):
        # Use dw_types that might come from config plus ensure inclusion of 'total'
        dw_types = ["single_detached", "single_attached", "apartments", "mobile"]

        result = fix_marginals(
            sample_marginals_df,
            label="vintage",
            dw_types=dw_types,
            type_total_label="total",
            cohort_total_label="1608-2025",
        )
        # Expect same number of rows as input (with vintage as index) and columns = dw_types U {total}
        expected_cols = set(dw_types + ["total"])
        assert set(result.columns) == expected_cols
        assert result.shape[0] == sample_marginals_df.shape[0]

    def test_marginals_match(self, sample_marginals_df):
        # Here we expect that the adjusted marginals will have the overall totals matching.
        dw_types = ["single_detached", "single_attached", "apartments", "mobile"]
        result = fix_marginals(
            sample_marginals_df,
            desired_sum=900,
            label="vintage",
            dw_types=dw_types,
            type_total_label="total",
            cohort_total_label="1608-2025",
        )
        # The "total" values by type and cohort should match, and be consistent with the desired sum

        # Check that the desired_sum was enforced
        assert result.loc["1608-2025", "total"] == 900

        # Check that the cohort sums and type sums match with the desired_sum
        cohort_marginals = result.drop("1608-2025", axis=0).loc[:, "total"].sum()
        type_marginals = result.drop("total", axis=1).loc["1608-2025", :].sum()
        assert cohort_marginals == type_marginals, (
            "Row totals do not match cohort total as expected."
        )
        assert cohort_marginals == 900

    def test_expected_values(seld, sample_marginals_df):
        dw_types = ["single_detached", "single_attached", "apartments", "mobile"]
        result = fix_marginals(
            sample_marginals_df,
            desired_sum=798,
            label="vintage",
            dw_types=dw_types,
            type_total_label="total",
            cohort_total_label="1608-2025",
        )
        assert all(
            result.loc["1608-2025", :].to_numpy() == np.array([798, 219, 200, 100, 279])
        )  # type marginals
        assert all(
            result.loc[:, "total"].to_numpy().ravel()
            == np.array([798, 259, 299, 240, 0])
        )  # cohort marginals


class TestReconcileDataWithMarginals:
    def test_returns_expected_result(self, sample_marginals_df):
        expected = np.array(
            [
                800,
                220,
                200,
                100,
                280,
                260,
                75,
                74,
                18,
                92,
                300,
                87,
                69,
                41,
                103,
                240,
                58,
                57,
                41,
                85,
                0,
                0,
                0,
                0,
                0,
            ]
        )

        result, _ = reconcile_data_with_marginals(sample_marginals_df)
        assert all(np.isclose(result.to_numpy().ravel(), expected))

    def test_diff_matches_marginals(self, sample_marginals_df):
        result, diff = reconcile_data_with_marginals(
            sample_marginals_df,
            desired_sum=900,
        )
        assert np.isclose(diff.sum(), 100, atol=5e-2)


class TestPivotWithMask:
    def test_sample_df(self, simple_df, simple_mask):
        """Test the pivot_with_mask function."""
        # Get sample tidy data, and a sample dataframe "mask" where some values are trusted (True) and other are not (False), keeping all other columns equal
        df, mask_df = pivot_with_mask(simple_df, simple_mask)

        # Check results
        expected_df = pd.DataFrame(
            {
                "apartments": [100, 300],
                "single_detached": [200, 400],
            },
            index=pd.Index(["1608-1920", "1921-1945"], name="vintage"),
        )
        expected_df.columns.name = "type"

        expected_mask = pd.DataFrame(
            {
                "apartments": [True, False],
                "single_detached": [False, True],
            },
            index=pd.Index(["1608-1920", "1921-1945"], name="vintage"),
        )
        expected_mask.columns.name = "type"

        pd.testing.assert_frame_equal(df, expected_df)
        pd.testing.assert_frame_equal(mask_df, expected_mask)

    def test_pivot_with_mask_from_marginals(
        self, sample_marginals_df, sample_marginals_mask
    ):  # TODO validate relevance? is this only to 100% confirm that the mask 'followed'?
        """pivot_with_mask should pivot values and correctly align the boolean mask."""
        tidy = sample_marginals_df.melt(
            id_vars="vintage", var_name="type", value_name="dwellings"
        )
        tidy_mask = sample_marginals_mask.melt(
            id_vars="vintage", var_name="type", value_name="dwellings"
        )
        tidy_mask["vintage"] = tidy["vintage"]

        # --- run function under test ---
        df, mask_df = pivot_with_mask(tidy, tidy_mask)
        print(df)

        # --- expected pivoted dataframe (values) ---
        expected_df = pd.DataFrame(
            {
                "apartments": [80, 220, 100, 40, 0],
                "mobile": [80, 200, 80, 40, 0],
                "single_attached": [20, 100, 50, 30, 0],
                "single_detached": [100, 280, 120, 60, 0],
                "some_other_col": [None, None, None, None, None],
                "total": [260, 800, 300, 240, 0],
            },
            index=pd.Index(
                ["1608-1920", "1608-2025", "1921-1945", "1946-1960", "2020-2025"],
                name="vintage",
            ),
        )
        expected_df.columns.name = "type"
        print(expected_df)

        # --- expected mask (boolean) ---
        expected_mask = pd.DataFrame(
            {
                "apartments": [
                    True,
                    True,
                    True,
                    True,
                    False,
                ],  # apartments 1608-1920 → False
                "mobile": [True, True, False, False, True],
                "single_attached": [False, True, False, True, True],
                "single_detached": [
                    True,
                    True,
                    True,
                    True,
                    True,
                ],  # single_detached 1921-1945 → False
                "some_other_col": [False, False, False, False, False],
                "total": [True, True, True, True, True],
            },
            index=pd.Index(
                ["1608-1920", "1608-2025", "1921-1945", "1946-1960", "2020-2025"],
                name="vintage",
            ),
        )
        expected_mask.columns.name = "type"

        # --- assertions ---
        pd.testing.assert_frame_equal(df, expected_df, check_dtype=False)
        pd.testing.assert_frame_equal(mask_df, expected_mask, check_dtype=False)
