import pytest
import pandas as pd
import numpy as np
from uncertimety.interpolate import (
    fill_missing_dwellings,
    round_consistent_sum,
    apply_ipfn,
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
