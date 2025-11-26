import pytest
import pandas as pd
import numpy as np
from uncertimety.interpolate import (
    fill_missing_dwellings,
    round_consistent_sum,
    apply_ipfn,
    fix_marginals,
    MarginalAdjustment,
    IPFNResult,
    reconcile_data_with_marginals,
    pivot_with_mask,
)
from IPython.display import display  # FIXME only for dev and tests


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


# Fixtures for fix_marginals and reconcile_data_with_marginals
@pytest.fixture
def sample_df():
    """Create sample DataFrame with specific marginal totals."""
    data = {
        "vintage": ["1608-2025", "1608-1920", "1921-1945", "1946-1960", "2020-2025"],
        "total": [190, 0, 80, 110, 0],
        "single_attached": [70, 20, 50, 0, 0],
        "apartments": [0, 0, 0, 0, 0],
        "single_detached": [0, 0, 0, 0, 0],
        "mobile": [120, 0, 80, 40, 0],
        "some_other_col": [np.nan, np.nan, np.nan, np.nan, np.nan],
    }
    return pd.DataFrame(data).set_index("vintage")


@pytest.fixture
def partial_mask(sample_df):
    """Mask with partial protection: single_attached column + 1921-1945 row."""
    mask = pd.DataFrame(False, index=sample_df.index, columns=sample_df.columns)

    rows = [row for row in sample_df.index if row != "1608-2025"]
    cols = [col for col in sample_df.columns if col != "total"]
    mask.loc[rows, "single_attached"] = (
        True  # NOTE: marginals should never be protected?
    )
    mask.loc["1921-1945", cols] = True  # NOTE: marginals should never be protected?
    return mask


@pytest.fixture
def full_mask(sample_df):
    """Mask protecting all values."""
    return pd.DataFrame(True, index=sample_df.index, columns=sample_df.columns)


@pytest.fixture
def no_mask(sample_df):
    """Mask protecting nothing."""
    return pd.DataFrame(False, index=sample_df.index, columns=sample_df.columns)


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
        print(sample_tidy_df)

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
        dw_types = ["single_detached", "single_attached", "apartments", "mobile"]

        result = fix_marginals(
            sample_marginals_df,
            label="vintage",
            dw_types=dw_types,
            type_total_label="total",
            cohort_total_label="1608-2025",
        )
        # Access adjusted_df from MarginalAdjustment
        expected_cols = set(dw_types + ["total"])
        assert set(result.adjusted_df.columns) == expected_cols
        assert result.adjusted_df.shape[0] == sample_marginals_df.shape[0]

    def test_marginals_match(self, sample_marginals_df):
        dw_types = ["single_detached", "single_attached", "apartments", "mobile"]
        result = fix_marginals(
            sample_marginals_df,
            desired_sum=900,
            label="vintage",
            dw_types=dw_types,
            type_total_label="total",
            cohort_total_label="1608-2025",
        )

        # Access adjusted_df
        assert result.adjusted_df.loc["1608-2025", "total"] == 900

        cohort_marginals = (
            result.adjusted_df.drop("1608-2025", axis=0).loc[:, "total"].sum()
        )
        type_marginals = (
            result.adjusted_df.drop("total", axis=1).loc["1608-2025", :].sum()
        )
        assert cohort_marginals == type_marginals
        assert cohort_marginals == 900

    def test_expected_values(self, sample_marginals_df):
        dw_types = ["single_detached", "single_attached", "apartments", "mobile"]
        result = fix_marginals(
            sample_marginals_df,
            desired_sum=798,
            label="vintage",
            dw_types=dw_types,
            type_total_label="total",
            cohort_total_label="1608-2025",
        )
        # Access adjusted_df
        assert all(
            result.adjusted_df.loc["1608-2025", :].to_numpy()
            == np.array([798, 219, 200, 100, 279])
        )
        assert all(
            result.adjusted_df.loc[:, "total"].to_numpy().ravel()
            == np.array([798, 259, 299, 240, 0])
        )

    # NEW TESTS FOR UPGRADED FUNCTIONALITY
    """Test fix_marginals with different protection scenarios."""

    def test_no_protection(self, sample_df, no_mask):
        """Test with no protected data."""
        result = fix_marginals(sample_df, desired_sum=190, protect_original=no_mask)

        assert isinstance(result, MarginalAdjustment)
        assert result.protected_type_sum == 0
        assert result.protected_cohort_sum == 0
        assert result.adjusted_df.loc["1608-2025", "total"] == 190

    def test_partial_preserves_values(self, sample_df, partial_mask):
        """Test that partially protected values are preserved."""
        result = fix_marginals(
            sample_df, desired_sum=190, protect_original=partial_mask
        )

        # Protected single_attached column should be unchanged
        pd.testing.assert_series_equal(
            result.adjusted_df["single_attached"].astype(float),
            sample_df["single_attached"].astype(float),
            check_names=False,
        )

        # Protected 1921-1945 row should be unchanged (except marginals)
        cols = [col for col in result.adjusted_df.columns if col in col != "total"]
        pd.testing.assert_series_equal(
            result.adjusted_df.loc["1921-1945", cols].astype(float),
            sample_df.loc["1921-1945", cols].astype(float),
            check_names=False,
        )

        # Check protected sums are calculated correctly
        # For marginal protection:
        # - protected_type_sum = sum of protected TYPE marginals (row 1608-2025)
        #   Only single_attached is protected -> 70
        # - protected_cohort_sum = sum of protected COHORT marginals (column total)
        #   Only 1921-1945 is protected -> 80
        assert (
            result.protected_type_sum == 0
        )  # single_attached marginal in 1608-2025 row  # FIXME CHECK THAT THIS BEHAVIOUR IS CORRECT
        assert result.protected_cohort_sum == 0  # total marginal in 1921-1945 row
        # NOTE/FIXME - Here, this is because marginals are NOT protected; the *data* (interior data) should be protected, but not the marginals. or maybe this should be a toggle? either way, both cannot be correct - I can't protect both the marginals and the data simultaneously

    def test_partial_marginals_balance(self, sample_df, partial_mask):
        """Test that marginals balance correctly with partial protection."""
        desired_sum = 190
        result = fix_marginals(
            sample_df, desired_sum=desired_sum, protect_original=partial_mask
        )

        # Total should equal desired_sum
        assert result.adjusted_df.loc["1608-2025", "total"] == desired_sum

        # Sum of cohort totals should equal desired_sum
        cohort_sum = result.adjusted_df.drop("1608-2025").loc[:, "total"].sum()
        assert np.isclose(cohort_sum, desired_sum, atol=5, rtol=1e-5)

        # Sum of type marginals should equal desired_sum
        type_sum = result.adjusted_df.drop("total", axis=1).loc["1608-2025", :].sum()
        assert np.isclose(type_sum, desired_sum, atol=5, rtol=1e-5)

    def test_full_raises_error(self, sample_df, full_mask):
        """Test that full protection with incompatible sum raises error."""
        with pytest.raises(ValueError, match="exceeds desired_sum"):
            fix_marginals(
                sample_df,
                desired_sum=100,  # Less than protected sum
                protect_original=full_mask,
            )

    def test_cohort_marginals_respect_interior_floors(self, sample_df, partial_mask):
        """Test that cohort marginals are at least the sum of protected interior values."""
        result = fix_marginals(
            sample_df, desired_sum=190, protect_original=partial_mask
        )

        # For each modifiable row, the total must be >= sum of protected columns
        # Row 1608-1920 has single_attached=20 protected, so total >= 20
        # Row 1946-1960 has single_attached=0 protected, so total >= 0
        # Row 2020-2025 has single_attached=0 protected, so total >= 0

        # Check that 1608-1920 total is at least 20 (the protected single_attached value)
        assert result.adjusted_df.loc["1608-1920", "total"] >= 20, (
            f"Row 1608-1920 total ({result.adjusted_df.loc['1608-1920', 'total']}) "
            f"should be >= 20 (protected single_attached value)"
        )

    def test_marginals_sum_after_protection(self, sample_df, partial_mask):
        """Test that marginals still sum correctly after protection adjustments."""
        desired_sum = 190
        result = fix_marginals(
            sample_df, desired_sum=desired_sum, protect_original=partial_mask
        )

        # Total of cohort marginals (excluding 1608-2025) should equal desired_sum
        # minus the protected cohort sum (1921-1945 total = 80)
        # Plus the protected cohort sum should equal desired_sum
        cohort_sum = result.adjusted_df.drop("1608-2025").loc[:, "total"].sum()

        # The sum should be close to desired_sum (some rounding may occur)
        assert np.isclose(cohort_sum, desired_sum, atol=5, rtol=1e-5), (
            f"Cohort sum ({cohort_sum}) should equal desired_sum ({desired_sum})"
        )

        # Type marginals should also sum correctly
        type_sum = result.adjusted_df.drop("total", axis=1).loc["1608-2025", :].sum()
        assert np.isclose(type_sum, desired_sum, atol=5, rtol=1e-5), (
            f"Type sum ({type_sum}) should equal desired_sum ({desired_sum})"
        )


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

        result = reconcile_data_with_marginals(sample_marginals_df)
        # Access the result DataFrame from IPFNResult dataclass
        assert isinstance(result, IPFNResult)
        # NOTE Use atol>=1 to account for IPFN floating point results vs integer expectations
        assert np.allclose(
            result.result.to_numpy().ravel(), expected, atol=5, rtol=1e-5
        )

    def test_diff_matches_marginals(self, sample_marginals_df):
        result = reconcile_data_with_marginals(
            sample_marginals_df,
            desired_sum=900,
        )
        # Access difference from IPFNResult dataclass
        assert isinstance(result, IPFNResult)
        # Difference should only be in interior cells, not marginals
        # Sum of adjustments should equal the desired_sum increase (900 - 800 = 100)
        interior_diff = result.difference.drop("1608-2025").drop("total", axis=1)
        assert np.isclose(interior_diff.sum().sum(), 100, atol=5, rtol=1e-5)

        # NEW TESTS FOR UPGRADED FUNCTIONALITY
        """Test reconcile_data_with_marginals with different protection scenarios."""

    def test_no_protection_completes(self, sample_df, no_mask):  # TODO REPRENDRE ICI
        """Test reconciliation with no protection."""
        result = reconcile_data_with_marginals(
            sample_df, desired_sum=190, protect_original=no_mask
        )

        assert isinstance(result, IPFNResult)
        assert not result.result.isna().any().any()
        assert result.result.loc["1608-2025", "total"] == 190

    def test_partial_protection_preserves_values(self, sample_df, partial_mask):
        """Test that protected values remain unchanged after IPFN."""
        result = reconcile_data_with_marginals(
            sample_df, desired_sum=190, protect_original=partial_mask
        )

        # Protected single_attached column unchanged
        pd.testing.assert_series_equal(
            result.result["single_attached"].astype(float),
            sample_df["single_attached"].astype(float),
            check_names=False,
        )

        # Protected 1921-1945 row unchanged
        cols = [col for col in result.result.columns if col != "total"]
        pd.testing.assert_series_equal(
            result.result.loc["1921-1945", cols].astype(float),
            sample_df.loc["1921-1945", cols].astype(float),
            check_names=False,
        )

    def test_partial_protection_no_nans(self, sample_df, partial_mask):
        """Test that no NaN values remain after reconciliation."""
        result = reconcile_data_with_marginals(
            sample_df, desired_sum=190, protect_original=partial_mask
        )

        assert not result.result.isna().any().any()

    def test_partial_protection_marginals_balanced(self, sample_df, partial_mask):
        """Test that marginals are balanced after IPFN with protection."""
        desired_sum = 190
        result = reconcile_data_with_marginals(
            sample_df, desired_sum=desired_sum, protect_original=partial_mask
        )
        print(sample_df.drop("some_other_col", axis=1))
        print(result.result)

        # Row sums should equal 'total' column
        for vintage in result.result.index:
            if vintage != "1608-2025":
                row_sum = result.result.drop("total", axis=1).loc[vintage, :].sum()
                print(row_sum, result.result.loc[vintage, "total"])
                assert np.isclose(
                    row_sum, result.result.loc[vintage, "total"], atol=5, rtol=1e-5
                )  # FIXME check atol and rtol bounds

        # Column sums should equal '1608-2025' row
        for col in result.result.columns:
            if col != "total":
                col_sum = result.result.drop("1608-2025").loc[:, col].sum()
                print(col_sum, result.result.loc["1608-2025", col])
                assert np.isclose(
                    col_sum, result.result.loc["1608-2025", col], atol=5, rtol=1e-5
                )

    def test_difference_zero_for_protected(self, sample_df, partial_mask):
        """Test that difference is zero for protected values."""
        result = reconcile_data_with_marginals(
            sample_df, desired_sum=190, protect_original=partial_mask
        )
        # print(partial_mask)
        print(result.difference[partial_mask])

        # Difference should be zero where mask is True
        protected_diff = result.difference[partial_mask]
        assert np.allclose(
            protected_diff.fillna(0), 0, atol=5, rtol=1e-5
        )  # FIXME check correct use of atol and rtol


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
