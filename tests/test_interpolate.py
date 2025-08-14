import pytest
import pandas as pd
import numpy as np
from uncertimety.interpolate import fill_missing_dwellings


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


class TestFillMissingDwellings:
    def test_no_nans(self, sample_tidy_df):
        """Test that the function removes all NaN values in the dwellings column."""
        # Count NaNs before
        nan_count_before = sample_tidy_df["dwellings"].isna().sum()
        assert nan_count_before > 0, "Test data should contain NaNs to start with"

        # Apply function
        filled_df = fill_missing_dwellings(sample_tidy_df)
        print(filled_df['dwellings'])

        # Verify no NaNs remain
        assert filled_df["dwellings"].isna().sum() == 0, "All NaNs should be filled"

    def test_preserves_structure(self, sample_tidy_df):
        """Test that the function preserves DataFrame structure."""
        filled_df = fill_missing_dwellings(sample_tidy_df)

        # Check shape and columns
        assert filled_df.shape[0] == sample_tidy_df.shape[0], "Row count should be preserved"
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
