import math
import pandas as pd
import numpy as np
from uncertimety.logger import init_logger
from IPython.display import display  # FIXME only for dev and tests

logger = init_logger()


def interpolate(infile):
    df = pd.read_parquet(infile)

    raw_mask = get_rawdata_mask(df)

    # Fill missing values using interpolation
    interpolated = fill_missing_dwellings(df)

    # run check_marginals (from dataprep) or a similar function to verify that the marginals are either the same, or closer than they were, after the fill_missing_values

    return interpolated, raw_mask


def get_rawdata_mask(
    df: pd.DataFrame, by: list[str] = ["census_year", "vintage", "type"]
):
    """
    Compute a boolean mask for the raw data based on non-NaN aggregations.

    This function groups the input DataFrame by the specified columns,
    sums the numeric data with a minimum count of 1 (so that if all values are NaN, the result remains NaN), resets the index, and then returns a boolean DataFrame, indicating which values contain data (i.e., are not nans)

    Args:
        df: A pandas DataFrame containing the raw data.
        by: A list of column names to group by. Defaults to ['census_year', 'vintage', 'type'].

    Returns:
        A pandas DataFrame of booleans with the same shape as the reset grouped DataFrame,
        where each True value indicates that the corresponding aggregated value is not NaN.

    Examples:
        >>> data = {'census_year': [2001, 2001, 2011],
        ...         'vintage': ['1608-2025', '1608-2025', '1608-1920'],
        ...         'type': ['total', 'apartments', 'total'],
        ...         'dwellings': [1000, None, 800]}
        >>> df = pd.DataFrame(data)
        >>> get_rawdata_mask(df)
             census_year vintage      type  dwellings
        0         2001  1608-2025     total       True
        1         2001  1608-2025  apartments      False
        2         2011  1608-1920     total       True
    """
    mask = df.groupby(by).sum(min_count=1).reset_index().notna()
    return mask


def fill_missing_dwellings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing dwelling values using backward fill within each (type, vintage) group.

    This function fills NaN values in the 'dwellings' column by looking at future values within the same type and vintage group and propagating them backward.

    This sets the minimal dwelling count that must have been built earlier, knowing this dwelling count (by type and vintage) was observed in later censuses.

    Args:
        df: DataFrame with 'census_year', 'type', 'vintage', and 'dwellings' columns

    Returns:
        DataFrame with NaN values in 'dwellings' filled using backward fill

    Examples:
        >>> df = pd.DataFrame({
        ...     'census_year': [1981, 1991, 2001, 1981, 1991, 2001],
        ...     'type': ['total', 'total', 'total', 'single_detached', 'single_detached', 'single_detached'],
        ...     'vintage': ['1608-2025', '1608-2025', '1608-2025', '1608-1920', '1608-1920', '1608-1920'],
        ...     'dwellings': [1000, np.nan, 1500, 500, np.nan, 450]
        ... })
        >>> filled = fill_missing_dwellings(df)
        >>> filled['dwellings'].isna().sum()
        0
    """
    # FIXME may break if passed an empty df
    # Validate input DataFrame
    required_columns = ["census_year", "type", "vintage", "dwellings"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"DataFrame is missing required columns: {missing_columns}")

    # Group by type and vintage
    groups = df.groupby(["type", "vintage"])

    # Process each group
    filled_groups = []
    for name, group in groups:
        filled_group = group.copy()

        # Apply backward fill on the dwellings column
        filled_group["dwellings"] = filled_group["dwellings"].bfill()
        print(filled_group["dwellings"].iloc[-1])

        if math.isnan(filled_group["dwellings"].iloc[-1]):
            # Edge case where the last value of a group was a nan, leading to unfilled values
            filled_group["dwellings"] = filled_group[
                "dwellings"
            ].ffill()  # NOTE-perhaps add a limit? can this cause issues in other datasets?

        filled_groups.append(filled_group)

    # Combine all processed groups
    result = pd.concat(filled_groups)

    if result.isna().any().any():
        msg = "Some NaN values remain after filling. Check the input data for completeness."
        logger.warning(msg)
        raise ValueError(msg)

    return result


if __name__ == "__main__":
    infile = "./data/clean/fulldata.parquet"
    df, mask = interpolate(infile)
    display(df)
    display(df.info())
    display(df.where(mask))


# TODO: look at def check_sums in the ipynb files; it seems to be the start of the IPFN procedure. Then, in interpolating the missing data, I discuss padding backwards. however, why not set zeros to known values, then interpolate linearly, but backwards?

# TODO: look at the 'corrected' values below, and at everything with "_fix" suffix
