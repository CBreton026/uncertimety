import math
import pandas as pd
import numpy as np
from operator import itemgetter
from ipfn import ipfn
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


def apply_rawdata_mask(
    df: pd.DataFrame, mask, by: list[str] = ["census_year", "vintage", "type"]
):
    # TODO make sure the mask is passed on the correct data! see comments in temp.ipynb
    rawdata_df = df.groupby(by).sum(min_count=1).reset_index()

    # ensure that the mask is applied to the correct data
    all_equal = all(
        df_col == mask_col for df_col, mask_col in zip(rawdata_df.columns, mask.columns)
    )

    if not all_equal:
        msg = f"Column mismatch: {rawdata_df.columns} vs {mask.columns}"
        logger.error(msg)
        raise ValueError(msg)

    return rawdata_df.where(mask)


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


def round_consistent_sum(arr, desired_sum):
    """_summary_

    Rounds an array of floats to integers while preserving a desired sum,
    using the Largest Remainder Method.

    it does so by calculating the difference between observed and desired sums; attributing it to array number by 'weight'; then rounding back to integers, while minimizing roundoff error.

    This function first scales the input array so that its sum matches
    the desired sum (if necessary). Then, for each element, it computes
    its floor (lower bound) and the remainder. The total difference
    between the desired sum and the sum of floors is then distributed by
    adding one to those elements with the largest remainders.

    adapted from Mikko Rantanen on stackoverflow:  https://stackoverflow.com/questions/792460/how-to-round-floats-to-integers-while-preserving-their-sum

    Args:
        arr (array-like): An array of floats.
        desired_sum (int): The desired total sum after rounding.

    Returns:
        List[int]: A list of rounded integer values that sum to desired_sum.

    Examples:
        >>> round_consistent_sum([1.2, 2.3, 3.9], 8)
        [1, 2, 5]

    # === test
    # c = np.array([1, 2, 3, 4])  #  [10,20,30,40]

    # for tot in range(100, 112):
    #     f = round_consistent_sum(c, tot)
    #     print(f)
    """
    # Convert input to a numpy array of floats
    arr = np.array(arr, dtype=float)

    current_sum = arr.sum()

    # Scale the array to fit the desired sum
    if current_sum != desired_sum:
        # overwrite original array so it sums to desired sum
        arr = arr * (desired_sum / current_sum)

    # Convert the floats back to integers while minimizing roundoff error
    temp = []
    lower_sum = 0
    for i, value in enumerate(arr):
        lower = math.floor(value)  # lower bound
        remainder = value - lower  # roundoff error
        lower_sum += lower
        temp.append((lower, remainder, i))

    # Sort the temp array on the roundoff error (largest first) so we add increments to elements closest to next integer
    # https://stackoverflow.com/questions/6835531/sorting-a-python-array-recarray-by-column)
    temp_sorted = sorted(temp, key=itemgetter(1), reverse=True)

    # Now desired_sum - lowerSum gives us the difference between sums of these
    # arrays. new_arr is ordered in such a way that the numbers closest to the
    # next one are at the top.
    # The number of increments needed so that the sum equals desired_sum
    increments_needed = int(desired_sum - lower_sum)

    # Take the n largest remainders to assign diff
    for i in range(increments_needed):
        # Add 1 to those most likely to round up to the next number so that
        # the difference is nullified.
        lower, remainder, idx = temp_sorted[i]
        temp_sorted[i] = (lower + 1, remainder, idx)

    # sort the array based on the original index.
    return [value for value, remainder, idx in sorted(temp_sorted, key=itemgetter(2))]


def fix_marginals(df: pd.DataFrame, max_iter: int = 500):
    pass


def apply_ipfn(
    arr: np.array,
    aggregates: list[list],
    dimensions: list[list[int]],
    convergence_rate=1e-6,
    max_iter: int = 500,
):
    """

    see https://pypi.org/project/ipfn/

    Args:
        arr (np.array): _description_
        aggregates (list[list]): _description_
        dimensions (list[list[int]]): _description_
        convergence_rate (_type_, optional): _description_. Defaults to 1e-6.
        max_iter (int, optional): _description_. Defaults to 500.

    Returns:
        _type_: _description_
    """

    ipf = ipfn.ipfn(
        arr,
        aggregates,
        dimensions,
        convergence_rate=convergence_rate,
        max_iteration=max_iter,
    )

    result = ipf.iteration()
    # TODO make sure that column order is ok; maybe use a df transform?

    return result


if __name__ == "__main__":
    infile = "./data/clean/fulldata.parquet"
    df, mask = interpolate(infile)
    display(df)
    display(df.info())
    display(apply_rawdata_mask(df, mask))

# TODO: look at the 'corrected' values, and at everything with "_fix" suffix
