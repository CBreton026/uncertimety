import math
import pandas as pd
import numpy as np
from operator import itemgetter
from ipfn import ipfn
from pathlib import Path
from typing import Optional
from uncertimety.logger import init_logger
from uncertimety.dataprep import load_dataset_config, sort_vintage_labels
from IPython.display import display  # FIXME only for dev and tests


logger = init_logger()

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
CONFIG = DATA_DIR / "config.toml"
if not CONFIG.exists():
    msg = f"No TOML file found at: {CONFIG}"
    logger.error(msg)
    raise FileNotFoundError(msg)

try:
    config = load_dataset_config(CONFIG)
    META_COLS = config["dwelling_stock"]["metadata"]
    HISTORIC_VINTAGES = config["dwelling_stock"]["historic_vintages"]
    TARGET_TYPES = config["dwelling_stock"]["target_dwelling_types"]
    MERGED_COLS = [
        "other_attached_dwelling",
        "other_dwelling",
    ]  # fixed, not from config
except KeyError as err:
    logger.error(f"Missing expected key in config file {CONFIG}: {err}")
    raise


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
    rawdata_df = (
        df.groupby(by).sum(min_count=1).reset_index()
    )  # FIXME replace with get_rawdata_mask?

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
        display(group.head(5))  # FIXME remove
        filled_group = group.copy()

        # TODO FIXME REPRENDRE - here one issue is that since I backfill everything, there may be some categories e.g., single_attached, for which the only available data is very recent; imagine for single_detached, older values were available; then, bfill would project *wrong* values backwards, and would shift the typesplit. I might have to do the 'totals' first, since I have the most data; then do a fix marginals; then bfill? However, doing that would ALSO shift the typesplit. should the bfill interpolation be limited to a given number of sequntial nans? or should I use a combination of ffill/bfill?

        # Apply backward fill on the dwellings column
        filled_group["dwellings"] = filled_group["dwellings"].bfill()
        # FIXME REPRENDRE the backfill is way too high for older data, i.e., data before 19XX. thus, use two passes -- see unfm_fix [ Finalized dataset, 1851-2021]

        # print(filled_group["dwellings"].iloc[-1])  # FIXME logging

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


def fix_marginals(
    df: pd.DataFrame,
    desired_sum=None,
    label: str = "vintage",
    dw_types: list[str] = TARGET_TYPES,
    type_total_label: str = "total",
    cohort_total_label: str = "1608-2025",
) -> pd.DataFrame:
    """
    Adjust the marginals (row and column totals) in the input DataFrame
    using the Largest Remainder Method (via round_consistent_sum) so that
    the component sums match the desired totals.

    The input DataFrame is expected to have a row (indexed by `label`) that
    serves as the cohort total (with label `cohort_total_label`) and a column
    (named by `type_total_label`) that is the type total.

    The function first restricts the DataFrame to the target dwelling type columns.
    To ensure the overall total column isn’t dropped, this revision forces inclusion
    of `type_total_label` within the filtered columns.

    Args:
        df (pd.DataFrame): DataFrame containing marginal data.
        target: (unused) placeholder for potential future use.
        label (str): The categorical column holding the index (e.g., 'vintage').
        dw_types (list[str]): List of dwelling types to process. Assumes type_total_label is included in dw_types
        type_total_label (str): Column name for the type total.
        cohort_total_label (str): Row label for the cohort total.

    Returns:
        pd.DataFrame: A DataFrame with adjusted marginals. The nonzero cells are adjusted
        via the Largest Remainder Method, then the zeros are concatenated back.

    Raises:
        KeyError: If the expected labels are missing from the DataFrame.
    """
    # NOTE In the future, it might be relevant to first treat the target types, and then 'reverse-engineer' the subcategories
    # Set index based on label if available.
    if label in df.columns:
        working_df = df.copy().set_index(label)
    else:
        working_df = df.copy()

    # Ensure we keep only the relevant columns, and convert dataset to float.
    # Force inclusion of type_total_label even if not present in dw_types.
    cols_to_keep = [type_total_label] + [
        col for col in sorted(dw_types) if col in df.columns and col != type_total_label
    ]
    working_df = working_df.loc[:, cols_to_keep].astype("float")

    # FIXME: this function does too much at once, everything above here might best be relegated to the reconcile_data_with_marginals function?

    # Only keep nonzero values; there can be no 'zero' values for the IPFN procedure; small (i.e. close to zero) values must be set to arbitrary small values (e.g., orders of magnitude smaller than actual data) beforehand
    # this method first converts values where condition is false to nans, then drop the nan rows
    nonzeros = working_df.where(working_df > 0).dropna()
    zeros = working_df.where(working_df == 0).dropna()

    # Extract current marginals
    try:
        if desired_sum is None:  # allows to set arbitrary desired_sums
            desired_sum = nonzeros.loc[cohort_total_label, type_total_label]

        cohort_marginals = nonzeros.drop(cohort_total_label, axis=0)[
            type_total_label
        ].to_numpy()
        type_marginals = (
            nonzeros.drop(type_total_label, axis=1)
            .loc[cohort_total_label, :]
            .to_numpy()
        )
    except KeyError as err:
        logger.error(f"Missing expected label in DataFrame: {err}")
        raise

    # Adjust marginals using Largest Remainder Method
    adj_type_marginals = round_consistent_sum(type_marginals, desired_sum)
    adj_cohort_marginals = round_consistent_sum(cohort_marginals, desired_sum)

    # Identify target indexes and columns
    rows = nonzeros.index[nonzeros.index != cohort_total_label]
    cols = [col for col in nonzeros.columns if col != type_total_label]

    # Log the modifications  # TODO also return them, e.g, through a dataclass?
    type_diff = adj_type_marginals - type_marginals
    cohort_diff = adj_cohort_marginals - cohort_marginals

    if type_diff.sum() + cohort_diff.sum() == 0:
        logger.info("Marginals fit; no modifications required.")
    else:
        if sum(type_diff) != 0:
            logger.info(
                f"Modified type marginals: {sum(type_diff)} units ({type_diff}) from {nonzeros.loc[cohort_total_label, cols].to_json()}"
            )

        if sum(cohort_diff) != 0:
            logger.info(
                f"Modified cohort marginals: {sum(cohort_diff)} units ({cohort_diff}) from {nonzeros.loc[rows, type_total_label].to_json()}"
            )

    # Assign results to DataFrame
    # For the type totals: assign to all rows except the cohort total row.
    nonzeros.loc[rows, type_total_label] = pd.Series(adj_cohort_marginals, index=rows)

    # For the cohort total: assign to all columns except the type_total column.
    nonzeros.loc[cohort_total_label, cols] = pd.Series(adj_type_marginals, index=cols)

    # For the desired_sum value
    nonzeros.loc[cohort_total_label, type_total_label] = desired_sum

    # Concat nonzeros and zero data
    result = pd.concat([nonzeros.astype("int"), zeros.astype("int")])

    # col_order = ["total"] + sorted(dw_types)
    return result


def apply_ipfn(
    arr: np.array,
    aggregates: list[list[float]],
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


def pivot_with_mask(
    group, trusted_data_mask, index="vintage", columns="type", values="dwellings"
):
    """
    Pivot the given group DataFrame and its corresponding trusted data mask into wide format.

    Args:
        group (pd.DataFrame): The DataFrame to pivot, typically a group from groupby.
        trusted_data_mask (pd.DataFrame): Boolean mask DataFrame indicating trusted data.
        index (str): Column to use as the new index.
        columns (str): Column to use to make new columns.
        values (str): Column to populate values in the pivoted DataFrame.

    Returns:
        tuple: A tuple (df, mask) where df is the pivoted DataFrame and mask is the pivoted mask DataFrame.
    """
    df = group.pivot(index=index, columns=columns, values=values)
    mask = (
        trusted_data_mask.loc[group.index]
        # .assign(value=rawdata_mask.loc[group.index, values])  # FIXME Useless?
        .pivot(index=index, columns=columns, values=values)
    )
    return df, mask


def reconcile_data_with_marginals(
    df: pd.DataFrame,
    label: str = "vintage",
    desired_sum: int = None,
    dw_types: list[str] = TARGET_TYPES,
    type_total_label: str = "total",
    cohort_total_label: str = "1608-2025",
    convergence_rate=1e-6,
    max_iter: int = 500,
    protect_original: Optional[
        pd.DataFrame
    ] = None,  # pass a mask if you want to protect the original data
) -> pd.DataFrame:
    # NOTE should we protect initial data?

    # accept dataframe
    working_df = df.copy()

    # add option to protect initial data
    if protect_original is not None:
        protected = None
        unprotected = None
        working_df = fix_marginals(
            working_df,
            desired_sum=desired_sum,
            label=label,
            dw_types=dw_types,
            type_total_label=type_total_label,
            cohort_total_label=cohort_total_label,
        )
        working_df = working_df.replace(
            0, 0.1
        )  # FIXME; here, replacing all zeros doesn't work, I need to replace only 'Nan' zeros, and zeros in totals
    else:
        # first, fix_marginals
        working_df = fix_marginals(
            working_df,
            desired_sum=desired_sum,
            label=label,
            dw_types=dw_types,
            type_total_label=type_total_label,
            cohort_total_label=cohort_total_label,
        )  # FIXME: this can return dataframes with zeroes, which can lead to issues in the IPFN. For now, kept for legacy tests; this might need to be corrected, like in the case where protect_original is passed a mask.
    print(working_df)  # TODO delete

    # then, apply_ipfn
    try:
        nonzeros = working_df.where(working_df > 0).dropna()
        zeros = working_df.where(working_df == 0).dropna()
        if nonzeros.empty:
            msg = (
                "No nonzero values found in working dataframe. "
                "Ensure that the dataframe contains valid data before applying IPFN."
            )
            logger.error(msg)
            raise ValueError(msg)
    except ValueError as err:
        msg = (
            f"Error while processing working_df: {err}. "
            "This might be caused by an empty or incorrectly formatted dataframe "
            "returned from fix_marginals."
        )
        logger.error(msg)
        raise RuntimeError(msg)

    cohort_marginals = nonzeros.drop(cohort_total_label, axis=0)[
        type_total_label
    ].to_numpy()
    type_marginals = (
        nonzeros.drop(type_total_label, axis=1).loc[cohort_total_label, :].to_numpy()
    )

    arr_ini = nonzeros.drop(
        index=[cohort_total_label], columns=[type_total_label]
    ).to_numpy()

    # FIXME adjust behaviour to keep original data (trusted_values)

    arr_new = apply_ipfn(
        arr=arr_ini.copy(),
        aggregates=[cohort_marginals, type_marginals],
        dimensions=[[0], [1]],  # dimension over which the 'sum' is made
        convergence_rate=convergence_rate,
        max_iter=max_iter,
    )  # FIXME: if there is only one value for ipfn (40, 0.1, 0.1, 0.1), everything should be alloted to the real value? now, 110 become (107, 2, 0, 0)...

    diff = (arr_new - arr_ini).round(decimals=2)
    logger.debug(f"Initial array before IPFN: {arr_ini.ravel()}")
    logger.debug(f"IPFN adjustment difference (rounded): {diff.ravel()}")
    logger.info(
        f"Sum of IPFN adjustment difference: {diff.sum().round(decimals=2)}"
    )  # NOTE: this should match the marginal adjustment

    target_rows = nonzeros.index[nonzeros.index != cohort_total_label]
    target_cols = [col for col in nonzeros.columns if col != type_total_label]
    nonzeros.loc[target_rows, target_cols] = (
        arr_new.round()
    )  # FIXME, instead of modyfying rows and cols, overwrite with a dataframe of target_rows, target_cols?

    result = pd.concat([nonzeros.astype("int"), zeros.astype("int")])
    # display(result)

    return (
        result,
        diff,
    )  # FIXME improve logging and tests by instead returning a container (dataclass) with relevant information, e.g., result, diff, type and cohort marginals adjustements, etc.?


if __name__ == "__main__":
    infile = "./data/clean/fulldata.parquet"
    data, mask = interpolate(infile)
    display(data)
    display(data.info())
    display(
        apply_rawdata_mask(data, mask)
    )  # TODO use rawdata mask to check the modified values of raw data for quality control

    groups = data.groupby(by=["census_year"])
    for name, group in groups:
        print(f"Group: {name}")

        df = (
            group.drop("census_year", axis=1)
            .pivot(index=["vintage"], columns=["type"])
            .droplevel(axis=1, level=0)
        )  # droplevel removes 'dwellings' from the columns level: (dwellings, total) -> total

        cols = [col for col in TARGET_TYPES if col in df]

        result, diff = reconcile_data_with_marginals(df)
        display(
            df[cols] - result
        )  # TODO compare masked and unmasked versions!! Should I protect the original data, or not? check copilot, and adjust reconcile.. function accordingly.

        # display(group.pivot(index='vintage'))

# TODO: look at the 'corrected' values, and at everything with "_fix" suffix
# Realistically, as soon as I start changing the marginals, I can't only change the interpolated data.

# TODO REPRENDRE consider adding the pre-1941 data (cs_data // old_cs_data), then run, and compare results to the initial dataset in a relplot (see previous to last cell in bac/dmfa_dataprep)


""" TODO plot the data, and compare with initial parquet dataset. maybe save figures? use a notebook to demonstrate the workflow, print the figures?
sns.relplot(
    d,
    x="year",
    y="dwellings",
    hue="type",
    row="vintage",
    col="dataset",  # method, 'res' or 'res_fix'
    kind="line",
    facet_kws={
        "sharey": "row",
    },
)

plt.tight_layout()
plt.show()

# fig, ax = plt.subplots()
# csdw_itp[(csdw_itp.index.get_level_values('vintage')=='0-2025') & (csdw_itp.index.get_level_values('type').isin(agg_types))].unstack(['type','vintage']).plot(ax=ax, legend=True)
# sns.move_legend(ax, loc='center left', bbox_to_anchor=[1,0.5])

"""
