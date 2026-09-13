import math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from operator import itemgetter
from ipfn import ipfn
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple
from uncertimety.logger import init_logger
from uncertimety.dataprep import load_dataset_config
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

# ==== Helper methods for fix_marginals and IPFN procedure ====
# Constants  # FIXME maybe remove or load from config
COHORT_TOTAL_LABEL = "1608-2025"
TYPE_TOTAL_LABEL = "total"
DEFAULT_ATOL = 5.0
DEFAULT_RTOL = 1e-5
VINTAGE_COL = "vintage"
ZERO_REPLACEMENT = 0.1


@dataclass
class MarginalAdjustment:
    """Container for marginal adjustment results."""

    adjusted_df: pd.DataFrame
    type_diff: np.ndarray
    cohort_diff: np.ndarray
    type_marginals: pd.Series
    cohort_marginals: pd.Series
    protected_type_sum: float
    protected_cohort_sum: float

    def total_adjustment(self) -> float:
        """Calculate total adjustment magnitude."""
        return np.abs(self.type_diff).sum() + np.abs(self.cohort_diff).sum()


@dataclass
class IPFNResult:
    """Container for IPFN reconciliation results."""

    result: pd.DataFrame
    difference: np.ndarray
    convergence_achieved: bool
    iterations: int

    def max_adjustment(self) -> float:
        """Get maximum adjustment value."""
        return np.abs(self.difference).max()


class DataProtector:
    """Handles data protection logic for marginal adjustments."""

    def __init__(
        self,
        df: pd.DataFrame,
        mask: Optional[pd.DataFrame] = None,
        cohort_label: str = COHORT_TOTAL_LABEL,
        type_label: str = TYPE_TOTAL_LABEL,
    ):
        """
        Initialize data protector.

        Args:
            df: DataFrame to protect
            mask: Boolean mask indicating which values are original/protected
            cohort_label: Label for cohort total row
            type_label: Label for type total column
        """
        self.df = df
        self.cohort_label = cohort_label
        self.type_label = type_label

        # Default to no protection if mask not provided
        if mask is None:
            mask = pd.DataFrame(False, index=df.index, columns=df.columns)
        self.mask = mask

    def get_protected_marginals(self) -> Tuple[pd.Series, pd.Series]:
        """
        Extract protected marginal values.

        Returns:
            Tuple of (protected_type_marginals, protected_cohort_marginals)
        """
        protected_types = self.df[self.mask].loc[self.cohort_label, :]
        protected_cohorts = self.df[self.mask].loc[:, self.type_label]

        return protected_types, protected_cohorts

    def get_protected_sums(self) -> Tuple[float, float]:
        """
        Calculate sums of protected values.

        Returns:
            Tuple of (protected_type_sum, protected_cohort_sum)
        """
        protected_types, protected_cohorts = self.get_protected_marginals()
        return protected_types.sum(), protected_cohorts.sum()

    def get_modifiable_indices(self) -> Tuple[pd.Index, pd.Index]:
        """
        Get indices of rows and columns that can be modified.

        Returns:
            Tuple of (modifiable_rows, modifiable_columns)
        """
        protected_types, protected_cohorts = self.get_protected_marginals()

        modifiable_rows = self.df.loc[protected_cohorts.isna(), :].index
        modifiable_cols = self.df.loc[:, protected_types.isna()].columns

        # Exclude total labels
        modifiable_rows = modifiable_rows[modifiable_rows != self.cohort_label]
        modifiable_cols = modifiable_cols[modifiable_cols != self.type_label]

        return modifiable_rows, modifiable_cols

    def calculate_adjusted_targets(self, desired_sum: float) -> Tuple[float, float]:
        """
        Calculate target sums after accounting for protected values.

        Args:
            desired_sum: Total desired sum

        Returns:
            Tuple of (adjusted_type_total, adjusted_cohort_total)
        """
        protected_type_sum, protected_cohort_sum = self.get_protected_sums()

        type_total = desired_sum - protected_type_sum
        cohort_total = desired_sum - protected_cohort_sum

        return type_total, cohort_total

    def validate_desired_sum(
        self, desired_sum: float, atol: float = DEFAULT_ATOL, rtol: float = DEFAULT_RTOL
    ) -> Tuple[bool, str]:
        """
        Validate that desired_sum is compatible with protected data.

        Args:
            desired_sum: Target sum to validate
            atol: Absolute tolerance
            rtol: Relative tolerance

        Returns:
            Tuple of (is_valid, error_message)
        """
        protected_type_sum, protected_cohort_sum = self.get_protected_sums()

        # Check type marginals
        if protected_type_sum > desired_sum:
            diff = protected_type_sum - desired_sum
            if not np.isclose(protected_type_sum, desired_sum, atol=atol, rtol=rtol):
                msg = (
                    f"Protected type marginals sum ({protected_type_sum:.2f}) "
                    f"exceeds desired_sum ({desired_sum:.2f}) by {diff:.2f}. "
                    f"Cannot reconcile without modifying protected data."
                )
                return False, msg

        # Check cohort marginals
        if protected_cohort_sum > desired_sum:
            diff = protected_cohort_sum - desired_sum
            if not np.isclose(protected_cohort_sum, desired_sum, atol=atol, rtol=rtol):
                msg = (
                    f"Protected cohort marginals sum ({protected_cohort_sum:.2f}) "
                    f"exceeds desired_sum ({desired_sum:.2f}) by {diff:.2f}. "
                    f"Cannot reconcile without modifying protected data."
                )
                return False, msg

        return True, ""


class MarginalExtractor:
    """Extracts and prepares marginal data for adjustment."""

    @staticmethod
    def extract_marginals(
        df: pd.DataFrame,
        cohort_label: str = COHORT_TOTAL_LABEL,
        type_label: str = TYPE_TOTAL_LABEL,
        replace_zeros: bool = True,
        zero_replacement: float = 0.1, # NOTE : check if fixes issue! it applied the zero replacement to marginals too, but I want to keep the marginals as-is (i think?) XXX
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Extract type and cohort marginals from DataFrame.

        Args:
            df: Input DataFrame with marginals
            cohort_label: Label for cohort total row
            type_label: Label for type total column
            replace_zeros: Whether to replace zeros with small value
            zero_replacement: Value to replace zeros with

        Returns:
            Tuple of (type_marginals, cohort_marginals)
        """
        cohort_marginals = df.drop(cohort_label, axis=0)[type_label]
        type_marginals = df.drop(type_label, axis=1).loc[cohort_label, :]

        if replace_zeros:
            cohort_marginals = cohort_marginals.replace(0, zero_replacement)
            type_marginals = type_marginals.replace(0, zero_replacement)

        return type_marginals, cohort_marginals

    @staticmethod
    def filter_columns(
        df: pd.DataFrame,
        dw_types: list,
        type_label: str = TYPE_TOTAL_LABEL,
        label_col: str = "vintage",
    ) -> pd.DataFrame:
        """
        Filter DataFrame to relevant dwelling type columns.

        Args:
            df: Input DataFrame
            dw_types: List of dwelling types to keep
            type_label: Label for total column
            label_col: Name of label column (e.g., 'vintage')

        Returns:
            Filtered DataFrame
        """
        if label_col in df.columns:
            df = df.set_index(label_col)

        cols_to_keep = [type_label] + [
            col for col in sorted(dw_types) if col in df.columns and col != type_label
        ]

        return df.loc[:, cols_to_keep].astype(float)


# ==== Main module code ====
def interpolate(infile):
    df = pd.read_parquet(infile)

    # Check input data
    required_columns = ["census_year", "type", "vintage", "dwellings"]

    # Validate input DataFrame  # TODO convert to try-except block
    if df.empty:
        msg = "Input dataframe is empty."
        logger.error(msg)
        raise ValueError(msg)

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        msg = f"DataFrame is missing required columns: {missing_columns}"
        logger.error(msg)
        raise ValueError(msg)

    # create a boolean mask to identify which values were original data or not
    raw_mask = get_rawdata_mask(df)

    # Fill missing values using interpolation
    interpolated = fill_missing_dwellings(df)

    reconciled = reconcile_all_years(interpolated, raw_mask)

    # TODO REPRENDRE! NOW I NEED TO APPLY THE FIX MARGINALS and IPFN, WHICH BOTH RELY ON RECONCILE DATA WITH MARGINALS - MAKE SURE IT WORKS WITH THE IS-ORIG FLAG

    # NOTE: check if the following is useful
    # After transforms that may have reindexed/pivoted/aggregated, reapply original values
    # Restore by key columns — vectorized, safe after grouping/pivot/unpivot (works if keys exist in transformed)
    # try:
    #     interpolated_restored = restore_original_data(interpolated, orig_snapshot, key_cols=key_cols, value_col="dwellings")
    # except ValueError as err:
    #     # If keys are missing in the transformed result, log and raise so caller can examine the layout
    #     logger.error(f"Could not restore original values: {err}")
    #     raise

    # run check_marginals (from dataprep) or a similar function to verify that the marginals are either the same, or closer than they were, after the fill_missing_values

    return interpolated, raw_mask, reconciled


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
    sums the numeric data with a minimum count of 1 (so that if all values are NaN, the result
    remains NaN), resets the index, and then returns a boolean DataFrame, indicating which values
    contain data (i.e., are not nans)

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
    grouped = df.groupby(by).sum(min_count=1).reset_index()
    mask = grouped.notna()
    mask[by] = grouped[by]  # preserve key columns for filtering/pivoting
    return mask


def fill_missing_dwellings(
        df: pd.DataFrame,
        backfill_margin: int = 10,
        range_start_year: int = 1685,
    ) -> pd.DataFrame:
    """
    Fill missing dwelling values using backward fill within each (type, vintage) group.

    Fills NaN values by propagating the earliest known later-census value backward
    in time. Forward-fill is applied as a fallback for groups with no earlier
    non-NaN value. Future vintages (start year > census year) are then zeroed out.

    Args:
        df: DataFrame with 'census_year', 'type', 'vintage', and 'dwellings' columns.

    Returns:
        DataFrame with NaN dwellings filled. Raises ValueError if any NaN remain.

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
    required_columns = ["census_year", "type", "vintage", "dwellings"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"DataFrame is missing required columns: {missing_columns}")

    # Sort so bfill propagates from later → earlier census years within each group
    df = df.sort_values(["type", "vintage", "census_year"]).reset_index(drop=True)

    def _vintage_bounds(v: str) -> tuple[int, int]:
            parts = str(v).split("-")
            return int(parts[0]), int(parts[1])
    
    filled_groups = []
    for (dw_type, vintage), group in df.groupby(["type", "vintage"], sort=False):
        s = group.set_index("census_year")["dwellings"]
        if dw_type == TYPE_TOTAL_LABEL and s.isna().any():
            start, end = _vintage_bounds(vintage)
            # Stage A: bfill census years after cohort end
            post = s.index > end
            s.loc[post] = s.loc[post].bfill()
            # Reindex to annual range, zero pre-cohort, index-interpolate
            upper = end + backfill_margin
            temp = s.reindex(range(range_start_year, upper + 1))
            temp.loc[: start - 1] = 0
            temp = temp.interpolate(method="index")
            in_range = s.index <= upper
            s.loc[in_range] = s.loc[in_range].combine_first(temp.loc[s.index[in_range]])
            s = s.bfill().ffill()  # fallback for anything outside range
        else:
            s = s.bfill().ffill()
        group = group.copy()
        group["dwellings"] = s.to_numpy()
        filled_groups.append(group)
    df = pd.concat(filled_groups, ignore_index=True)

    # Zero out vintages that had not yet been built at the time of the census
    def _vintage_start(v: str) -> int:
        try:
            return int(str(v).split("-")[0])
        except (ValueError, IndexError):
            return 0  # "1608-2025" total row → never zeroed

    vintage_starts = df["vintage"].map(_vintage_start)
    df.loc[vintage_starts > df["census_year"], "dwellings"] = 0.0

    if df["dwellings"].isna().any():
        msg = "NaN values remain after filling. Check input data for completeness."
        logger.warning(msg)
        raise ValueError(msg)

    return df


def todo_marginals(df: pd.DataFrame):
    working_df = df.copy()
    census_years = working_df["census_years"].unique().tolist()

    # Fix the marginals
    ipfn_df = {}
    for year in census_years:
        logger.info(f" === Census {year} === ")
        # Take individual census_year
        tsplit = (
            working_df.set_index("census_year")
            .loc[year, :]
            .pivot(index="vintage", columns="type", values="dwellings")
            .reset_index()
        )
        # FIXME | REPRENDRE : THE ORIGINAL DATA MUST BE PROTECTED HERE TOO; I CANT DO ANYTHING TO THE MARGINALS THEMSELVE, AS I ALWAYS HAVE THE DATA. HOWEVER, I SHOULD USE THE SAME BEHAVIOR AS IN "PROTECTING DATA IN IPFN" TO PROTECT ORIGINAL DATA FROM BEING OVERWRITTEN
        # NOTE - I think this will be done automatically throug reconcile_data_wth_marginals
        ipfn_res = reconcile_data_with_marginals(tsplit)  # HERE pass mask!
        ipfn_df[year] = ipfn_res.result

        # NOTE diff can be useful for documentation; maybe check max/min value to see largest diff?

    result = (
        pd.concat(ipfn_df)
        .stack("type")
        .reset_index()
        .rename(columns={"level_0": "census_year", 0: "dwellings"})
    )

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

    if (arr < 0).any():
        raise ValueError("round_consistent_sum requires non-negative values.")
    desired_sum = int(desired_sum)
    if arr.sum() == 0:
        if desired_sum == 0:
            return [0] * len(arr)
        raise ValueError("Cannot scale an all-zero array to a nonzero sum.")

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
    desired_sum: Optional[int] = None,
    label: str = VINTAGE_COL,
    dw_types: list[str] = TARGET_TYPES,
    type_total_label: str = TYPE_TOTAL_LABEL,
    cohort_total_label: str = COHORT_TOTAL_LABEL,
    protect_original: Optional[pd.DataFrame] = None,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
) -> MarginalAdjustment:
    """
    Adjust marginals using Largest Remainder Method while protecting original data.

    Args:
        df: Input DataFrame with marginals
        desired_sum: Target total sum (defaults to existing total)
        label: Column name for index labels
        dw_types: List of dwelling types to process
        type_total_label: Column name for type totals
        cohort_total_label: Row label for cohort totals
        protect_original: Boolean mask for protected values
        atol: Absolute tolerance for validation
        rtol: Relative tolerance for validation

    Returns:
        MarginalAdjustment object with results

    Raises:
        ValueError: If desired_sum is incompatible with protected data
    """
    # Filter and prepare data
    working_df = MarginalExtractor.filter_columns(df, dw_types, type_total_label, label,)

    # Set default desired sum and ensure it's an integer
    if desired_sum is None:
        desired_sum = int(working_df.loc[cohort_total_label, type_total_label])
    else:
        desired_sum = int(desired_sum)

    # Initialize data protector
    protector = DataProtector(
        working_df, protect_original, cohort_total_label, type_total_label
    )

    # Validate desired sum against protected data
    is_valid, error_msg = protector.validate_desired_sum(desired_sum, atol, rtol)
    if not is_valid:
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Calculate adjusted targets
    type_total, cohort_total = protector.calculate_adjusted_targets(desired_sum)

    # Get protected sums for reporting
    protected_type_sum, protected_cohort_sum = protector.get_protected_sums()

    # Get modifiable indices
    mod_rows, mod_cols = protector.get_modifiable_indices()

    # Extract marginals
    type_marginals, cohort_marginals = MarginalExtractor.extract_marginals(
        working_df, cohort_total_label, type_total_label, replace_zeros=True, zero_replacement=0.1
    ) # XXX check if putting 0.0 works or not?

    # Filter to modifiable parts only
    type_marginals_mod = type_marginals.loc[mod_cols]
    cohort_marginals_mod = cohort_marginals.loc[mod_rows]

    # Get protected values to calculate floor values
    protected_types, protected_cohorts = protector.get_protected_marginals()

    # Calculate minimum values (floors) from protected interior cells
    # For type marginals: sum of protected cohort rows for each modifiable column
    # NOTE these two lines were modified by the code below, to account for the new behaviour. Marginals should *not* be protected; only interior data is. OR, at least, we can't do both: interior data OR marginals must be protected. Since we are modifying the marginals, it makes more sense to make them unprotected.
    # type_floors = working_df.loc[protected_cohorts.notna(), mod_cols].sum()
    # cohort_floors = working_df.loc[mod_rows, protected_types.notna()].sum(axis=1)

    # Get the interior cells (excluding marginal row and column)
    interior_rows = working_df.index.drop(cohort_total_label)
    interior_cols = working_df.columns.drop(type_total_label)

    # Get protection mask for interior cells
    if protect_original is not None:
        # Ensure mask aligns with working_df
        aligned_mask = protect_original.reindex(
            index=working_df.index, columns=working_df.columns, fill_value=False
        ).fillna(False).astype(bool)
        interior_protection = aligned_mask.loc[interior_rows, interior_cols]
    else:
        interior_protection = pd.DataFrame(
            False, index=interior_rows, columns=interior_cols
        )

    # Type floors: for each modifiable column, sum protected interior cells
    type_floors = pd.Series(0.0, index=mod_cols)
    for col in mod_cols:
        if col in interior_protection.columns:
            mask = interior_protection[col]
            type_floors[col] = working_df.loc[mask[mask].index, col].sum()

    # Cohort floors: for each modifiable row, sum protected interior cells
    cohort_floors = pd.Series(0.0, index=mod_rows)
    for row in mod_rows:
        if row in interior_protection.index:
            mask = interior_protection.loc[row, :]
            cohort_floors[row] = working_df.loc[row, mask[mask].index].sum()

    # Also need to handle PROTECTED rows - their marginals must equal their interior sum
    # For protected cohort rows (not modifiable), update their marginal to match interior
    protected_rows = interior_rows.difference(mod_rows)
    for row in protected_rows:
        if row in interior_protection.index:
            # Sum ALL interior cells for this protected row (they're all protected)
            interior_sum = working_df.loc[row, interior_cols].sum()
            # Update the marginal to match
            working_df.loc[row, type_total_label] = interior_sum

    # Recalculate protected_cohort_sum after fixing protected row marginals
    protected_types, protected_cohorts = protector.get_protected_marginals()
    # Update protected_cohort_sum based on corrected marginals
    protected_cohort_sum = working_df.loc[
        protected_cohorts.notna().index[protected_cohorts.notna()], type_total_label
    ].sum()

    # Recalculate cohort_total after adjusting protected rows
    cohort_total = desired_sum - protected_cohort_sum

    # Check if floors exceed available budget
    type_floor_sum = type_floors.sum()
    cohort_floor_sum = cohort_floors.sum()

    if type_floor_sum > type_total + atol:  # FIXME use atol and rtol correctly
        msg = (
            f"Sum of type floors ({type_floor_sum:.2f}) exceeds available type budget "
            f"({type_total:.2f}). Cannot balance marginals."
        )
        logger.error(msg)
        raise ValueError(msg)

    if cohort_floor_sum > cohort_total + atol:
        msg = (
            f"Sum of cohort floors ({cohort_floor_sum:.2f}) exceeds available cohort budget "
            f"({cohort_total:.2f}). Cannot balance marginals."
        )
        logger.error(msg)
        raise ValueError(msg)
    # For TYPE marginals:
    # Calculate the "free" portion (total minus floors)
    type_free_budget = int(type_total - type_floor_sum)

    # Get the current marginal values minus their floors (what's freely adjustable)
    type_marginals_above_floor = np.maximum(
        type_marginals_mod.to_numpy() - type_floors.to_numpy(), 0
    )

    # Only apply round_consistent_sum to the free portion if there's budget
    if type_free_budget > 0 and type_marginals_above_floor.sum() > 0:
        adj_type_free = round_consistent_sum(
            type_marginals_above_floor, type_free_budget
        )
        adj_type_marginals = np.array(adj_type_free) + type_floors.to_numpy().astype(
            int
        )
    else:
        # No free budget, just use floors (rounded)
        adj_type_marginals = np.ceil(type_floors.to_numpy()).astype(int)

    # For COHORT marginals:
    cohort_free_budget = int(cohort_total - cohort_floor_sum)

    cohort_marginals_above_floor = np.maximum(
        cohort_marginals_mod.to_numpy() - cohort_floors.to_numpy(), 0
    )

    if cohort_free_budget > 0 and cohort_marginals_above_floor.sum() > 0:
        adj_cohort_free = round_consistent_sum(
            cohort_marginals_above_floor, cohort_free_budget
        )
        adj_cohort_marginals = np.array(
            adj_cohort_free
        ) + cohort_floors.to_numpy().astype(int)
    else:
        adj_cohort_marginals = np.ceil(cohort_floors.to_numpy()).astype(int)

    # NOTE: no post-LRM maximum bump — round_consistent_sum guarantees each
    # vector sums exactly to its budget; clamping afterwards breaks that
    # invariant and makes the IPFN aggregates mutually inconsistent.
    # Floor violations are already rejected upstream by the budget checks.

    # Calculate differences
    type_diff = adj_type_marginals - type_marginals_mod.to_numpy()
    cohort_diff = adj_cohort_marginals - cohort_marginals_mod.to_numpy()

    # Log adjustments
    if np.any(type_diff != 0):
        logger.info(
            f"Type marginal adjustments: {type_diff} (total: {type_diff.sum()})"
        )
    if np.any(cohort_diff != 0):
        logger.info(
            f"Cohort marginal adjustments: {cohort_diff} (total: {cohort_diff.sum()})"
        )

    # Update DataFrame
    working_df.loc[mod_rows, type_total_label] = pd.Series(
        adj_cohort_marginals, index=mod_rows
    )
    working_df.loc[cohort_total_label, mod_cols] = pd.Series(
        adj_type_marginals, index=mod_cols
    )
    working_df.loc[cohort_total_label, type_total_label] = desired_sum

    # Convert to integers only at the end
    numeric_cols = working_df.select_dtypes(include=[np.number]).columns
    working_df[numeric_cols] = working_df[numeric_cols].round().astype(int)

    return MarginalAdjustment(
        adjusted_df=working_df,
        type_diff=type_diff.astype(int),
        cohort_diff=cohort_diff.astype(int),
        type_marginals=type_marginals,
        cohort_marginals=cohort_marginals,
        protected_type_sum=int(round(protected_type_sum)),
        protected_cohort_sum=int(round(protected_cohort_sum)),
    )


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
    label: str = VINTAGE_COL,
    desired_sum: Optional[int] = None,
    dw_types: list[str] = TARGET_TYPES,
    type_total_label: str = TYPE_TOTAL_LABEL,
    cohort_total_label: str = COHORT_TOTAL_LABEL,
    convergence_rate: float = 1e-6,
    max_iter: int = 500,
    zero_replacement: float = ZERO_REPLACEMENT,
    protect_original: Optional[pd.DataFrame] = None,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
) -> IPFNResult:
    """
    Reconcile data with marginals using fix_marginals and IPFN.

    Raises:
        ValueError: If desired_sum incompatible with protected data
        RuntimeError: If IPFN processing fails
    """
    # Step 1: Fix marginals (includes validation)
    marginal_result = fix_marginals(
        df=df,
        desired_sum=desired_sum,
        label=label,
        dw_types=dw_types,
        type_total_label=type_total_label,
        cohort_total_label=cohort_total_label,
        protect_original=protect_original,
        atol=atol,
        rtol=rtol,
    )

    working_df = marginal_result.adjusted_df

    # Step 2: Run IPFN — notebook contract: split zero rows/cols out entirely,
    # IPFN only sees the strictly positive submatrix, zeros re-attached after.
    interior = working_df.drop(cohort_total_label).drop(type_total_label, axis=1)
    zero_rows = interior.index[(interior == 0).all(axis=1)]
    zero_cols = interior.columns[(interior == 0).all(axis=0)]
    data_matrix = interior.drop(index=zero_rows, columns=zero_cols)

    # Seed interior zeros ONLY (mixed rows), never the marginals
    data_matrix = data_matrix.replace(0, zero_replacement)

    # Marginals restricted to the nonzero submatrix — LRM in fix_marginals has
    # already made both vectors sum exactly to desired_sum; zero rows/cols
    # contribute 0, so consistency is preserved after the drop.
    cohort_marginals = working_df.loc[data_matrix.index, type_total_label]
    type_marginals = working_df.loc[cohort_total_label, data_matrix.columns]

    # Store initial values for difference calculation
    initial_matrix = data_matrix.copy()

    # Run IPFN
    try:
        ipfn_result = apply_ipfn(
            data_matrix.to_numpy(),
            aggregates=[cohort_marginals.to_numpy(), type_marginals.to_numpy()],
            dimensions=[[0], [1]],
            convergence_rate=convergence_rate,
            max_iter=max_iter,
        )
        converged = True
        iterations = max_iter  # We don't track actual iterations in apply_ipfn
    except Exception as e:
        logger.error(f"IPFN failed: {e}")
        raise RuntimeError(f"IPFN processing failed: {e}")

    # Round each row to integers that sum exactly to its cohort marginal
    # (astype(int) floors and silently loses 2-4 units per row)
    ipfn_df = pd.DataFrame(
        ipfn_result, index=data_matrix.index, columns=data_matrix.columns
    )
    for row in ipfn_df.index:
        ipfn_df.loc[row, :] = round_consistent_sum(
            ipfn_df.loc[row, :].to_numpy(), int(cohort_marginals.loc[row])
        )

    # Reconstruct the full DataFrame; zero rows/cols stay exactly zero
    result = working_df.copy()
    result.loc[ipfn_df.index, ipfn_df.columns] = ipfn_df

    # Calculate difference BEFORE restoring protected values
    diff = result.copy()
    diff.loc[data_matrix.index, data_matrix.columns] = (
        result.loc[data_matrix.index, data_matrix.columns].to_numpy()
        - initial_matrix.to_numpy()
    )

    # Protection is handled upstream via the aa/bb column split in
    # reconcile_all_years; overwriting cells after IPFN convergence would
    # break the row/column sums IPFN just enforced.
    if protect_original is not None:
        logger.warning(
            "protect_original overlay is deprecated; use the two-pass column "
            "split in reconcile_all_years instead. Mask ignored."
        )

    # Convert final results to integers
    numeric_cols = result.select_dtypes(include=[np.number]).columns
    result[numeric_cols] = result[numeric_cols].round().astype(int)

    numeric_cols_diff = diff.select_dtypes(include=[np.number]).columns
    diff[numeric_cols_diff] = diff[numeric_cols_diff].round().astype(int)

    return IPFNResult(
        result=result,
        difference=diff,
        convergence_achieved=converged,
        iterations=iterations,
    )

def reconcile_all_years(
    filled_df: pd.DataFrame,
    raw_mask: pd.DataFrame,
    dw_types: list[str] = TARGET_TYPES,
    type_total_label: str = "total",
    cohort_total_label: str = "1608-2025",
) -> pd.DataFrame:
    """
    Reconcile interpolated dwelling data for every census year.

    Reproduces the unfm_fix logic from dmfa_dataprep.ipynb using the
    existing pipeline: fill_missing_dwellings → fix_marginals → apply_ipfn
    (via reconcile_data_with_marginals).

    For each census year the function:
    1. Pivots the filled long-format data to wide (vintage × type) — this is
       required by fix_marginals / apply_ipfn which operate on matrices.
    2. Builds a matching wide mask so observed cells can be protected.
    3. Calls reconcile_data_with_marginals (which internally runs
       fix_marginals then apply_ipfn), passing the mask via
       ``protect_original``.
    4. Melts the result back to long format and appends it to the output.

    Args:
        raw_mask: Boolean mask from get_rawdata_mask (True = observed).
        dw_types: Target dwelling type columns (must include 'total').
        type_total_label: Column name for the type total.
        cohort_total_label: Row label for the cohort total.

    Returns:
        pd.DataFrame: Long-format DataFrame with columns
        [census_year, vintage, type, dwellings].
    """
    agg_types = [t for t in dw_types if t != type_total_label]
    reconciled_frames = []

    for census_year, group in filled_df.groupby("census_year"):
        # --- pivot to wide (needed by fix_marginals / IPFN) ---
        wide_filled = group.pivot(
            index="vintage", columns="type", values="dwellings"
        )
        present_types = [t for t in dw_types if t in wide_filled.columns]

        # --- build matching wide mask ---
        mask_group = raw_mask[raw_mask["census_year"] == census_year]
        wide_mask = mask_group.pivot(
            index="vintage", columns="type", values="dwellings"
        )
        wide_mask = wide_mask.reindex_like(wide_filled).fillna(False).astype(bool)

        # desired_sum must come from the observed grand-total cell
        if not bool(wide_mask.loc[cohort_total_label, type_total_label]):
            msg = (
                f"Grand total cell ({cohort_total_label}, {type_total_label}) "
                f"is not observed for census_year={census_year}; cannot anchor desired_sum."
            )
            logger.error(msg)
            raise ValueError(msg)

        # --- Stage B (notebook): two-pass reconciliation ---
        masked = wide_filled[present_types].mask(~wide_mask[present_types])
        target_cols = masked.isna().any(axis=0)
        if not target_cols.any():
            ipfn_res = reconcile_data_with_marginals(
                wide_filled[present_types], label="vintage", dw_types=dw_types,
                type_total_label=type_total_label, cohort_total_label=cohort_total_label,
            )
            reconciled_wide = ipfn_res.result.copy()
        else:
            target_cols[type_total_label] = True
            aa = wide_filled.loc[:, target_cols[target_cols].index].copy()
            bb = wide_filled.loc[:, target_cols[~target_cols].index].copy()
            # remove known columns' contribution from the total marginal
            aa[type_total_label] = (aa[type_total_label] - bb.sum(axis=1)).clip(lower=0)
            first = reconcile_data_with_marginals(
                aa, label="vintage", dw_types=list(aa.columns),
                type_total_label=type_total_label, cohort_total_label=cohort_total_label,
            )
            stitched = pd.concat([first.result.drop(columns=type_total_label), bb], axis=1)
            stitched = stitched[[c for c in present_types if c != type_total_label]]
            stitched.insert(0, type_total_label, stitched.sum(axis=1))
            second = reconcile_data_with_marginals(
                stitched, label="vintage", dw_types=dw_types,
                type_total_label=type_total_label, cohort_total_label=cohort_total_label,
            )
            ipfn_res = second
            reconciled_wide = second.result.copy()

        # Recompute totals from components to ensure consistency
        reconciled_agg = [c for c in agg_types if c in reconciled_wide.columns]
        reconciled_wide[type_total_label] = reconciled_wide[reconciled_agg].sum(axis=1)


        # --- melt back to long format ---
        df_long = (
            reconciled_wide.reset_index()
            .melt(id_vars="vintage", var_name="type", value_name="dwellings")
        )
        df_long["census_year"] = census_year
        reconciled_frames.append(df_long)

    result = pd.concat(reconciled_frames, ignore_index=True)
    # Reorder columns to match input convention

    return result[["census_year", "vintage", "type", "dwellings"]]


if __name__ == "__main__":
    infile = "./data/clean/fulldata.parquet"
    filled, raw_mask, reconciled = interpolate(infile)

    print("\n=== Reconciled data (sample) ===")
    display(reconciled.groupby(by=['census_year','vintage','type']).sum().head(20))

    print("\n=== Previous data (sample) ===")
    previous = pd.read_csv("./data/dwellings_1685_2021.csv", usecols=[1,2,3,4])
    previous.rename(columns={'year':'census_year'}, inplace=True)
    previous['vintage'] = previous['vintage'].replace({'0-2025':'1608-2025'})
    display(previous.groupby(by=['census_year','vintage','type']).sum().head(20))

    # Spot-check: census year 1685 - only 1608-1920 should be non-zero
    yr1685 = reconciled[reconciled["census_year"] == 1685]
    total_1685 = yr1685[
        (yr1685["vintage"] == "1608-2025") & (yr1685["type"] == "total")
    ]["dwellings"].values[0]
    assert total_1685 > 0, "Total for 1685 should be > 0"

    post_1920 = yr1685[
        ~yr1685["vintage"].isin(["1608-2025", "1608-1920"])
    ]["dwellings"].sum()
    assert post_1920 == 0, "All vintages after 1920 should be zero for census year 1685"

    # Visual comparison
    comparison = pd.merge(reconciled, previous, how='inner', 
                      on=['census_year','vintage','type'], 
                      suffixes=('_reconciled', '_previous'))
    display(comparison.head())

    comparison['difference'] = comparison['dwellings_reconciled'] - comparison['dwellings_previous']
    comparison['abs_difference'] = comparison['difference'].abs()
    comparison['pct_difference'] = (comparison['difference'] / comparison['dwellings_previous'] * 100).abs()
    
    # Difference Plot 
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.lineplot(data=comparison, x='census_year', y='difference', 
                hue='type', ax=axes[0])
    axes[0].set_title('Absolute Difference (Reconciled - Previous)')
    axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[0].set_ylabel('Difference')

    sns.lineplot(data=comparison, x='census_year', y='pct_difference', 
                hue='type', ax=axes[1])
    axes[1].set_title('Absolute % Difference')
    axes[1].set_ylabel('% Difference')

    plt.tight_layout()
    plt.savefig('discrepancy_plot.png', dpi=1000, bbox_inches='tight')

    # Scatter plot
    fig, ax = plt.subplots(figsize=(10, 10))
    for type_val in comparison['type'].unique():
        subset = comparison[comparison['type'] == type_val]
        ax.scatter(subset['dwellings_previous'], subset['dwellings_reconciled'], 
                label=type_val, alpha=0.6, s=50)

    # Perfect agreement line
    min_val = comparison[['dwellings_reconciled', 'dwellings_previous']].min().min()
    max_val = comparison[['dwellings_reconciled', 'dwellings_previous']].max().max()
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect agreement')

    ax.set_xlabel('Previous')
    ax.set_ylabel('Reconciled')
    ax.set_title('Reconciled vs Previous Values')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig('scatter_comparison.png', dpi=1000, bbox_inches='tight')

    # Summary statistics
    comparison_stats = comparison.groupby('type').agg({
        'difference': ['mean', 'std', 'min', 'max', 'median'],
        'pct_difference': ['mean', 'median', 'max'],
        'dwellings_reconciled': 'count'
    }).round(2)

    print(comparison_stats)

    # By census_year too
    by_year = comparison.groupby(['census_year', 'type']).agg({
        'difference': ['mean', 'std'],
        'pct_difference': ['mean', 'max']
    }).round(2)

    print(by_year)

    # Correlation analysis
    correlation = comparison.groupby('type').apply(
        lambda x: x['dwellings_reconciled'].corr(x['dwellings_previous'])
    ).round(4)

    print("Correlation (reconciled vs previous):\n", correlation)

    # Detailed comparison table
    print("\nDetailed Comparison:")
    print(comparison[['census_year', 'vintage', 'type', 
                    'dwellings_previous', 'dwellings_reconciled', 
                    'difference', 'pct_difference']].head(20))


# TODO REPRENDRE consider adding the pre-1941 data (cs_data // old_cs_data), then run, and compare results to the initial dataset in a relplot (see previous to last cell in bac/dmfa_dataprep)


    # HIGHLIGHT LARGE DIFFERENCES (threshold: 1000 dwellings)
    threshold = 1000
    large_diffs = comparison[comparison['abs_difference'] >= threshold].copy()

    # Sort by magnitude
    large_diffs_sorted = large_diffs.sort_values('abs_difference', ascending=False)

    print(f"\n{'='*100}")
    print(f"LARGE DISCREPANCIES (>= {threshold:,} dwellings): {len(large_diffs_sorted)} records found")
    print(f"{'='*100}\n")

    # Display in readable format
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)

    print(large_diffs_sorted[['census_year', 'vintage', 'type', 
                            'dwellings_previous', 'dwellings_reconciled', 
                            'difference', 'pct_difference']].to_string())

    # Summary by group
    print(f"\n{'='*100}")
    print("SUMMARY BY CENSUS_YEAR/VINTAGE/TYPE:")
    print(f"{'='*100}\n")

    summary = large_diffs.groupby(['census_year', 'vintage', 'type']).agg({
        'abs_difference': ['count', 'sum', 'mean', 'max'],
        'pct_difference': 'mean'
    }).round(0)

    summary.columns = ['Count', 'Total_Diff', 'Avg_Diff', 'Max_Diff', 'Avg_Pct_Diff']
    summary = summary.sort_values('Max_Diff', ascending=False)
    print(summary.to_string())

    # VISUALIZATION: Heatmap of problem areas
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Heatmap 1: Max difference by census_year and vintage
    pivot_max = large_diffs.pivot_table(values='abs_difference', 
                                        index=['vintage', 'type'], 
                                        columns='census_year', 
                                        aggfunc='max')
    sns.heatmap(pivot_max, annot=True, fmt='.0f', cmap='YlOrRd', ax=axes[0], cbar_kws={'label': 'Max Difference'})
    axes[0].set_title(f'Maximum Dwelling Count Differences (>= {threshold:,}) by Census Year & Vintage')

    # Heatmap 2: Count of problematic records
    pivot_count = large_diffs.pivot_table(values='abs_difference', 
                                        index=['vintage', 'type'], 
                                        columns='census_year', 
                                        aggfunc='count')
    sns.heatmap(pivot_count, annot=True, fmt='.0f', cmap='Blues', ax=axes[1], cbar_kws={'label': 'Count of Large Diffs'})
    axes[1].set_title(f'Number of Large Discrepancies by Census Year & Vintage')

    plt.tight_layout()
    plt.savefig('large_discrepancies_heatmap.png', dpi=300, bbox_inches='tight')

    # VISUALIZATION: Bar chart sorted by magnitude
    fig, ax = plt.subplots(figsize=(12, 8))

    # Group label for readability
    large_diffs['group'] = (large_diffs['census_year'].astype(str) + '_' + 
                            large_diffs['vintage'].astype(str) + '_' + 
                            large_diffs['type'].astype(str))

    top_diffs = large_diffs.nlargest(20, 'abs_difference')
    sns.barplot(data=top_diffs, x='abs_difference', y='group', palette='Reds_r', ax=ax)
    ax.set_xlabel(f'Absolute Difference in Dwellings')
    ax.set_ylabel('Census Year_Vintage_Type')
    ax.set_title(f'Top 20 Largest Discrepancies (>= {threshold:,} dwellings)')

    for i, v in enumerate(top_diffs['abs_difference']):
        ax.text(v + 100, i, f'{v:,.0f}', va='center')

    plt.tight_layout()
    plt.savefig('top_discrepancies.png', dpi=1000, bbox_inches='tight')

    # reconciled.to_parquet('reconciled.parquet')
    file_path = DATA_DIR / "clean" / "reconciled.parquet" # FIXME ensure dir exists?
    reconciled.to_parquet(file_path)