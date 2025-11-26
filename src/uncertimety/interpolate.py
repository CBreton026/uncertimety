import math
import pandas as pd
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
        zero_replacement: float = ZERO_REPLACEMENT,
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

    # Create a snapshot of original data for restoration; retrieve mask for later transforms
    key_cols = ["census_year", "vintage", "type"]
    snapshot, dataset = mark_original_data(df, key_cols=key_cols, value_col="dwellings")

    # Fill missing values using interpolation
    interpolated = fill_missing_dwellings(dataset)

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

    return interpolated  # , raw_mask


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


def mark_original_data(
    df: pd.DataFrame, value_col: str = "dwellings"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Mark original (non-NaN) data points to preserve them during interpolation.

    Args:
        df: DataFrame with data to mark
        value_col: Column containing values to check for NaN

    Returns:
        Tuple of (snapshot_df, marked_df) where:
            - snapshot_df: DataFrame showing which values were originally present
            - marked_df: Original DataFrame with additional 'is_orig' column
    """
    # Create the mask based on non-NaN values in the value column
    is_original = df[value_col].notna()

    # Create snapshot showing original data availability
    snapshot = df.copy()
    snapshot["is_orig"] = is_original

    # Add is_orig column to the main dataframe
    df_marked = df.copy()
    df_marked["is_orig"] = is_original

    return snapshot, df_marked


def restore_original_data(
    transformed: pd.DataFrame,
    orig_snapshot: pd.DataFrame,
    key_cols=None,
    value_col: str = "dwellings",
):
    """
    Reapply original values from orig_snapshot into transformed by key columns.
    Uses a left merge + combine_first so it works even after group/pivot/indices changes.
    """
    if key_cols is None:
        key_cols = ["census_year", "vintage", "type"]
    # Ensure keys exist in both frames
    missing = [
        k
        for k in key_cols
        if k not in transformed.columns or k not in orig_snapshot.columns
    ]
    if missing:
        raise ValueError(f"Missing key columns for restore: {missing}")

    # FIXME instead of restoring, maybe only check/validate if orig data was preserved
    merged = transformed.merge(
        orig_snapshot, on=key_cols, how="left", suffixes=("", "_orig")
    )
    merged[value_col] = merged[f"{value_col}_orig"].combine_first(merged[value_col])
    merged.drop(
        columns=[c for c in merged.columns if c.endswith("_orig")], inplace=True
    )
    return merged


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
        new_data, diff = reconcile_data_with_marginals(tsplit)  # HERE pass mask!

        ipfn_df[year] = new_data
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
    working_df = MarginalExtractor.filter_columns(df, dw_types, type_total_label, label)

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
        working_df, cohort_total_label, type_total_label, replace_zeros=True
    )

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
            index=working_df.index, 
            columns=working_df.columns, 
            fill_value=False
        ).fillna(False)
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
    protected_cohort_sum = working_df.loc[protected_cohorts.notna().index[protected_cohorts.notna()], type_total_label].sum()

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
        adj_type_free = round_consistent_sum(type_marginals_above_floor, type_free_budget)
        adj_type_marginals = np.array(adj_type_free) + type_floors.to_numpy().astype(int)
    else:
        # No free budget, just use floors (rounded)
        adj_type_marginals = np.ceil(type_floors.to_numpy()).astype(int)

    # For COHORT marginals:
    cohort_free_budget = int(cohort_total - cohort_floor_sum)
    
    cohort_marginals_above_floor = np.maximum(
        cohort_marginals_mod.to_numpy() - cohort_floors.to_numpy(), 0
    )
    
    if cohort_free_budget > 0 and cohort_marginals_above_floor.sum() > 0:
        adj_cohort_free = round_consistent_sum(cohort_marginals_above_floor, cohort_free_budget)
        adj_cohort_marginals = np.array(adj_cohort_free) + cohort_floors.to_numpy().astype(int)
    else:
        adj_cohort_marginals = np.ceil(cohort_floors.to_numpy()).astype(int)

    # Ensure floors are respected (safety check)
    adj_type_marginals = np.maximum(adj_type_marginals, np.ceil(type_floors.to_numpy()).astype(int))
    adj_cohort_marginals = np.maximum(adj_cohort_marginals, np.ceil(cohort_floors.to_numpy()).astype(int))

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

    # # Replace remaining zeros for IPFN
    # if protect_original is not None:
    #     working_df = working_df.replace(0, ZERO_REPLACEMENT)

    # Replace zeros for IPFN to avoid division by zero
    # Store which values were originally zero so we can restore them later if needed
    zero_mask = working_df == 0 # FIXME unused for now?
    working_df = working_df.replace(0, zero_replacement)

    # Step 2: Run IPFN
    # Extract the data matrix (excluding marginals)
    data_matrix = working_df.drop(cohort_total_label).drop(type_total_label, axis=1)

    # Extract target marginals
    cohort_marginals = working_df.drop(cohort_total_label).loc[:, type_total_label]
    type_marginals = working_df.drop(type_total_label, axis=1).loc[
        cohort_total_label, : 
    ]

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

    # Reconstruct the full DataFrame
    result = working_df.copy()
    result.loc[data_matrix.index, data_matrix.columns] = ipfn_result.astype(int)

    # # Calculate difference # TODO delete, DEPRECATED 
    # diff = result.copy().astype(float)
    # diff.loc[data_matrix.index, data_matrix.columns] = (
    #     ipfn_result - initial_matrix.to_numpy()
    # )

    # # If we have protection, restore protected values and zero out their diffs
    # if protect_original is not None:
    #     # Restore original protected values
    #     try: 
    #         result[protect_original] = df.set_index(label)[protect_original]
    #         diff[protect_original] = 0
    #     except KeyError as err:
    #         logger.warning(f"Error: {err}. Attempting without set_index")  # This is because "vintage" might already be in df.index
    #         result[protect_original] = df[protect_original]
    #         diff[protect_original] = 0

    # return IPFNResult(
    #     result=result,
    #     difference=diff,
    #     convergence_achieved=converged,
    #     iterations=iterations,
    # )

    # Calculate difference BEFORE restoring protected values
    diff = result.copy()
    diff.loc[data_matrix.index, data_matrix.columns] = (
        result.loc[data_matrix.index, data_matrix.columns].to_numpy() 
        - initial_matrix.to_numpy()
    )

    # If we have protection, restore protected values and zero out their diffs
    if protect_original is not None:
        # Align the mask with result DataFrame (same index/columns)
        aligned_mask = protect_original.reindex(
            index=result.index,
            columns=result.columns,
            fill_value=False
        ).fillna(False)
        
        # Get original values aligned with result
        try:
            original_aligned = df.set_index(label).reindex(
                index=result.index,
                columns=result.columns
            )
        except KeyError as err:
            logger.warning(f"Error: {err}. Attempting without set_index")  # This is because "vintage" might already be in df.index
            original_aligned = df.reindex(
                index=result.index,
                columns=result.columns
            )
        
        # Restore protected values
        result = result.where(~aligned_mask, original_aligned)
        
        # Zero out differences for protected values
        diff = diff.where(~aligned_mask, 0)

    # Restore zeros that should remain zero (where original was zero and not protected)
    if protect_original is not None:
        aligned_mask = protect_original.reindex(
            index=result.index,
            columns=result.columns,
            fill_value=False
        ).fillna(False)
        # Where originally zero AND not protected, set back to zero
        should_be_zero = zero_mask & ~aligned_mask
        result = result.where(~should_be_zero, 0)
    else:
        # No protection, restore all original zeros
        result = result.where(~zero_mask, 0)

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

if __name__ == "__main__":
    infile = "./data/clean/fulldata.parquet"
    data, mask = interpolate(infile)


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
