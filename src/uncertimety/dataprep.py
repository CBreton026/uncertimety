"""
Census Data Preparation Module

This module handles the import, cleaning, and transformation of census data
for the uncertimety project. It provides functions for:
- Importing and cleaning census datasets
- Harmonizing vintage labels across datasets
- Standardizing dwelling type categories
- Validating data consistency

Usage:
    from uncertimety.dataprep import import_census_dataset

    # Import census data
    datasets = import_census_dataset(replacements=config["census"]["type_map"])
"""

import re
import toml
import itertools
import pandas as pd
import numpy as np
from pathlib import Path
from math import isclose
from typing import Optional, Dict, List, Tuple, Union, Iterable, Any
from pandas import testing as tm
from uncertimety.logger import init_logger
from uncertimety.config_loader import load_config
from uncertimety.random_control import auto_seed_from_config, PY_RANDOM, NP_RANDOM
from IPython.display import display  # FIXME only for temp test
from datetime import datetime
from dateutil import relativedelta

logger = init_logger()

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
CONFIG = DATA_DIR / "config.toml"
if not CONFIG.exists():
    msg = f"No TOML file found at: {CONFIG}"
    logger.error(msg)
    raise FileNotFoundError(msg)


def load_dataset_config(config_path: Path) -> Dict:
    """Load and validate the dataset configuration."""
    try:
        config = load_config(config_path)
        required_keys = [
            "dwelling_stock.metadata",
            "dwelling_stock.historic_vintages",
            "dwelling_stock.target_dwelling_types",
        ]

        for key_path in required_keys:
            parts = key_path.split(".")
            temp = config
            for part in parts:
                if part not in temp:
                    raise KeyError(f"Missing key: {key_path}")
                temp = temp[part]

        return config
    except (FileNotFoundError, toml.TomlDecodeError) as err:
        logger.error(f"Failed to load configuration from {config_path}: {err}")
        raise


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


def clean_name(
    name: str,
    replacements: Optional[Dict[str, str]] = None,
    sep: str = "_",
    remove_numbers: bool = False,
    pattern: Optional[str] = None,
) -> str:
    """
    Normalize and clean a string by replacing punctuation and whitespace,
    applying word-level substitutions, and formatting with a given separator.
    """
    if replacements is None:
        replacements = {}

    # Optional: remove numbers (standalone or in parentheses)
    if remove_numbers:
        # remove patterns like "(123)", standalone numbers, or " - 32"
        name = re.sub(r"\(\s*\d+\s*\)", "", name)  # remove (123)
        name = re.sub(r"\b\d+\b", "", name)  # remove standalone numbers

    if pattern is None:
        # Match one or more spaces or ASCII punctuation characters
        pattern = r"[\s!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~]{1,}"

    # replaces matches by a single space, and enforces lower caps
    normalized = re.sub(pattern, " ", name).lower()

    # splits the normalized name on spaces (" ");then strips the resulting substrings by using map; checks for any empty strings (i.e., "") using filter; list groups the resulting words
    words = list(filter(None, map(str.strip, normalized.split(" "))))

    # on a word-by-word basis, changes any desired replacements. This prevents changing substrings that are parts of words (e.g., the 'or' in before won't be changed)
    for i, word in enumerate(words):
        if word in replacements:
            words[i] = replacements.get(word)

    # looks at the sep-joined words, and changes any remaining punctuation from pattern to the selected "sep"; repeated punctuation are also changed.
    clean_name = re.sub(pattern, sep, sep.join(list(filter(None, words))))

    return clean_name.lower()


def normalize_vintage_label(vintage: str, sep="-") -> str:
    """Normalize a single vintage label."""
    replacements = {"or": "", "to": ""}
    vintage = clean_name(vintage, replacements=replacements, sep=sep)

    # FIXME: it might make more sense to add the sep after "lt" or "ge" too - this could be more consistent?

    if "total" in vintage:
        return "total"
    elif "before" in vintage:
        return "le" + sep + vintage.split(sep)[0].strip()
    elif "after" in vintage or ">" in vintage:
        return "ge" + sep + vintage.split(sep)[0].strip()
    elif "1" in vintage.split(sep):
        # covers the case of e.g., 1986-1, 1996-1, 1986 (1), 1961-1971(1)
        # here, we keep the '1' as a sign that the last year is incomplete - it will have to be split when harmonizing the vintages
        return sep.join(list(map(str.strip, vintage.split(sep))))
    else:
        return vintage


def clean_vintage(vintage_list, sep="-"):
    """Clean and normalize a list of vintage labels."""
    return [normalize_vintage_label(v, sep=sep) for v in vintage_list]


def normalize_column_name(col_name: str, replacements: dict, sep="_") -> str:
    """Normalize a single column label."""
    normalized = clean_name(
        col_name,
        replacements={
            "house": "",
        },
        sep=sep,
        remove_numbers=True,
    )

    # special cases
    if normalized in ("census_year", "vintage", "total"):
        return normalized
    elif normalized == "apartment":
        return "apartments"  # <- apartment is in several cols, special case

    # other cases
    for pattern, replacement in replacements.items():
        if pattern in normalized:
            return replacement

    return normalized


def clean_col_names(col_names, replacements, sep):
    return [
        normalize_column_name(name, replacements=replacements, sep=sep)
        for name in col_names
    ]


def import_census_dataset(
    data_dir: str = DATA_DIR, replacements: dict = None
) -> Dict[str, pd.DataFrame]:
    """
    Import and clean census datasets from CSV files.

    This function loads datasets exported from B2020 and completed manually from census PDFs,
    then returns them in a cleaned dictionary of dataframes indexed by census year.

    The dataframe is returned in long format. To convert it, use e.g.: df.pivot(index=['census_year', 'vintage'], columns='type', values='stock')

    the units are dwellings.

    Args:

        data_dir (str, optional): Path to directory containing the census files.

    Returns:
        Dict[str, pd.DataFrame]: Cleaned dataframes keyed by census year.

    Raises:
        ValueError: If no files matching the pattern are found.
    """
    if replacements is None:
        msg = f"No replacements dict was passed, received: {replacements}."
        logger.error(msg)
        raise ValueError(msg)

    pattern = data_dir / "census" / "qc_s_c_g*.csv"
    infiles = list(pattern.parent.glob(pattern.name))

    if not infiles:
        msg = f"No census CSV files found in path: {pattern}. Check that the data directory exists and contains CSV files matching the pattern."
        logger.error(msg)
        raise ValueError(msg)

    imported_dfs = {}
    for infile in infiles:
        try:
            df = pd.read_csv(infile, encoding="utf-8", sep=",")
            logger.info(f"Loaded file: {infile.name}")
        except (FileNotFoundError, OSError, UnicodeDecodeError) as err:
            logger.exception(f"Failed to read file '{infile}': {err}")
            raise ValueError(f"Could not read file: {infile}") from err

        # Extract census year from filename
        census_year = "".join(filter(str.isdigit, infile.stem[-4:]))
        if len(census_year) != 4:
            msg = f"Could not extract a valid 4-digit census year from filename: {infile.name}"
            logger.error(msg)
            raise ValueError(msg)

        df["census_year"] = census_year

        # Clean and rename columns
        original_cols = df.columns.to_list()
        print(original_cols)  # FIXME, temp crutch
        # FIXME - validate how other_movable is treated, as it is now
        # aggregated with mobile, however, mobile != other_movable - this
        # overestimates the number of mobile dwellings. however, this should have little to no impacts for the purposes of this research given the low number of mobile dwellings
        df.columns = clean_col_names(original_cols, replacements=replacements, sep="_")
        logger.debug(f"Cleaned columns for {infile.name}: {df.columns.tolist()}")

        # Figure out the vintage column
        pattern = "const"
        possible_vintage = df.filter(like=pattern, axis=1).columns

        if possible_vintage.empty:
            # TODO case for year 2001 where poor data was available (see FIXME below)
            try:
                # FIXME: no edge case required - it's not the same file format! it's named qc_s_g* instead of qc_s_c_g. thus, this code block is unnecessary
                if str(census_year) == "2001":
                    # Attempt fallback, e.g., first column
                    fallback_col = df.columns[0]
                    logger.warning(
                        f"Using fallback vintage column for year 2001: {fallback_col}"
                    )
                    df.rename(columns={fallback_col: "vintage"}, inplace=True)
                    # FIXME here comes the specific 2001 treatment
                    possible_vintage = pd.Index(
                        ["vintage"]
                    )  # ensure consistent downstream
                else:
                    raise ValueError("No fallback strategy available for this year")
            except Exception as err:
                err_type = type(err).__name__
                err_msg = str(err)
                full_msg = (
                    f"Could not find a suitable vintage column in file '{infile.name}'. "
                    f"Caught {err_type}: {err_msg}"
                )
                logger.exception(full_msg)
                raise ValueError(full_msg) from err

        elif len(possible_vintage) > 1:
            msg = (
                f"Too many suitable vintage columns in file '{infile.name}'. "
                f"Found: {list(possible_vintage)}"
            )
            logger.error(msg)
            raise ValueError(msg)

        # Rename vintage column explicitly
        vintage_col = possible_vintage[0]
        df.rename(columns={vintage_col: "vintage"}, inplace=True)

        # Clean vintage values
        df["vintage"] = clean_vintage(df["vintage"].to_list())

        # Reorder columns for clarity
        ordered_cols = ["census_year", "vintage"] + [
            col for col in df.columns if col not in ("census_year", "vintage")
        ]  # FIXME use meta_cols default parameter
        df = df[ordered_cols]

        # Save in imported dfs
        imported_dfs[census_year] = df
        logger.info(f"Successfully processed census year {census_year}.")

    return imported_dfs


def load_and_normalize_overwrites(toml_file: str) -> dict:
    """
    Loads and normalizes overwrite data from TOML.
    Replaces TOML-safe placeholders (-1) with Python None and corrects field names.
    """
    # FIXME - not DRY compared to config_loader - use it, e.g., in overwrite_census_dataset?

    with open(toml_file, "r") as f:
        raw = toml.load(f)

    overwrites = {}
    for year, fields in raw.items():
        normalized_fields = {}
        for name, value in fields.items():
            if isinstance(value, list):
                normalized_fields[name] = [None if i == -1 else i for i in value]
            elif isinstance(value, dict):
                normalized_fields[name] = {
                    key: None if val == -1 else val for key, val in value.items()
                }
            else:
                normalized_fields[name] = None if value == -1 else value
        overwrites[str(year)] = normalized_fields
    return overwrites


def apply_overwrites(df, instructions, year):
    """
    Applies column-level overwrites to an existing DataFrame.
    """
    # TODO unit tests
    for col, values in instructions.items():
        if col in {"create_from", "vintage_patch", "drop_rows_starting_at"}:
            continue
        try:
            df[col] = values
            logger.debug(f"Overwritten '{col}' for year {year}")
        except Exception as err:
            logger.warning(f"Could not overwrite '{col}' in year {year}: {err}")


def patch_vintage_and_trim(df, instructions):
    """
    Modifies the 'vintage' column and trims rows if specified in instructions.
    """
    # TODO unit tests
    patch = instructions.get("vintage_patch")
    drop_from = instructions.get("drop_rows_starting_at")

    if patch:
        try:
            row = int(patch["row"])
            df.at[row, "vintage"] = patch["value"]
            df.at[row, "census_year"] = patch["census_year"]
        except Exception as err:
            logger.warning(f"Could not patch vintage: {err}")

    if drop_from is not None:
        try:
            df.drop(index=df.index[drop_from:], inplace=True)
        except Exception as err:
            logger.warning(f"Could not drop rows from {drop_from}: {err}")


def overwrite_census_dataset(
    dataframes: Dict[str, pd.DataFrame], data_dir: str = DATA_DIR
) -> Dict[str, pd.DataFrame]:
    # TODO add unittests relevant for this function, and the functions it calls (load_and_normalize_overwrites, patch_vintage_and_trim, apply_overwrites)
    """
    Overwrites or creates census DataFrames using manual values from a TOML file.

    Parameters:
        infile (str): Path to TOML file with manual overwrites.
        dataframes (dict): Dict of pandas DataFrames keyed by census year as strings.

    Returns:
        dict: The updated dictionary of DataFrames.
    """
    infile = data_dir / "manual_overwrite.toml"

    if not infile:
        msg = f"No TOML file found for path: {infile}"
        logger.error(msg)
        raise ValueError(msg)

    overwrites = load_and_normalize_overwrites(infile)

    for year, instructions in overwrites.items():
        if year in dataframes:
            logger.info(f"Overwriting existing DataFrame for year {year}")
            apply_overwrites(dataframes[year], instructions, year)
        else:
            logger.info(f"Creating new DataFrame for year {year}")
            try:
                ref_year = str(instructions["create_from"])
                base_df = dataframes[ref_year].iloc[:, :2].copy()
                base_df["census_year"] = str(year)
                dataframes[year] = base_df
            except KeyError:
                logger.error(
                    f"Reference year {ref_year} not found to create year {year}"
                )
                continue

            patch_vintage_and_trim(dataframes[year], instructions)
            apply_overwrites(dataframes[year], instructions, year)
    # TODO check that other_dwelling equals the sum of relevant dwellings, that apartments is approx the sum of apartments, that total makes sense, etc. this could be its own function later on

    return dataframes


def check_year(year) -> int:
    """
    Validates and converts a year to int.

    Accepts strings or integers, checks for valid range (1608–2149).

    Args:
        year (str or int): The year value.

    Returns:
        int: A validated, coerced year.

    Raises:
        TypeError: If input is not str or int.
        ValueError: If value is not within accepted year range.
    """
    try:
        year_int = int(year)
    except (ValueError, TypeError):
        raise TypeError(
            f"Expected int or str convertible to int, got {type(year).__name__}: {year}"
        )

    if year_int < 0:
        raise ValueError(f"Year must be non-negative, got: {year}")
    elif year_int not in range(1608, 2150):
        raise ValueError(
            f"Encountered unexpected year (must be 1608–2149), got: {year_int}"
        )

    return year_int


def round_to_next_5(year) -> int:
    """
    Rounds a year up to the next multiple of 5.

    Args:
        year (str or int): A valid year.

    Returns:
        int: The next multiple of 5 above the given year.
    """
    year = check_year(year)
    return ((year // 5) + 1) * 5


def infer_last_full_year(census_year) -> int:
    """
    Infers the last full cohort year based on the given census year.

    This is used to cap cohort ranges like '1986+' or partial census intervals.

    Examples: #FIXME fix these examples
        - 1986 → 1985 (→ 1986-1990 cohort)
        - 1996 → 1995 (→ 1986-1995 span)
        - 2001 → 2000 (→ 2001-2005 cohort)

    Args:
        census_year (str or int): Census year to use as cutoff.

    Returns:
        int: Last full year (typically ending in a 0 or 5).
    """
    census_year = check_year(census_year)
    return ((census_year - 1) // 5) * 5


def get_monthly_activity(inactive_months: int = None) -> list:
    """_summary_

        the number of months with no construction activity, i.e., no new dwellings built, starting in January. is passed to get_monthly_activity, which returns a list of 12 floats summing to 1, representing the fraction of annual construction assumed to occur in each month. Default is uniform (1/12 per month).

    Args:
        inactive_months (int, optional): _description_. Defaults to None.

    Raises:
        ValueError: _description_

    Returns:
        list: _description_
    """

    if inactive_months is None:
        inactive_months = 0

    if inactive_months < 0 or inactive_months > 11:
        msg = f"inactive_months must be an int between 0 and 11, received {inactive_months}"
        raise ValueError(msg)

    active_months = 12 - inactive_months
    return [0] * inactive_months + [1 / active_months] * active_months


def get_vintage_shares(
    vintage_label: str,
    sep: str = "-",
    inactive_months: int = None,
    census_month: int = 5,
) -> list[float]:
    """
    Calculates the proportion of construction in a vintage range that:
    - occurred over full years
    - occurred in the final (incomplete) census year

    Args:
        vintage_label (str): e.g. "1946-1960"
        sep (str): separator, typically "-"
        inactive_months (int): number of months with no construction in a year
        census_month (int): the month (Jan = 1, May = 5) in which the census takes place in the incomplete census year

    Returns:
        list[float]: [full_years_share, census_year_share], guaranteed to sum to 1.0

    Raises:
        ValueError: If shares do not sum to 1.0
    """

    monthly_activity = get_monthly_activity(inactive_months=inactive_months)

    try:
        start_str, end_str = vintage_label.strip().split(sep)[:2]
        start_year = datetime.strptime(start_str.strip(), "%Y")
        end_year = datetime.strptime(end_str.strip(), "%Y")
    except (ValueError, IndexError) as err:
        msg = (
            f"Invalid vintage label '{vintage_label}'; expected format 'YYYY{sep}YYYY'."
        )
        logger.error(msg)
        raise ValueError(msg) from err

    full_years = relativedelta.relativedelta(end_year, start_year).years
    if full_years < 0:
        msg = f"Start year {start_str} must be before end year {end_str} in label."
        logger.error(msg)
        raise ValueError(msg)

    # Sum monthly activity in the final (census) year; January to last month before census in the incomplete year, defaults to may
    census_year_share = sum(monthly_activity[:census_month])

    full_years_share = full_years * sum(monthly_activity)  # full years * 1
    total = full_years_share + census_year_share

    shares = [full_years_share / total, census_year_share / total]
    if not _check_sums(shares, target=1.0):
        msg = f"Shares do not sum to 1.0: {shares}"
        logger.error(msg)
        raise ValueError(msg)

    return shares


def _check_sums(
    values: Union[Iterable[float], "np.ndarray", "pd.Series"],
    target: float = 1.0,
    rtol: float = 1e-9,
    atol: float = 1e-9,
    raise_error: bool = False,
) -> bool:
    """
    Check whether numeric values sum approximately to a target value (default: 1.0).

    Args:
        values (Iterable[float] | np.ndarray | pd.Series): A list-like container of floats. If values is int or float, converts to list first.
        target (float): Expected total sum (default: 1.0).
        rtol (float): Relative tolerance for closeness check.
        atol (float): Absolute tolerance for closeness check.
        raise_error (bool): Whether to raise ValueError on failure.

    Returns:
        bool: True if values sum to target within tolerance, else False.

    Raises:
        TypeError: If input is not a recognized numeric iterable.
        ValueError: If sum does not match target and `raise_error=True`.
    """
    try:
        if isinstance(values, float) or isinstance(values, int):
            values = [values]

        values = list(values)  # Accept Series, np.ndarray, etc.
        total = sum(values)
    except TypeError as err:
        logger.error("Invalid input: values must be iterable of floats", exc_info=True)
        raise TypeError("Input must be an iterable of numbers") from err

    if not all(isinstance(x, (int, float)) for x in values):
        raise TypeError("All elements must be numeric (int or float)")

    if isclose(total, target, rel_tol=rtol, abs_tol=atol):
        return True

    # If it's not close
    logger.warning(
        f"Sum of share(s) is {total:.12f}, target was {target:.12f} "
        f"(rtol={rtol}, atol={atol})"
    )  # FIXME this warning is normal/intended when single shares are tested in harmonize_vintage labels. remove?

    if raise_error:
        raise ValueError(f"Sum {total:.12f} is not close to target {target:.12f}")

    return False


def _extract_year_from_token(token: str, symbol: str, sep="-") -> int:
    """
    Extracts a year from a token like '<1920' or '1986+' by splitting on symbol.

    Used for error handling based on previous naming conventions.
    """
    # TODO check if applications in other functions in dataprep.py - if not, delete?
    try:
        year_str = token.split(symbol)[0] if symbol == "+" else token.split(symbol)[1]

        if not year_str.isdigit():
            raise ValueError

        if symbol == "+" or symbol == ">":
            return "ge" + sep + year_str
        elif symbol == "<":
            return "le" + sep + year_str
        else:
            msg = f"Could not recognize symbol {symbol}"
            logger.error(msg)
            raise ValueError

    except Exception as err:
        raise ValueError(
            f"Could not extract year from '{token}' using symbol '{symbol}'"
        ) from err


def drop_duplicate_rows(
    df: pd.DataFrame,
    meta_cols: List[str] = META_COLS,
) -> pd.DataFrame:
    """
    Removes duplicate vintage rows that contain only NaNs in data columns.
    Keeps one row per vintage. Raises if multiple rows have valid data.

    Args:
        df (pd.DataFrame): Input dataframe with 'vintage' column.
        data_cols (List[str]): Columns to check for NaN vs valid data.

    Returns:
        pd.DataFrame: Deduplicated dataframe.
    """
    # FIXME read meta_cols from config?
    if "vintage" not in df.columns:
        raise ValueError("Expected 'vintage' column in dataframe")

    data_cols = df.columns.difference(meta_cols)
    deduped_rows = []

    for vintage, group in df.groupby("vintage", sort=False):
        valid_rows = group[group[data_cols].notna().any(axis=1)]
        nan_rows = group[~group.index.isin(valid_rows.index)]

        if len(valid_rows) > 1:
            raise ValueError(f"Multiple non-NaN rows found for vintage '{vintage}'.")

        if not valid_rows.empty:
            deduped_rows.append(valid_rows.iloc[0])
        elif not nan_rows.empty:
            deduped_rows.append(nan_rows.iloc[0])
    return pd.DataFrame(deduped_rows).reset_index(drop=True)


def validate_vintage_interval(interval: Tuple[int, int]) -> None:
    """Ensures vintage interval is valid, i.e., end >= start."""
    start, end = interval

    if int(end) < int(start):
        raise ValueError(
            f"Invalid vintage interval: start={start} is greater than end={end}"
        )


def parse_single_vintage(
    label: str,
    census_year: int,
    sep: str = "-",
    model_start: int = 1608,
    last_census: int = 2021,
    parse_census_year: bool = True,
    round_last_vintage: bool = False,
) -> tuple[list[tuple[int, int]], list[float]]:
    # FIXME: rename model_start? default to zero or None?
    """
    Parses a single vintage label into [start, end] intervals and associated shares.

    Handles:
    - 'le-1920' → (model_start, 1920)
    - 'ge-1986' → (1986, round_to_next_5(census_year))
    - '1991-1' or '1971-1981-1' → split vintage
    - 'total' → (model_start, round_to_next_5(census_year))

    Returns:
        tuple: (list of (start, end), list of shares)
    """
    label = str(label).strip()
    census_year = check_year(census_year)
    last_census = (
        round_to_next_5(last_census) if round_last_vintage else last_census
    )  # FIXME add check_year()?

    intervals = []
    shares = [1.0]

    # Set 'final year' of census
    last_vintage = round_to_next_5(census_year) if round_last_vintage else census_year

    try:
        parts = label.split(sep)
    except (ValueError, IndexError) as err:
        logger.error(f"Failed to split label '{label}': {err}")
        return [], []

    # === Handle special cases first
    if label.lower() == "total":
        intervals = [(model_start, last_census)]

    elif "le" in parts:
        try:
            year = check_year(parts[-1])
            intervals = [(model_start, year)]

        except ValueError as err:
            logger.error(err)
            return [], []

    elif "ge" in parts:
        try:
            year = check_year(parts[-1])
            intervals = [(year, last_vintage)]

        except ValueError as err:
            logger.error(err)
            return [], []

    # === Incomplete census year (e.g., 1991-1, 1971-1981-1)
    elif "1" in parts:
        bounds = parts[:-1]
        if len(bounds) == 1:  # i.e., only one year
            year = check_year(bounds[0])
            # intervals = [(year, round_to_next_5(year))]  # FIXME: should add has a single year, not a block of years. Otherwise, it 'opens' the bounds of total too much. in 1986, total ends in 1986, not in 1990!
            intervals = [(year, last_vintage)]
        elif len(bounds) == 2:
            start, end = map(check_year, bounds)
            if end > census_year:
                # Manage case when splitting a census year; this prevents edge case where for census 1986, '1986-1990' was parsed (due to census year), then passed as 1986-1990-1 and split into 1986-1985 and 1990-1986
                logger.warning(
                    f"Bounds exceed census year: {bounds} in label '{label}' for census {census_year}"
                )
                intervals = [(start, last_vintage)]
            else:
                intervals = [
                    (start, infer_last_full_year(end)),  # full years
                    # (end, round_to_next_5(end)),  # partial census year
                    (end, last_vintage),  # FIXME see above
                ]

                # Enforce strict ordering
                # FIXME why round to five? why not directly aim for final categories? jsut to keep details? or this is treated later?
                shares = get_vintage_shares(label, sep=sep)

    # === Standard YYYY-YYYY case
    elif len(parts) == 2:
        start, end = map(check_year, parts)

        #  Case if census year falls in YYYY-YYYY range; for later censuses (2006+), there's no (1) mention to identify a partial census year
        if parse_census_year and start <= census_year <= end:
            # Then treat as if it was a split year
            label_with_flag = f"{start}{sep}{end}{sep}1"
            return parse_single_vintage(
                label_with_flag,
                census_year=census_year,
                last_census=last_census,
                sep=sep,
                model_start=model_start,
                parse_census_year=False,  # Prevents recursion
                round_last_vintage=round_last_vintage,
            )
        intervals = [(start, end)]

    else:
        logger.warning(f"Unexpected label format: {label}")
        try:
            # Attempt to "translate" the unmatched label
            symbol = re.sub(r"\d", "", parts[0])  # matches and replaces all digits
            new_label = _extract_year_from_token(parts[0], symbol)
            return parse_single_vintage(
                new_label,
                census_year,
                sep=sep,
                model_start=model_start,
                parse_census_year=parse_census_year,
                round_last_vintage=round_last_vintage,
            )
        except Exception as err:
            logger.error(f"Failed to parse label '{label}': {err}")

    # === Validate and return
    try:
        for interval in intervals:
            validate_vintage_interval(interval)
    except ValueError as err:
        logger.error(f"Invalid interval in label '{label}': {err}")
        return [], []

    return intervals, shares


def _validate_data_preservation(
    original_row: pd.Series,
    new_rows: list[dict] | list[pd.Series],
    data_columns: list[str],
    idx: int = None,
    original_label: str = None,
    atol: float = 5,
    rtol: float = 1e-5,
) -> list[dict]:  # fixme rename to row preservation?
    """
    Validate that the sum of numeric values in the new rows equals the original row.

    See https://pandas.pydata.org/docs/reference/api/pandas.testing.assert_series_equal.html#pandas.testing.assert_series_equal

    Args:
        original_row (pd.Series): Original row from the DataFrame.
        new_rows (list): List of dict-like or Series-like rows after harmonization.
        data_columns (list[str]): Columns to validate.
        idx (int, optional): Index of the original row (for error message).
        original_label (str, optional): Label or ID of the original row.
        atol (float): Absolute tolerance allowed.
        rtol (float): Relative tolerance allowed.

    Returns:
        list[dict]: Same `new_rows`, if validation passes.

    Raises:
        ValueError: If the new rows do not preserve the total values in `data_columns`.
    """
    try:
        orig = original_row[data_columns].apply(pd.to_numeric, errors="coerce")
        summed = (
            pd.DataFrame(new_rows)[data_columns]
            .apply(pd.to_numeric, errors="coerce")  # invalid parsing set as NaN
            .sum(skipna=True, min_count=1)
        )

        tm.assert_series_equal(
            orig,
            summed,
            check_dtype=False,
            check_exact=False,
            rtol=rtol,
            atol=atol,
            check_names=False,
        )
        return new_rows

    except AssertionError as err:
        msg = f"Value mismatch after splitting row {idx} (label '{original_label}'): {err}"
        logger.warning(msg)
        raise ValueError(msg) from err


def add_missing_vintages_types(
    dataframe: pd.DataFrame,
    target_vintages: Optional[List[str]] = None,
    target_types: Optional[List[str]] = None,
    config_dir=CONFIG,
    sep: str = "-",
    meta_cols=["census_year", "vintage", "source", "split"],
) -> pd.DataFrame:
    """
    Ensures the dataframe contains all expected dwelling types and vintages.

    - Adds missing dwelling type columns (filled with NaN)
    - Adds missing vintage rows (filled with 0 or NaN depending on vintage position)

    Args:
        dataframe (pd.DataFrame): Input data with 'vintage', 'census_year' and dwelling type columns
        target_vintages (List[str], optional): List of expected vintage strings (e.g., "1946-1970")
        target_types (List[str], optional): List of expected dwelling type column names
        data_dir (Path): Path to directory containing config.toml
        sep (str): Separator for vintage labels (default: "-")

    Returns:
        pd.DataFrame: Updated dataframe with full set of vintages and types
    """
    if target_types is None:
        try:
            target_types = config["dwelling_stock"]["total_dwelling_types"]
        except (NameError, KeyError) as err:
            msg = f"Failed to access target types in config at {CONFIG}"
            logger.error(msg)
            raise err

    if target_vintages is None:
        try:
            target_vintages = config["dwelling_stock"]["historic_vintages"]
        except (NameError, KeyError) as err:
            msg = f"Failed to access target types in config at {CONFIG}"
            logger.error(msg)
            raise err

    unique_years = dataframe["census_year"].unique()
    if len(unique_years) != 1:
        msg = f"Expected a single census year, but found multiple: {unique_years.to_list()}"
        logger.error(msg)
        raise ValueError(msg)

    census_year = int(unique_years[0])
    type_cols = dataframe.columns.difference(["census_year", "vintage"])
    present_vintages = set(dataframe["vintage"].astype("str"))
    present_types = set(type_cols)

    missing_types = set(target_types) - present_types
    missing_vintages = set(target_vintages) - present_vintages

    # Add missing dwelling type columns filled with NaN
    for missing_col in missing_types:
        logger.warning(f"Adding missing column: {missing_col}")
        dataframe[missing_col] = np.nan

    # Add missing vintages
    new_rows = []
    for missing_vintage in sorted(
        missing_vintages
    ):  # FIXME sorted(a, key=lambda student: student[1])
        try:
            start, end = [check_year(year) for year in missing_vintage.split(sep)]
        except Exception as err:
            logger.warning(
                f"Skipping malformed vintage label '{missing_vintage}': {err}"
            )
            continue

        if start > census_year:
            values = {dwellings: 0 for dwellings in target_types}
        else:
            values = {dwellings: np.nan for dwellings in target_types}

        new_row = {
            "census_year": census_year,
            "vintage": missing_vintage,
            **values,
        }
        new_rows.append(new_row)

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        dataframe = pd.concat([dataframe, new_df], ignore_index=True, sort=False)

    # ==== Fix column order
    data_cols = dataframe.columns.difference(meta_cols)
    ordered_cols = (
        meta_cols[0:2] + data_cols.to_list() + meta_cols[2:]
    )  # FIXME this is hardcoded - manage other 'meta' inputs without magic number?

    return dataframe[[col for col in ordered_cols if col in dataframe.columns]]


def harmonize_vintage_labels(
    df: pd.DataFrame,
    sep: str = "-",
    model_start=1608,
    last_census=2021,
    meta_cols=["census_year", "vintage", "source", "split"],
    round_last_vintage=True,
) -> pd.DataFrame:
    """
    Harmonizes vintage labels in the input DataFrame into specified target vintage intervals.

    Args:
        df (pd.DataFrame): Input dataframe with a 'vintage' column and numerical data columns.
        target_vintages (List[Tuple[int, int]]): Standardized intervals to harmonize vintages into.
        sep (str): Separator used in vintage labels.

    Returns:
        pd.DataFrame: A harmonized version of the input dataframe.
    """
    logger.info("Starting vintage label harmonization on %d rows", len(df))
    data_columns = df.columns.difference(["vintage", "census_year"])
    expanded_rows = []

    # Create an expanded dataframe
    for idx, row in df.iterrows():
        original_label = str(row["vintage"])
        logger.debug("Parsing vintage label '%s' at index %d", original_label, idx)

        parsed_intervals, shares = parse_single_vintage(
            original_label,
            census_year=row["census_year"],
            sep=sep,
            model_start=model_start,
            round_last_vintage=round_last_vintage,
        )
        new_rows = []

        if not parsed_intervals or not shares:
            msg = f"Could not parse vintage label '{original_label}' at index {idx}"
            logger.warning(msg)  # FIXME Raise ValueError?
            continue

        # Fix the labels and shares
        for interval, share in zip(parsed_intervals, shares):
            start, end = interval
            new_label = f"{start}{sep}{end}"
            new_row = row.copy()
            new_row["vintage"] = new_label
            new_row["source"] = (
                idx,
                original_label,
            )  # NOTE useful to check data preservation at the dataframe level

            # Check if the original values were modified (shares != 1.0)
            new_row["split"] = (
                True if not _check_sums(share, target=1.0, raise_error=False) else False
            )

            for col in data_columns:
                val = row[col]
                if pd.isna(val):
                    new_row[col] = np.nan
                else:
                    val_out = int(round(val * share))
                    new_row[col] = val_out

            new_rows.append(new_row)

        # === Consistency check: test if numeric data is preserved ===
        try:
            new_rows = _validate_data_preservation(
                row,
                new_rows,
                data_columns=data_columns,
                idx=idx,
                original_label=original_label,
            )
            logger.debug(
                "Harmonized label '%s' → %d intervals; data validated.",
                original_label,
                len(new_rows),
            )
        except ValueError as err:
            logger.error("Data preservation failed for row %d: %s", idx, err)
            raise

        expanded_rows.extend(new_rows)
    expanded_df = pd.DataFrame(
        expanded_rows
    ).reset_index(
        drop=True
    )  # FIXME ensure correct column order, based on 'target' (?) [df.columns.tolist() + ["source", "split"]]

    if len(expanded_df) == len(df):
        logger.info(
            "Finished harmonization: kept the %d initial rows", len(expanded_df)
        )
    else:
        logger.info("Finished harmonization: expanded to %d rows", len(expanded_df))

    # ==== Consistency check : remove empty duplicate rows
    deduped_df = drop_duplicate_rows(expanded_df)
    removed_rows = len(expanded_df) - len(deduped_df)
    if removed_rows > 0:
        logger.info(
            f"Dropping {removed_rows} empty duplicated rows from the dataframe."
        )

    return deduped_df


def vintage_label_to_tuple(label: str, sep: str = "-") -> Tuple[int, int]:
    """Converts 'YYYY-YYYY' vintage label into (start, end) tuple."""
    # FIXME - add some of the previous logic from normalize_vintage_label?
    # NOTE REPRENDRE
    try:
        start, end = map(int, str(label).strip().split(sep))
        return (start, end)
    except ValueError:
        raise ValueError(f"Invalid vintage label format: {label}")


def convert_df_to_int(
    df: pd.DataFrame,
    meta_cols: List[str] = ["census_year", "vintage", "source", "split"],
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Converts all non-meta columns in the DataFrame to nullable Int32 dtype.

    Args:
        df (pd.DataFrame): Input DataFrame.
        meta_cols (List[str]): Columns to exclude from conversion.
        inplace (bool): Whether to modify df in-place.

    Returns:
        pd.DataFrame: DataFrame with converted integer columns.
    """
    working_df = df if inplace else df.copy()
    data_cols = [col for col in df.columns if col not in meta_cols]

    # Safely convert only numeric (float/int) columns
    for col in data_cols:
        if pd.api.types.is_numeric_dtype(working_df[col]):
            working_df[col] = working_df[col].astype("Int32")

    return working_df


def calculate_missing_types(
    df: pd.DataFrame,
    agg_types: Optional[Dict[str, List[str]]] = None,
    config_dir=CONFIG,
    meta_cols: List[str] = META_COLS,
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Fills missing dwelling stock types by summing their component subtypes.

    Args:
        df (pd.DataFrame): Input DataFrame with dwelling type data.
        agg_types (dict, optional): Mapping of aggregate dwelling types to lists of subtypes.
        config_dir (path-like, optional): Path to configuration if `agg_types` is not provided.
        meta_cols (list, optional): Columns to exclude from processing.
        inplace (bool): Whether to modify `df` in-place. If False, returns a copy.

    Returns:
        pd.DataFrame: DataFrame with missing dwelling types imputed.
    """
    if inplace:
        orig_df = df.copy()  # keep copy for data validation

    working_df = df if inplace else df.copy()
    census_year = int(working_df["census_year"].unique()[0])
    logger.debug(f"Checking for missing dwelling counts in census {census_year}")

    data_cols = [col for col in working_df.columns if col not in meta_cols]
    nan_cols = working_df[data_cols].isna().any()

    if agg_types is None:
        try:
            agg_types = config["dwelling_stock"][
                "agg_types"
            ]  # FIXMETODO LOAD AT TOP OF MODULE
        except (ImportError, KeyError, NameError) as err:
            msg = f"Failed to access aggregated types in config at {config_dir}"
            logger.error(msg)
            raise RuntimeError(msg) from err

    for target_col in nan_cols[nan_cols].index:
        subtypes = agg_types.get(target_col)
        if not subtypes:
            continue  # No aggregation is possible for this column

        logger.info(
            f"Attempting to fill missing values for '{target_col}' from subtypes: {subtypes}"
        )
        try:
            # Compute sum only if all subtype values are available
            summed = working_df[subtypes].sum(axis=1, min_count=len(subtypes))

            # Find where summing failed due to missing values
            missing_rows = summed.isna()

            if missing_rows.any():
                logger.warning(
                    f"Column '{target_col}' could not be calculated for {missing_rows.sum()} rows due to missing subtypes: {subtypes}"
                )
                # Only replace values where the result is valid (i.e., not NaN)
                working_df.loc[~missing_rows, target_col] = summed[~missing_rows]
            else:
                working_df[target_col] = summed
                logger.debug(
                    f"Filled all values for '{target_col}' using subtypes: {subtypes}"
                )

        except KeyError as err:
            logger.warning(
                f"Skipping '{target_col}': one or more subtypes missing from DataFrame: {err}"
            )

        # Validate preservation of original data
        _validate_frame_preservation(orig_df if inplace else df, working_df)

    return working_df


def _filter_numeric_subset(
    df: pd.DataFrame, vintages: list[str] = [], columns: list[str] = []
) -> pd.DataFrame:
    """Filter DataFrame by vintages and columns, converting to numeric."""
    present_cols = [
        col for col in columns if col in df.columns
    ]  # NOTE: this avoids a KeyError if a col in 'columns' is not present in the df.columns

    return df.loc[df["vintage"].isin(vintages), present_cols].apply(
        pd.to_numeric, errors="coerce"
    )


def _validate_frame_preservation(
    original_df: pd.DataFrame,
    new_df: pd.DataFrame,
    data_rows: list[str] = [],
    data_columns: list[str] = [],
    atol: float = 5,
    rtol: float = 1e-5,
) -> list[dict]:
    """
    Validate that the numeric values in `new_df` are close to those in `original_df`
    for the specified vintages and columns.

    Raises:
        ValueError: If the values differ beyond the allowed tolerance.
    """
    if data_rows is None:
        data_rows = config["dwelling_stock"]["historic_vintages"]

    if data_columns is None:
        data_columns = config["dwelling_stock"]["target_dwelling_types"] + [
            "other_attached_dwelling",
            "other_dwelling",
        ]

    orig = _filter_numeric_subset(original_df, data_rows, data_columns)
    new = _filter_numeric_subset(new_df, data_rows, data_columns)

    try:
        tm.assert_frame_equal(
            orig,
            new,
            check_dtype=True,
            check_exact=False,
            rtol=rtol,
            atol=atol,
            check_names=False,
        )
    except AssertionError as err:
        msg = f"Frame mismatch, values differ from original dataframe: {err}"
        logger.warning(msg)
        raise ValueError(msg) from err


def filter_relevant_types_vintages(
    df: pd.DataFrame,
    vintages: list[str] = HISTORIC_VINTAGES,
    target_types: list[str] = TARGET_TYPES,
    merged_cols: list[str] = MERGED_COLS,
    col_order: list[str] | None = None,
    inplace: bool = False,
    keep_other_cols: bool = False,
    meta_cols: list[str] = META_COLS,
) -> pd.DataFrame:
    """
    Filter and reorder a dwelling stock DataFrame based on vintages and column order.

    Parameters:
        df: Input DataFrame.
        vintages: List of vintages to keep (if None, load from config).
        col_order: List of main columns to keep and reorder (others are optional). The meta_cols are automatically added.
        inplace: Whether to operate on the DataFrame in-place.
        keep_other_cols: Whether to keep columns not listed in col_order and meta_cols
        meta_cols: list of additional columns to keep
        config_dir: Path to the config directory.

    Returns:
        A filtered and reordered DataFrame.
    """
    working_df = df if inplace else df.copy()

    # Reorder columns
    if col_order is None:
        col_order = merge_unique_ordered([meta_cols, target_types, merged_cols])
    else:
        col_order = merge_unique_ordered([meta_cols, col_order])

    if keep_other_cols:
        other_cols = [col for col in working_df.columns if col not in col_order]
    else:
        other_cols = []

    final_cols = (
        [col for col in col_order if col in working_df.columns] + other_cols
    )  # FIXME not DRY, have separate function to filter cols? see standardize_census()

    return working_df.loc[
        working_df["vintage"].isin(vintages), final_cols
    ]  # FIXME use, or replace by, filter_numeric_subset?


def standardize_census(
    df: pd.DataFrame,
    inplace: bool = False,
    vintages: list[str] = HISTORIC_VINTAGES,
    target_types: list[str] = TARGET_TYPES,
    meta_cols: list[str] = META_COLS[:2],
    merged_cols: list[str] = MERGED_COLS,
    col_order: list[str] | None = None,
) -> pd.DataFrame:
    """
    Merge 'other_attached_dwelling' and 'other_dwelling' intelligently,
    keeping the one with fewer NaNs, and standardize the result to 'other_dwelling'.
    Filters and reorders columns accordingly.
    """
    working_df = df if inplace else df.copy()

    # Reorder columns
    if col_order is None:
        col_order = merge_unique_ordered([meta_cols, target_types, ["other_dwelling"]])
    else:
        col_order = merge_unique_ordered(
            *[meta_cols, col_order, ["other_dwelling"]]
        )  # FIXME hardcoded; also, needs to be a set, otherwise it gets duplicated with successive passes

    # Standardize the 'other_dwelling' and 'other_attached_dwelling' types
    try:
        nan_count = working_df.loc[:, merged_cols].isna().sum()
        if nan_count["other_attached_dwelling"] < nan_count["other_dwelling"]:
            keep_col = "other_attached_dwelling"
        else:
            keep_col = "other_dwelling"

        if keep_col != "other_dwelling":
            working_df["other_dwelling"] = working_df[keep_col]
            working_df = working_df.drop(columns=[keep_col])
    except KeyError as err:
        msg = f"Could not find one or more {merged_cols} in DataFrame: {err}"
        logger.error(msg)
        raise

    return filter_relevant_types_vintages(
        working_df,
        vintages=vintages,
        col_order=col_order,
        meta_cols=meta_cols,
        keep_other_cols=False,
    )


def merge_unique_ordered(*lists: List[Any]) -> List[Any]:
    """
    Return an ordered list of unique items from the input lists.

    Handles both standard usage (a, b, c) and common mistake ([a, b, c]).

    Args:
        *lists (List[Any]): Any number of lists to merge.

    Returns:
        List[Any]: A flattened, ordered list of unique items.
    """
    # Handle common mistake: merge_unique_ordered([a, b, c]) instead of merge_unique_ordered(a, b, c)
    if (
        len(lists) == 1
        and isinstance(lists[0], list)
        and all(isinstance(sub, list) for sub in lists[0])
    ):
        lists = tuple(lists[0])  # unpack the inner list

    flat = itertools.chain.from_iterable(lists)
    return list(dict.fromkeys(flat))


def validate_dwelling_counts(
    df: pd.DataFrame,
    inplace: bool = False,
):
    working_df = df if inplace else df.copy()

    # TODO reprendre ici; voir ce que j'avais déjà codé précédemment si parties réutilisables.

    # if columns outside target cols or if vintages outside target vintages, then filter_relevant

    # accept a dataframe, filter relevant types and vintages, including other_dwelling
    # check sums over types; check if the 'total' agrees with the sums, using atol of ~5 (20?) and rtol 1e-5.
    # check sums over cohorts (0-2025) vs rest

    # check marginals : compare 'total' and '0-2025'
    # logger.warn any problems

    # IF there are issues, then launch IPFN procedure (?)
    return working_df


def check_series_sum(
    df: pd.DataFrame,
    target: str,  # either a row or column label
    groupby: str = "vintage",
    total_label: str = None,
    atol: float = 5,
    rtol: float = 1e-5,
    axis: bool = 0,
) -> Tuple[bool, pd.Series]:
    """
    Check if components sum to total within grouped data.

    Args:
        df: DataFrame with data to check
        value_col: Column containing values to sum (e.g., 'total')
        groupby: Column to group by (e.g., 'vintage')
        vintage_total: Label in groupby representing the total
        atol: Absolute tolerance for comparison (used by np.isclose)
        rtol: Relative tolerance for comparison (used by np.isclose)

    Returns:
        Tuple of (bool, Series) where:
            - bool indicates if all groups pass the check
            - Series contains the difference between total and sum of components for each group
    """
    if total_label is None:
        total_label = "total" if axis == 0 else "1608-2025"  # FIXME use constants?

    grouped = df.groupby(groupby).sum()

    if axis == 0:  # for rows, sum over types
        try:
            total = grouped.loc[target, total_label]
            components = grouped.loc[target].drop(total_label)
        except KeyError as err:
            raise KeyError(
                f"Target '{target}' not found in DataFrame {grouped.index}: {err}. Check that target and axis are consistent."
            )
    elif axis == 1:  # for columns, sum over vintages
        try:
            total = grouped.loc[total_label, target]
            components = grouped[target].drop(total_label)
        except KeyError as err:
            raise KeyError(
                f"Target '{target}' not found in DataFrame {grouped.columns}: {err}"
            )
    else:
        raise ValueError(f"Invalid axis {axis}. Use 0 for rows or 1 for columns.")

    # Sum the components
    component_sum = components.fillna(0).sum()

    # Calculate difference
    difference = total - component_sum

    # Check if within tolerance (using numpy's isclose for both absolute and relative tolerance)
    check_passed = np.isclose(total, component_sum, rtol=rtol, atol=atol)

    result = pd.Series(
        {
            "target": target,
            "total_label": total_label,
            "total": total,
            "component_sum": component_sum,
            "difference": difference,
            "check_passed": check_passed,
        }
    )

    return check_passed, result


# TODO Reprendre check_marginals
def check_marginals(
    df: pd.DataFrame,
    groupby: str = "vintage",
    vintage_label: str = "1608-2025",
    type_label: str = "total",
    atol: float = 5,
    rtol: float = 1e-5,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Check if the marginals (total counts) are consistent across vintages and types.

    Args:
        df: DataFrame with vintage and dwelling type data
        groupby: Column containing vintage labels (e.g., 'vintage')
        vintage_label: The vintage label representing the total (e.g., "1608-2025")
        type_label: The column name representing total dwellings (e.g., "total")
        atol: Absolute tolerance for comparison
        rtol: Relative tolerance for comparison

    Returns:
        Tuple[bool, Dict]:
            - Boolean indicating if marginals are consistent
            - Dictionary with detailed results including:
                - 'sum_by_type': Series with type sum details
                - 'sum_by_vintage': Series with vintage sum details
                - 'sums_match': Whether component sums match
                - 'difference': Difference between component sums
    """
    # Check that required columns exist
    if type_label not in df.columns or groupby not in df.columns:
        msg = f"Columns {type_label} or {groupby} are missing from DataFrame columns: {df.columns}"
        logger.error(msg)
        raise ValueError(msg)

    # Check that the required vintage label exists in the groupby column
    if vintage_label not in df[groupby].values:
        msg = f"Vintage label '{vintage_label}' not found in '{groupby}' column: {df[groupby].unique()}"
        logger.error(msg)
        raise ValueError(msg)

    try:
        # Check if dwelling types sum to the vintage total
        type_passed, sum_by_type = check_series_sum(
            df, target=vintage_label, groupby=groupby, atol=atol, rtol=rtol, axis=0
        )

        # Check if vintages sum to the type total
        vintage_passed, sum_by_vintage = check_series_sum(
            df, target=type_label, groupby=groupby, atol=atol, rtol=rtol, axis=1
        )
        # There are two things we need to check: first, that the component sums match for types and vintages agree; second, that this matches the total value

        # Compare the component sums from both approaches (should be equal)
        sums_match = np.isclose(
            sum_by_type["component_sum"],
            sum_by_vintage["component_sum"],
            atol=atol,
            rtol=rtol,
        )

        difference = sum_by_type["component_sum"] - sum_by_vintage["component_sum"]

        # Combine all checks
        all_passed = type_passed and vintage_passed and sums_match

        results = {
            "sum_by_type": sum_by_type,
            "sum_by_vintage": sum_by_vintage,
            "sums_match": sums_match,
            "difference": difference,
        }

        if all_passed:
            logger.info(
                f"Marginal check passed: type sum {sum_by_type['component_sum']} "
                f"and vintage sum {sum_by_vintage['component_sum']} agree."
            )
        else:
            if sums_match:
                logger.warning(
                    f"Marginal check failed: component sums match {sum_by_type['component_sum']}, but differ from total {sum_by_type['total']}. Check dataset for errors."
                )
                # TODO return all_passed True here?
            else:
                logger.warning(
                    f"Marginal check failed: component sums don't match type_passed={type_passed}, "
                    f"vintage_passed={vintage_passed}, sums_match={sums_match}, "
                    f"difference={difference}"
                )
    except KeyError as err:
        logger.error(f"Error checking marginals: {err}")
        return False, {"error": str(err)}

    return all_passed, results


def process_census_dataframe(df, census_year):
    """Process a single census dataframe through the full pipeline."""

    logger.info("Harmonizing census %d", int(census_year))
    df = harmonize_vintage_labels(df, sep="-", model_start=1608)

    logger.info("Expanding census %d with missing types and vintages", int(census_year))
    df = add_missing_vintages_types(df)

    logger.info("Converting census %d data to nullable Int32.", int(census_year))
    convert_df_to_int(df, inplace=True)

    logger.info(
        "Filling census %d aggregate types by summing over subtypes",
        int(census_year),
    )
    df = calculate_missing_types(df)

    logger.info(
        "Extracting relevant subset for census %d",
        int(census_year),
    )
    df = standardize_census(df)
    return df


if __name__ == "__main__":
    # Set program-level rng seed
    auto_seed_from_config()

    # Example usage
    print(NP_RANDOM.normal(0, 1))

    replacements = config["census"]["type_map"]

    census_dw = import_census_dataset(replacements=replacements)

    overwritten = overwrite_census_dataset(census_dw)

    standardized_data = {
        year: process_census_dataframe(df, year) for year, df in overwritten.items()
    }

    for census_year in standardized_data:
        display(standardized_data[census_year])

    # save as temporary html
    dataset = pd.concat(standardized_data).sort_index()
    dataset = filter_relevant_types_vintages(
        dataset, meta_cols=["census_year", "vintage"]
    )
    dataset.to_html("./temp.html")

    # TODO: standardize the dataset AFTER the IPFN procedure, as we need 'all' dwellings to be balanced. Therefore, I need to treat the other_dwelling / other_attached dwelling for consistency.

# TODO Refactor this file as a module, and separate the functions into different .py files (i.e., separation of concerns). Check for DRY and code smells.

# FIXME change all references to atol and rtol to values from config file?

# TODO Reprendre à "redefine target cohorts" in dmfa_dataprep.ipynb, AND Check_sums - also have a look at round_consistent_sum, fix_marginals, and res_fix


# TODO compare to dwellings 1685-2021.csv ?

# TODO: look at def check_sums in the ipynb files; it seems to be the start of the IPFN procedure. Then, in interpolating the missing data, I discuss padding backwards. however, why not set zeros to known values, then interpolate linearly, but backwards?
# IMPORTANTLY : "The 'masks' saved here will be useful to identify: 1 - which census have interpolated data (which previously had nans), 2 - what data was known before the interpolation, to prevent the IPFN of modifying these values" I'll want to keep the Nans and relevant totals (as marginals) for the IPFN procedure. "
# TODO round_consistent_sum
# TODO fix_marginals
# TODO: look at the 'corrected' values below, and at everything with "_fix" suffix
# TODO: compare with the exported dataset to see if I get the same results

# TODO, CHECK these values e.g., in manual overwrites. there are expected issues (in check_sums) for:
"""
# 1961
# Fix 1961 total
# Sums over cohorts and types both lead to the same value for total, when all other sums agree; where are the missing 7_000?
csdw_c["1961"].loc[0, "total"] = 1191368.0

# 1986
# cohorts are close (~175), types is wrong by 1M because of the undefined single attached and apartment types. When summing over all available types, the results are fine.
# NOTE Using CHASS data breaks the individual type sums (by cohorts)
# csdw_c['1986'].loc[0,'total'] = 2_357_100
# csdw_c['1986'].loc[0,'single_detached'] = 1_032_600
# csdw_c['1986'].loc[0,'apartment>5'] = 116_120
# csdw_c['1986'].loc[0,'mobile'] = 16_950
# csdw_c['1986'].loc[0,'other_attached_dwelling'] = 1_191_440
# csdw_c['1986'].drop(2).iloc[1:,2].sum()

# 1991
# cohorts are fine, types are close (~390)
csdw_c["1991"].loc[0, "single_detached"] = 1_175_085  # CHASS data; found the mistake
csdw_c["1991"].loc[0, "apartment>5"] = 137_105  # CHASS data; found the mistake
csdw_c["1991"].loc[0, "mobile"] = 24_720  # CHASS data; found the mistake
csdw_c["1991"].loc[0, "apartments"] = (
    csdw_c["1991"].loc[0, combined_types["apartments"]].sum()
)
# csdw_c['1991'].drop(2).iloc[1:,2].sum()

# 1996
# cohorts are fine, types need fixing
csdw_c["1996"].loc[0, "apartment_duplex"] = 171_265  # CHASS data; found the mistake
csdw_c["1996"].loc[0, "apartments"] = (
    csdw_c["1996"].loc[0, combined_types["apartments"]].sum()
)
# csdw_c['1996'].drop(2).iloc[1:,2].sum()

"""
# TODO et, plus tard, "# Autres ajouts manuels, à réviser # FIXME"
