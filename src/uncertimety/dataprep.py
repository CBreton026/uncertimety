import re
import toml
import pandas as pd
from pathlib import Path
from typing import Optional, Dict
from uncertimety.logger import init_logger
from IPython.display import display  # FIXME only for temp test
from datetime import datetime
from dateutil import relativedelta

logger = init_logger()

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


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

    return clean_name


def normalize_vintage_label(vintage: str, sep="-") -> str:
    """Normalize a single vintage label."""
    replacements = {"or": "", "to": ""}
    vintage = clean_name(vintage, replacements=replacements, sep=sep)

    if "total" in vintage:
        return "total"
    elif "before" in vintage:
        return "<" + vintage.split(sep)[0].strip()
    elif "after" in vintage or ">" in vintage:
        return vintage.split(sep)[0].strip() + "+"
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
    # col_name = col_name.split("(")[0].strip().lower()
    # print(col_name)

    if normalized in ("census_year", "vintage"):
        return normalized
    elif normalized == "apartment":
        return replacements.get(normalized, "apartments")

    # FIXME assumes a single key is present, and replaces based on first match
    for key, replacement in replacements.items():
        if key in normalized:
            if key == "apartment":
                # "apartment" is in several names, and breaks this function - ignore it
                # FIXME - I could also simply remove it from the replacements dict?
                pass
            else:
                return replacement

    # default case, e.g., "total"
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
        msg = f"No census CSV files found in path: {pattern}"
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
        ]
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
    # TODO unit tests

    with open(toml_file, "r") as f:
        raw = toml.load(f)

    overwrites = {}
    for year, fields in raw.items():
        normalized_fields = {}
        for k, v in fields.items():
            # Fix type names like 'apartment_lt_5' → 'apartment<5'
            name = k.replace("_lt_", "<").replace("_gt_", ">")
            # Replace false with None
            if isinstance(v, list):
                normalized_fields[name] = [None if i == -1 else i for i in v]
            elif isinstance(v, dict):
                normalized_fields[name] = {
                    key: None if val == -1 else val for key, val in v.items()
                }
            else:
                normalized_fields[name] = None if v == -1 else v
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
            val = patch["value"]
            df.at[row, "vintage"] = val
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
        list[float]: [full_years_share, census_year_share]
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

    return [full_years_share / total, census_year_share / total]


def extract_year_from_token(token: str, symbol: str) -> int:
    """
    Extracts a year from a token like '<1920' or '1986+' by splitting on symbol.
    """
    # TODO check if applications in other functions in dataprep.py
    try:
        year_str = token.split(symbol)[0] if symbol == "+" else token.split(symbol)[1]
        return check_year(int(year_str))
    except Exception as err:
        raise ValueError(
            f"Could not extract year from '{token}' using symbol '{symbol}'"
        ) from err


def parse_single_vintage(
    label: str, census_year: int, sep: str = "-", model_start: int = 1608
) -> tuple[list[tuple[int, int]], list[float]]:
    # FIXME: rename model_start? default to zero or None?
    """
    Parses a single vintage label into [start, end] intervals and associated shares.

    Handles:
    - '<1920' → (model_start, 1920)
    - '1986+' → (1986, round_to_next_5(census_year))
    - '1991-1' or '1971-1981-1' → split vintage
    - 'total' → (model_start, round_to_next_5(census_year))

    Returns:
        tuple: (list of (start, end), list of shares)
    """
    label = str(label).strip()
    census_year = check_year(census_year)

    intervals = []
    shares = [1.0]

    # === Special cases first
    if label.lower() == "total":
        intervals = [
            (model_start, round_to_next_5(census_year))
        ]  # FIXME might need to be from model start to census year directly?
        return intervals, shares

    if label.startswith("<"):
        try:
            year = extract_year_from_token(label, "<")
            intervals = [(model_start, year)]
            return intervals, shares
        except ValueError as err:
            logger.error(err)
            return [], []

    if label.endswith("+"):
        try:
            year = extract_year_from_token(label, "+")
            intervals = [(year, round_to_next_5(census_year))]
            return intervals, shares
        except ValueError as err:
            logger.error(err)
            return [], []

    # === Incomplete census year (e.g., 1991-1, 1971-1981-1)
    if "1" in label.split(sep):
        parts = label.split(sep)[:-1]
        if len(parts) == 1:
            year = check_year(parts[0])
            intervals = [(year, round_to_next_5(year))]
        elif len(parts) == 2:
            start, end = map(check_year, parts)
            intervals = [
                (start, infer_last_full_year(end)),  # full years
                (end, round_to_next_5(end)),  # partial census year
            ]
            # FIXME why round to five? why not directly aim for final categories? jsut to keep details? or this is treated later?
            shares = get_vintage_shares(label, sep=sep)
        return intervals, shares

    # === Standard YYYY-YYYY case
    try:
        parts = label.split(sep)
        if len(parts) == 2:
            start, end = map(check_year, parts)
            intervals = [(start, end)]
        else:
            logger.warning(f"Unexpected label format: {label}")
    except Exception as err:
        logger.error(f"Failed to parse label '{label}': {err}")

    return intervals, shares


if __name__ == "__main__":
    replacements = {
        "total": "total",
        "movable": "mobile",
        "apartment": "apartments",
        "fewer": "apartment<5",
        "more": "apartment>5",
        "semi-": "semi_detached",
        "duplex": "apartment_duplex",
    }
    census_dw = import_census_dataset(replacements=replacements)
    display(census_dw["1991"])

    pd.concat(census_dw).sort_index().to_html("./temp.html")

    overwritten = overwrite_census_dataset(census_dw)
    display(overwritten["1991"])

    # TODO add checks to see that the overwrite is done properly, and that it results in totals that 'make sense' (horizontally and vertically)
