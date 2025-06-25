import re
import toml
import pandas as pd
from pathlib import Path
from typing import Optional, Dict
from uncertimety.logger import init_logger
from IPython.display import display  # FIXME only for temp test


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
        # covers the case of e.g., 1986-1, 1996-1, 1986 (1)
        # FIXME I actually might want to keep the one, depending on future use case in dataprep.py TODO: see dmfa_dataprep for this
        return vintage.split(sep)[0].strip()
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
