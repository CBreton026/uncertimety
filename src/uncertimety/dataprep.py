import re
from typing import Optional, Dict


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

    if "before" in vintage:
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

    if normalized in ("annee", "vintage"):
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


def clean_col_names(col_names, replacements):
    return [normalize_column_name(name, replacements) for name in col_names]
