import toml
from pathlib import Path
from typing import Any
from uncertimety.logger import init_logger

logger = init_logger()

# Define base paths
BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = BASE_DIR / "data"


def load_config(filename: str, data_dir: Path = DEFAULT_DATA_DIR) -> dict[str, Any]:
    """
    Load a TOML configuration file from the given data directory.

    Args:
        filename (str): Name of the TOML file to load (with extension).
        data_dir (Path): Directory where the TOML file is located.

    Returns:
        dict[str, Any]: Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the config file is not found.
        toml.TomlDecodeError: If the config file cannot be parsed.
    """
    infile = data_dir / filename
    try:
        with open(infile, "r") as f:
            config = toml.load(f)
            logger.info(f"Successfully loaded config from {infile}")
            return config
    except FileNotFoundError:
        logger.error(f"Config file not found: {infile}")
        raise
    except toml.TomlDecodeError as err:
        logger.error(f"Failed to parse TOML file {infile}: {err}")
        raise
