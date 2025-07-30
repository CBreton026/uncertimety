import random
import toml
import numpy as np
from pathlib import Path
from uncertimety.logger import init_logger

logger = init_logger()

# -- Constants and paths --
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
CONFIG = DATA_DIR / "config.toml"

# -- Global random generators --
PY_RANDOM = random.Random()
NP_RANDOM = np.random.default_rng()

def load_seed_from_config(config_path=CONFIG) -> int:
    if not config_path.exists():
        msg = f"No TOML file found at: {config_path}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    try:
        config = toml.load(config_path)
        seed = config["random"].get("seed")
        if seed is None:
            raise KeyError("Missing 'seed' key under [random] section.")
        logger.info(f"Loaded seed {seed} from config.")
        return seed
    except (FileNotFoundError, toml.TomlDecodeError, KeyError) as err:
        logger.error("Failed to load random seed from config.")
        raise err

def auto_seed_from_config():
    """
    Automatically seed RNGs from TOML config if available.
    """
    seed = load_seed_from_config()
    seed_all(seed)

def seed_all(seed: int):
    """
    Seeds all program-wide random generators for reproducibility.
    Call this early in the program (e.g. main.py).
    """
    PY_RANDOM.seed(seed)
    global NP_RANDOM
    NP_RANDOM = np.random.default_rng(seed)
    logger.info(f"Global RNGs seeded with seed={seed}")

