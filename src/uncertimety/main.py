import toml
from numpy.random import PCG64
import numpy as np
from uncertimety.logger import init_logger
from uncertimety.config_loader import load_config

# from config import load_config
# from data_import import import_data
# from cleaning import clean_data
# from modeling import run_model
# from figures import generate_figures

# Initialize logger
logger = init_logger()


def main():
    CONFIG_FILE = "config.toml"

    # Load config file
    try:
        config = load_config(CONFIG_FILE)
        logger.info("Configuration loaded.")

    except (FileNotFoundError, toml.TomlDecodeError):
        logger.error("Configuration failed to load.")

    # Control randomness for reproducibility; create a random number generator with the PCG64 bit generator and a fixed seed
    logger.info("Controlling randomness...")
    bit_generator = PCG64(seed=config["random"]["seed"])
    rng = np.random.Generator(bit_generator)

    # Creating constants
    try:
        logger.info("Importing constants...")
        HIST_START = config["time"]["historic_start"]
        HIST_END = config["time"]["historic_end"]
        FUT_START = HIST_END + 1
        FUT_END = config["time"]["future_end"]

        HISTORIC_TIME = list(range(HIST_START, HIST_END + 1))
        FUTURE_TIME = list(range(FUT_START, FUT_END + 1))
        MODEL_TIME = list(range(HIST_START, FUT_END + 1))

    except KeyError as err:
        logger.error(f"Missing key in config: {err}")
        raise

    # random temp stuff to remove errors
    print(rng.normal(0, 1))
    print(HISTORIC_TIME[0:5], FUTURE_TIME[0:5], MODEL_TIME[0:5])
    # try:
    #     raw_data = import_data(config)
    #     logger.info("Data imported.")
    # except Exception as err:
    #     logger.exception("Data import failed.")
    #     return

    """ TODO
    COLUMN_REPLACEMENTS = config["census"]["column_replacements"]

    # Use it
    cleaned = clean_col_names(raw_columns, COLUMN_REPLACEMENTS

    # try:
    #     cleaned_data = clean_data(raw_data)
    #     logger.info("Data cleaned.")
    # except Exception as err:
    #     logger.exception("Data cleaning failed.")
    #     return
    """
    # try:
    #     results = run_model(cleaned_data, config)
    #     logger.info("Model run completed.")
    # except Exception as err:
    #     logger.exception("Model execution failed.")
    #     return

    # try:
    #     generate_figures(results)
    #     logger.info("Figures generated.")
    # except Exception as err:
    #     logger.exception("Figure generation failed.")
    #     return


if __name__ == "__main__":
    logger.info("Launching Uncertimety run...")
    main()
    logger.info("Uncertimety run complete.")
