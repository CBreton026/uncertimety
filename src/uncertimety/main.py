import toml
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

    try:
        config = load_config(CONFIG_FILE)
        print(config["time"]["historic_start"])
        logger.info("Configuration loaded.")
    except (FileNotFoundError, toml.TomlDecodeError):
        logger.error("Configuration failed to load.")

    # try:
    #     raw_data = import_data(config)
    #     logger.info("Data imported.")
    # except Exception as err:
    #     logger.exception("Data import failed.")
    #     return

    # try:
    #     cleaned_data = clean_data(raw_data)
    #     logger.info("Data cleaned.")
    # except Exception as err:
    #     logger.exception("Data cleaning failed.")
    #     return

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
