import os
import json
import logging.config


def configure_logging(configure_file_path="LogConfigure.json", default_level=logging.INFO, env_key="LOG_CFG"):
    path = configure_file_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        print("logging_config.json path exit")
        with open(path, "r") as f:
            config = json.load(f)
            logging.config.dictConfig(config)
    else:
        print("logging_config.json path not exit")
        logging.basicConfig(level=default_level)


def set_log_info():
    logging.info("Let's start to log some information.")

    logging.error("There are so many errors.")
    logging.info("go")


if __name__ == "__main__":
    configure_logging("logging_config.json")
    set_log_info()