import logging
import logging.config

def setup_logger(log_file: str):
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            },
        },
        "handlers": {
            "file_handler": {
                "class": "logging.FileHandler",
                "formatter": "default",
                "filename": log_file,
                "mode": "w",  # Open file in write mode to overwrite on each run
            },
        },
        "root": {
            "handlers": ["file_handler"],
            "level": "DEBUG",
        },
    }

    logging.config.dictConfig(logging_config)