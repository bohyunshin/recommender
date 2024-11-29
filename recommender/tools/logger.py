# import logging
#
# def setup_logger(log_file):
#     # Create a logger object
#     logger = logging.getLogger("recommender")
#     logger.setLevel(logging.DEBUG)
#
#     # Create a file handler to log messages to a file
#     file_handler = logging.FileHandler(log_file, mode='w')  # Open file in write mode to overwrite on each run
#
#     # Set a logging format
#     formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
#     file_handler.setFormatter(formatter)
#
#     # Add the file handler to the logger
#     logger.addHandler(file_handler)
#
#     return logger

import logging
import logging.config

def setup_logger(log_file):
    logging_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'default': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            },
        },
        'handlers': {
            'file_handler': {
                'class': 'logging.FileHandler',
                'formatter': 'default',
                'filename': log_file,
                'mode': 'w',  # Open file in write mode to overwrite on each run
            },
        },
        'root': {
            'handlers': ['file_handler'],
            'level': 'DEBUG',
        },
    }

    logging.config.dictConfig(logging_config)