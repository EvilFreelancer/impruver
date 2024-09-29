import logging
from transformers import logging as t_logging


def get_logger(level=t_logging.log_levels['info']) -> logging.Logger:
    t_logging.set_verbosity(level)
    logger = t_logging.get_logger(__file__)
    logger.setLevel(level)
    return logger
