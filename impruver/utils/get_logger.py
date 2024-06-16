import logging


def get_logger(level=logging.getLevelName(logging.INFO)) -> logging.Logger:
    logger = logging.getLogger(__file__)
    logger.setLevel(level)
    return logger
