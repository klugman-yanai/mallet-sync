# src/logger.py

"""A custom logger. Functional, but still in development, not all features are implemented."""

import logging

from typing import cast

from standard_logger import LoggerConfig, StandardLogger, setup_logging

log_config = LoggerConfig(
    app_name='Multiple Mallet Recorder',
    app_author='Kardome',
    log_file_path=False,
)


def get_logger(name: str) -> StandardLogger:
    """Gets a logger instance and casts it for type checking custom methods."""
    # Casting is necessary for static type checkers (like Pyright/Mypy)
    # to recognize the custom methods (panel, rule, progress) on the
    # logger instance returned by logging.getLogger().
    return cast(StandardLogger, logging.getLogger(name))


logger = get_logger(__name__)

setup_logging(log_config)
