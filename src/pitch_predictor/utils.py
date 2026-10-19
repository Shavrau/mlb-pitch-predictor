"""Utility functions for logging and Spark session management."""

import logging
from typing import Any


def get_logger(name: str = "pitch_predictor") -> logging.Logger:
    """Configures and returns a standard logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def get_spark_session() -> Any:
    """
    Get active SparkSession.
    
    Compatible with Databricks cluster runtime (where `spark` is predefined)
    or Databricks Connect.
    """
    try:
        from pyspark.sql import SparkSession
        return SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()
    except ImportError:
        raise ImportError(
            "pyspark is not installed. If running locally, install databricks-connect or pyspark."
        )
