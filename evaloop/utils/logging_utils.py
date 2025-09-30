"""Logging utilities for EvaLoop framework."""

import os
import logging
from pathlib import Path
from typing import Optional


def setup_logging(
    level: str = "INFO",
    log_dir: str = "logs",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """
    Set up logging configuration for EvaLoop.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR).
        log_dir: Directory to save log files.
        log_file: Specific log file name (defaults to evaloop.log).
        format_string: Custom format string for log messages.
        
    Returns:
        Configured logger instance.
    """
    # Create log directory
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Set log file name
    if log_file is None:
        log_file = "evaloop.log"
    
    log_path = Path(log_dir) / log_file
    
    # Set format string
    if format_string is None:
        format_string = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ],
        force=True  # Override any existing configuration
    )
    
    # Get logger
    logger = logging.getLogger("evaloop")
    logger.info(f"Logging initialized. Level: {level}, Log file: {log_path}")
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Logger name.
        
    Returns:
        Logger instance.
    """
    return logging.getLogger(f"evaloop.{name}")


def set_log_level(level: str):
    """
    Set the logging level for all EvaLoop loggers.
    
    Args:
        level: New logging level.
    """
    logging.getLogger("evaloop").setLevel(getattr(logging, level.upper()))
    
    # Update all existing loggers
    for logger_name in logging.Logger.manager.loggerDict:
        if logger_name.startswith("evaloop"):
            logging.getLogger(logger_name).setLevel(getattr(logging, level.upper()))
