# src/utils/logging_config.py
import logging

def setup_logging():
    """Configure logging for the application."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    return logger