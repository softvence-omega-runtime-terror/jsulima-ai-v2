import logging
import logging.handlers
import sys
from pathlib import Path
from app.core.config import settings


def setup_logging():
    Path(settings.log_directory).mkdir(exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=settings.log_level,
        format='%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            # Console handler
            logging.StreamHandler(sys.stdout),
            # File handler with rotation
            logging.handlers.RotatingFileHandler(
                Path(settings.log_directory) / Path(settings.log_file),
                maxBytes=10*1024*1024,  # 10MB
                backupCount=3,
                encoding='utf-8'
            )
        ]
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for the given name.
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        Logger instance
    """
    return logging.getLogger(name)