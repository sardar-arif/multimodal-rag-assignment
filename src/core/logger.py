import logging
import sys
from src.core.config import settings

def setup_logger(name: str = "multimodal_rag") -> logging.Logger:
    """
    Configures a standardized logger for the project modules.
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    logger.setLevel(settings.log_level.upper())
    return logger

# Global default logger
logger = setup_logger()
