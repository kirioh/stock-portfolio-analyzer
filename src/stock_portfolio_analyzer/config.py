import os
import yaml
from .logger import get_logger

logger = get_logger(__name__)

def load_config(config_file: str = None) -> dict:
    """
    Loads the configuration from a YAML file or environment variables.
    """
    config = {}
    
    # Default config location if not specified
    if not config_file:
        # Look for config in current working directory
        config_file = os.path.join(os.getcwd(), "config.yaml")
        # If not found, try the package directory
        if not os.path.exists(config_file):
            current_dir = os.path.dirname(__file__)
            config_file = os.path.join(os.path.dirname(os.path.dirname(current_dir)), "config.yaml")
    
    # Load from YAML file if it exists
    if os.path.exists(config_file):
        try:
            with open(config_file, "r") as f:
                config = yaml.safe_load(f)
                logger.info(f"Loaded configuration from {config_file}")
        except Exception as e:
            logger.warning(f"Failed to load config from {config_file}: {e}")
    else:
        logger.warning(f"Config file not found at {config_file}, using defaults and environment variables")
    
    # Set defaults if not specified
    if "data_source" not in config:
        config["data_source"] = "yfinance"
    if "retry_attempts" not in config:
        config["retry_attempts"] = 3

    # Override with environment variables if present
    if "DATA_SOURCE" in os.environ:
        config["data_source"] = os.environ["DATA_SOURCE"]
    if "RETRY_ATTEMPTS" in os.environ:
        config["retry_attempts"] = int(os.environ["RETRY_ATTEMPTS"])
    if "CSV_FILE_PATH" in os.environ:
        config["csv_file_path"] = os.environ["CSV_FILE_PATH"]

    return config
