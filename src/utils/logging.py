# src/utils/logging.py
"""Logging utilities for the CHF project."""

import logging
import logging.config
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import yaml
import colorlog


def setup_logging(config_path: Optional[Path] = None, 
                  log_level: Optional[str] = None,
                  log_dir: Optional[Path] = None) -> None:
    """
    Setup logging configuration for the project.
    
    Args:
        config_path: Path to logging config file
        log_level: Override log level
        log_dir: Directory for log files
    """
    # Default paths
    if config_path is None:
        config_path = Path("configs/logging_config.yaml")
    if log_dir is None:
        log_dir = Path("logs")
    
    # Create log directory
    log_dir.mkdir(exist_ok=True)
    
    # Load configuration
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Update file paths with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for handler_name, handler_config in config['logging']['handlers'].items():
            if 'filename' in handler_config:
                filename = handler_config['filename']
                if '{timestamp}' in filename:
                    handler_config['filename'] = filename.format(timestamp=timestamp)
                # Ensure full path
                handler_config['filename'] = str(log_dir / Path(handler_config['filename']).name)
        
        # Apply configuration
        logging.config.dictConfig(config['logging'])
    else:
        # Fallback configuration
        setup_basic_logging(log_level or 'INFO')
    
    # Override log level if specified
    if log_level:
        logging.getLogger().setLevel(getattr(logging, log_level.upper()))
    
    # Log startup information
    logger = logging.getLogger(__name__)
    logger.info("="*60)
    logger.info("CHF Project Logging Initialized")
    logger.info(f"Log Level: {logging.getLogger().level}")
    logger.info(f"Log Directory: {log_dir}")
    logger.info("="*60)


def setup_basic_logging(log_level: str = 'INFO') -> None:
    """Setup basic logging configuration as fallback."""
    # Create color formatter
    formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        handlers=[console_handler]
    )


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance with optional level override.
    
    Args:
        name: Logger name
        level: Optional log level override
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    
    if level:
        logger.setLevel(getattr(logging, level.upper()))
    
    return logger


class LoggerMixin:
    """Mixin class to add logging capabilities to any class."""
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for the class."""
        if not hasattr(self, '_logger'):
            self._logger = get_logger(self.__class__.__module__ + '.' + self.__class__.__name__)
        return self._logger


def log_function_call(func):
    """Decorator to log function calls and execution time."""
    import functools
    import time
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        
        # Log function call
        logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        
        # Execute function
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.debug(f"{func.__name__} completed in {execution_time:.3f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.3f}s: {str(e)}")
            raise
    
    return wrapper


def log_model_info(model: Any, logger: logging.Logger) -> None:
    """Log model architecture and parameter information."""
    logger.info("Model Information:")
    logger.info("-" * 40)
    
    # Try to get model summary
    if hasattr(model, 'summary'):
        logger.info(model.summary())
    elif hasattr(model, '__str__'):
        logger.info(str(model))
    
    # Count parameters
    if hasattr(model, 'parameters'):
        # PyTorch model
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
    elif hasattr(model, 'get_params'):
        # Scikit-learn model
        params = model.get_params()
        logger.info(f"Model parameters: {params}")
    
    logger.info("-" * 40)


def create_experiment_logger(experiment_name: str, 
                           config: Dict[str, Any]) -> logging.Logger:
    """
    Create a logger for a specific experiment.
    
    Args:
        experiment_name: Name of the experiment
        config: Experiment configuration
        
    Returns:
        Logger instance
    """
    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path("logs/experiments") / f"{experiment_name}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create file handler for experiment
    logger = get_logger(f"experiment.{experiment_name}")
    file_handler = logging.FileHandler(exp_dir / "experiment.log")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)
    
    # Log experiment configuration
    logger.info("="*60)
    logger.info(f"Experiment: {experiment_name}")
    logger.info(f"Timestamp: {timestamp}")
    logger.info("Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    logger.info("="*60)
    
    return logger