# src/utils/config.py
"""Configuration management utilities for the CHF project."""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import copy
from dataclasses import dataclass, field
import logging


logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Data class for model configuration."""
    name: str
    init_params: Dict[str, Any] = field(default_factory=dict)
    training_params: Dict[str, Any] = field(default_factory=dict)
    tuning_params: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'init_params': self.init_params,
            'training_params': self.training_params,
            'tuning_params': self.tuning_params
        }


def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If file format is unsupported
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    logger.info(f"Loading configuration from: {config_path}")
    
    # Load based on file extension
    if config_path.suffix in ['.yaml', '.yml']:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    elif config_path.suffix == '.json':
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        raise ValueError(f"Unsupported config format: {config_path.suffix}")
    
    # Validate configuration
    validate_config(config)
    
    return config


def save_config(config: Dict[str, Any], config_path: Path) -> None:
    """
    Save configuration to file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    if config_path.suffix in ['.yaml', '.yml']:
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
    elif config_path.suffix == '.json':
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    else:
        raise ValueError(f"Unsupported config format: {config_path.suffix}")
    
    logger.info(f"Configuration saved to: {config_path}")


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge multiple configuration dictionaries.
    
    Args:
        *configs: Configuration dictionaries to merge
        
    Returns:
        Merged configuration
    """
    def deep_merge(base: Dict, update: Dict) -> Dict:
        """Recursively merge dictionaries."""
        result = copy.deepcopy(base)
        
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
        
        return result
    
    # Start with empty dict
    merged = {}
    
    # Merge all configs
    for config in configs:
        merged = deep_merge(merged, config)
    
    return merged


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration structure and values.
    
    Args:
        config: Configuration dictionary
        
    Raises:
        ValueError: If configuration is invalid
    """
    # Check for required sections
    required_sections = []  # Add required sections as needed
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")
    
    # Validate model configurations if present
    if 'models' in config or any(key in ['xgboost', 'lightgbm', 'neural_network', 'svm', 'pinn'] for key in config):
        validate_model_configs(config)
    
    logger.debug("Configuration validation passed")


def validate_model_configs(config: Dict[str, Any]) -> None:
    """Validate model-specific configurations."""
    # Known model types
    known_models = ['xgboost', 'lightgbm', 'neural_network', 'svm', 'pinn']
    
    for model_name in known_models:
        if model_name in config:
            model_config = config[model_name]
            
            # Check for common required fields
            if 'init_params' not in model_config:
                logger.warning(f"Model {model_name} missing 'init_params'")
            
            # Model-specific validation
            if model_name == 'neural_network':
                if 'architecture' not in model_config:
                    logger.warning(f"Neural network missing 'architecture' config")
            elif model_name == 'pinn':
                if 'physics' not in model_config:
                    logger.warning(f"PINN missing 'physics' config")


def get_model_config(config: Dict[str, Any], model_name: str) -> ModelConfig:
    """
    Extract model configuration.
    
    Args:
        config: Full configuration dictionary
        model_name: Name of the model
        
    Returns:
        ModelConfig instance
    """
    if model_name not in config:
        raise ValueError(f"Model {model_name} not found in configuration")
    
    model_dict = config[model_name]
    
    # Extract parameters
    init_params = model_dict.get('init_params', {})
    training_params = model_dict.get('training', {})
    tuning_params = model_dict.get('tuning', None)
    
    # Handle old 'epochs' key
    if 'epochs' in model_dict:
        training_params['epochs'] = model_dict['epochs']
    
    return ModelConfig(
        name=model_name,
        init_params=init_params,
        training_params=training_params,
        tuning_params=tuning_params
    )


def expand_param_grid(params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Expand parameter grid for hyperparameter tuning.
    
    Args:
        params: Dictionary with parameter names and values (lists for tuning)
        
    Returns:
        List of parameter combinations
    """
    from itertools import product
    
    # Separate fixed and tunable parameters
    fixed_params = {}
    tunable_params = {}
    
    for key, value in params.items():
        if isinstance(value, list):
            tunable_params[key] = value
        else:
            fixed_params[key] = value
    
    # Generate combinations
    if not tunable_params:
        return [params]
    
    param_combinations = []
    keys = list(tunable_params.keys())
    values = list(tunable_params.values())
    
    for combination in product(*values):
        param_dict = fixed_params.copy()
        for key, value in zip(keys, combination):
            param_dict[key] = value
        param_combinations.append(param_dict)
    
    return param_combinations


def update_config_with_args(config: Dict[str, Any], args: Any) -> Dict[str, Any]:
    """
    Update configuration with command-line arguments.
    
    Args:
        config: Base configuration
        args: Command-line arguments (argparse.Namespace)
        
    Returns:
        Updated configuration
    """
    updated_config = copy.deepcopy(config)
    
    # Convert args to dict
    args_dict = vars(args) if hasattr(args, '__dict__') else args
    
    # Update configuration with non-None arguments
    for key, value in args_dict.items():
        if value is not None:
            # Handle nested keys (e.g., 'model.epochs' -> config['model']['epochs'])
            if '.' in key:
                keys = key.split('.')
                current = updated_config
                for k in keys[:-1]:
                    if k not in current:
                        current[k] = {}
                    current = current[k]
                current[keys[-1]] = value
            else:
                updated_config[key] = value
    
    return updated_config


class ConfigManager:
    """Centralized configuration manager."""
    
    def __init__(self, base_config_dir: Path = Path("configs")):
        """Initialize configuration manager."""
        self.base_config_dir = base_config_dir
        self.configs = {}
        self._load_all_configs()
    
    def _load_all_configs(self) -> None:
        """Load all configuration files."""
        config_files = [
            "model_configs.yaml",
            "physics_configs.yaml",
            "logging_config.yaml"
        ]
        
        for config_file in config_files:
            config_path = self.base_config_dir / config_file
            if config_path.exists():
                config_name = config_path.stem
                self.configs[config_name] = load_config(config_path)
                logger.debug(f"Loaded {config_name} configuration")
    
    def get(self, config_name: str, key: Optional[str] = None) -> Any:
        """Get configuration or specific key."""
        if config_name not in self.configs:
            raise ValueError(f"Configuration '{config_name}' not found")
        
        config = self.configs[config_name]
        
        if key is None:
            return config
        
        # Handle nested keys
        keys = key.split('.')
        current = config
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                raise KeyError(f"Key '{key}' not found in {config_name}")
        
        return current
    
    def update(self, config_name: str, updates: Dict[str, Any]) -> None:
        """Update configuration."""
        if config_name not in self.configs:
            self.configs[config_name] = {}
        
        self.configs[config_name] = merge_configs(
            self.configs[config_name], 
            updates
        )
    
    def save_all(self, output_dir: Path) -> None:
        """Save all configurations to directory."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for config_name, config in self.configs.items():
            config_path = output_dir / f"{config_name}.yaml"
            save_config(config, config_path)