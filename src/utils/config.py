"""Configuration management utilities."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from omegaconf import DictConfig, OmegaConf


def load_config(config_path: str) -> DictConfig:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        OmegaConf configuration object
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    config = OmegaConf.load(config_path)
    return config


def save_config(config: DictConfig, save_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration object
        save_path: Path to save configuration
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    OmegaConf.save(config, save_path)


def merge_configs(base_config: DictConfig, override_config: Optional[DictConfig] = None) -> DictConfig:
    """
    Merge configuration objects.
    
    Args:
        base_config: Base configuration
        override_config: Override configuration
        
    Returns:
        Merged configuration
    """
    if override_config is None:
        return base_config
    
    return OmegaConf.merge(base_config, override_config)


def create_directories(config: DictConfig) -> None:
    """
    Create necessary directories from configuration.
    
    Args:
        config: Configuration object
    """
    paths = config.get("paths", {})
    
    directories = [
        paths.get("data_dir", "data"),
        paths.get("checkpoint_dir", "checkpoints"),
        paths.get("log_dir", "logs"),
        paths.get("output_dir", "outputs"),
        paths.get("assets_dir", "assets"),
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def get_config_value(config: DictConfig, key: str, default: Any = None) -> Any:
    """
    Get configuration value with default fallback.
    
    Args:
        config: Configuration object
        key: Configuration key (supports dot notation)
        default: Default value if key not found
        
    Returns:
        Configuration value or default
    """
    try:
        return OmegaConf.select(config, key)
    except:
        return default


def print_config(config: DictConfig) -> None:
    """
    Print configuration in a readable format.
    
    Args:
        config: Configuration object
    """
    print("Configuration:")
    print(OmegaConf.to_yaml(config))
