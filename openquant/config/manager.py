"""Centralized configuration management with schema validation."""
from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from pydantic import ValidationError

from .schemas import Config
from ..utils.logging import get_logger

LOGGER = get_logger(__name__)


class ConfigManager:
    """Centralized configuration manager with Pydantic validation.
    
    Usage:
        # Initialize with default config
        config = ConfigManager()
        
        # Or load from YAML
        config = ConfigManager.from_yaml("configs/production.yaml")
        
        # Access configuration
        dd_limit = config.get("risk_limits.dd_limit")
        
        # Update configuration
        config.set("risk_limits.dd_limit", 0.15)
        
        # Get section
        risk_config = config.get_section("risk_limits")
    """
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize configuration manager.
        
        Args:
            config: Pydantic Config object. If None, uses default values.
        """
        self._config = config or Config()
        self._config_path: Optional[Path] = None
    
    @classmethod
    def from_yaml(cls, path: str | Path) -> ConfigManager:
        """Load configuration from YAML file with validation.
        
        Args:
            path: Path to YAML configuration file
            
        Returns:
            ConfigManager instance
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValidationError: If config doesn't match schema
        """
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f) or {}
        
        try:
            config = Config(**config_dict)
            LOGGER.info(f"Configuration loaded and validated from {config_path}")
        except ValidationError as e:
            LOGGER.error(f"Configuration validation failed: {e}")
            raise
        
        manager = cls(config)
        manager._config_path = config_path
        return manager
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> ConfigManager:
        """Load configuration from dictionary with validation.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            ConfigManager instance
            
        Raises:
            ValidationError: If config doesn't match schema
        """
        config = Config(**config_dict)
        return cls(config)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation.
        
        Args:
            key: Configuration key in dot notation (e.g., "risk_limits.dd_limit")
            default: Default value if key not found
            
        Returns:
            Configuration value or default
            
        Example:
            >>> config.get("risk_limits.dd_limit")
            0.2
        """
        keys = key.split(".")
        value = self._config
        
        for k in keys:
            if hasattr(value, k):
                value = getattr(value, k)
            else:
                return default
        
        return value
    
    def get_section(self, section: str) -> Optional[Any]:
        """Get entire configuration section.
        
        Args:
            section: Section name (e.g., "risk_limits", "backtest")
            
        Returns:
            Configuration section object or None
            
        Example:
            >>> risk_config = config.get_section("risk_limits")
            >>> risk_config.dd_limit
            0.2
        """
        if hasattr(self._config, section):
            return getattr(self._config, section)
        return None
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value using dot notation.
        
        Args:
            key: Configuration key in dot notation
            value: New value
            
        Raises:
            ValueError: If key path is invalid
            ValidationError: If value doesn't match schema
            
        Example:
            >>> config.set("risk_limits.dd_limit", 0.15)
        """
        keys = key.split(".")
        if len(keys) == 1:
            setattr(self._config, keys[0], value)
        else:
            obj = self._config
            for k in keys[:-1]:
                if hasattr(obj, k):
                    obj = getattr(obj, k)
                else:
                    raise ValueError(f"Invalid configuration path: {key}")
            
            setattr(obj, keys[-1], value)
        
        LOGGER.debug(f"Configuration updated: {key} = {value}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary.
        
        Returns:
            Configuration dictionary
        """
        return self._config.dict()
    
    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file.
        
        Args:
            path: Output file path
        """
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
        
        LOGGER.info(f"Configuration saved to {output_path}")
    
    def validate(self) -> bool:
        """Validate current configuration against schema.
        
        Returns:
            True if valid
            
        Raises:
            ValidationError: If validation fails
        """
        Config(**self.to_dict())
        return True
    
    def reload(self) -> None:
        """Reload configuration from original file if available.
        
        Raises:
            RuntimeError: If no config file path is set
        """
        if self._config_path is None:
            raise RuntimeError("Cannot reload: no configuration file path set")
        
        with open(self._config_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f) or {}
        
        self._config = Config(**config_dict)
        LOGGER.info(f"Configuration reloaded from {self._config_path}")
    
    def merge(self, other: Dict[str, Any] | ConfigManager) -> None:
        """Merge another configuration into this one.
        
        Args:
            other: Dictionary or ConfigManager to merge
        """
        if isinstance(other, ConfigManager):
            other_dict = other.to_dict()
        else:
            other_dict = other
        
        current_dict = self.to_dict()
        merged = self._deep_merge(current_dict, other_dict)
        self._config = Config(**merged)
        LOGGER.info("Configuration merged")
    
    @staticmethod
    def _deep_merge(base: Dict, override: Dict) -> Dict:
        """Deep merge two dictionaries."""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = ConfigManager._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    @property
    def config(self) -> Config:
        """Get raw Pydantic config object."""
        return self._config


_global_config: Optional[ConfigManager] = None


def get_config() -> ConfigManager:
    """Get global configuration manager instance.
    
    Returns:
        Global ConfigManager instance
        
    Example:
        >>> from openquant.config.manager import get_config
        >>> config = get_config()
        >>> dd_limit = config.get("risk_limits.dd_limit")
    """
    global _global_config
    if _global_config is None:
        config_file = os.getenv("OPENQUANT_CONFIG", "configs/default.yaml")
        config_path = Path(config_file)
        
        if config_path.exists():
            _global_config = ConfigManager.from_yaml(config_path)
            LOGGER.info(f"Loaded global config from {config_path}")
        else:
            _global_config = ConfigManager()
            LOGGER.info("Using default configuration")
    
    return _global_config


def set_global_config(config: ConfigManager) -> None:
    """Set global configuration manager instance.
    
    Args:
        config: ConfigManager instance to set as global
    """
    global _global_config
    _global_config = config
    LOGGER.info("Global configuration updated")


def reset_global_config() -> None:
    """Reset global configuration to None."""
    global _global_config
    _global_config = None
    LOGGER.info("Global configuration reset")
