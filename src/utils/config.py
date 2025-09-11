"""Configuration management utilities."""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
import platform
import torch


class Config:
    """Configuration manager for the sign language detection framework."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration.
        
        Args:
            config_path: Path to configuration file. If None, uses default config.yaml
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config.yaml"
        
        self.config_path = Path(config_path)
        self._config = self._load_config()
        self._setup_device()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML configuration: {e}")
    
    def _setup_device(self):
        """Setup device based on system and config."""
        device_config = self.get('system.device', 'auto')
        
        if device_config == 'auto':
            if torch.cuda.is_available():
                self._device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self._device = torch.device('mps')  # Apple Silicon
            else:
                self._device = torch.device('cpu')
        else:
            self._device = torch.device(device_config)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation.
        
        Args:
            key: Configuration key in dot notation (e.g., 'training.batch_size')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any):
        """Set configuration value using dot notation.
        
        Args:
            key: Configuration key in dot notation
            value: Value to set
        """
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    @property
    def device(self) -> torch.device:
        """Get the configured device."""
        return self._device
    
    @property
    def is_cuda_available(self) -> bool:
        """Check if CUDA is available."""
        return torch.cuda.is_available()
    
    @property
    def is_mps_available(self) -> bool:
        """Check if MPS (Apple Silicon) is available."""
        return hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    
    @property
    def platform_info(self) -> Dict[str, str]:
        """Get platform information."""
        return {
            'system': platform.system(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'torch_version': torch.__version__,
        }
    
    def save(self, path: Optional[str] = None):
        """Save current configuration to file.
        
        Args:
            path: Path to save configuration. If None, overwrites original file.
        """
        save_path = Path(path) if path else self.config_path
        
        with open(save_path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False, indent=2)
    
    def __getitem__(self, key: str):
        """Allow dict-like access to configuration."""
        return self.get(key)
    
    def __setitem__(self, key: str, value: Any):
        """Allow dict-like assignment to configuration."""
        self.set(key, value)


# Global configuration instance
config = Config()
