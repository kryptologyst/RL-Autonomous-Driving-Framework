"""Simple configuration management for RL Autonomous Driving project."""

from typing import Dict, Any, Optional


class SimpleConfig:
    """Simple configuration manager for the RL project."""
    
    def __init__(self):
        """Initialize configuration with defaults."""
        self._config = {
            "environment": {
                "name": "CartPole-v1",
                "continuous": False,
                "domain_randomize": False,
                "render_mode": None
            },
            "agent": {
                "algorithm": "PPO",
                "learning_rate": 3e-4,
                "gamma": 0.99,
                "epsilon": 0.1,
                "batch_size": 64,
                "buffer_size": 10000
            },
            "training": {
                "total_timesteps": 100000,
                "eval_freq": 5000,
                "save_freq": 25000,
                "log_freq": 1000,
                "n_eval_episodes": 5
            },
            "model": {
                "input_shape": [4],
                "action_dim": 2,
                "hidden_size": 64
            },
            "logging": {
                "use_tensorboard": True,
                "use_wandb": False,
                "log_dir": "./logs"
            },
            "device": "auto"
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'environment.name')
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
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'environment.name')
            value: Value to set
        """
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def __getitem__(self, key: str) -> Any:
        """Allow dict-like access."""
        return self.get(key)
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Allow dict-like setting."""
        self.set(key, value)
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists."""
        return self.get(key) is not None
