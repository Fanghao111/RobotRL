"""
Configuration loader for Push Task RL project.
Loads settings from config.yaml file.

Config is cached globally and only loaded once per process.
"""

import os
import yaml
from typing import Any, Dict

# Default config path
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

# Global cache - loaded once per process
_config: Dict[str, Any] = None
_config_loaded_path: str = None

# Cached sub-configs for faster access
_env_config: Dict[str, Any] = None
_reward_config: Dict[str, Any] = None
_eval_config: Dict[str, Any] = None


def load_config(config_path: str = None, verbose: bool = True) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. Uses default if None.
        verbose: Whether to print load message.
    
    Returns cached config if already loaded from same path.
    """
    global _config, _config_loaded_path, _env_config, _reward_config, _eval_config
    
    if config_path is None:
        config_path = CONFIG_PATH
    
    # Return cached if already loaded from same path
    if _config is not None and _config_loaded_path == config_path:
        return _config
    
    # Load from file
    with open(config_path, 'r') as f:
        _config = yaml.safe_load(f)
    _config_loaded_path = config_path
    
    # Clear sub-config caches
    _env_config = None
    _reward_config = None
    _eval_config = None
    
    if verbose:
        print(f"[Config] Loaded from: {config_path}")
    
    return _config


def get_config() -> Dict[str, Any]:
    """Get the loaded configuration. Loads from default path if not loaded yet."""
    global _config
    if _config is None:
        load_config(verbose=True)
    return _config


def get_env_config() -> Dict[str, Any]:
    """Get environment configuration (cached)."""
    global _env_config
    if _env_config is None:
        _env_config = get_config()["env"]
    return _env_config


def get_reward_config() -> Dict[str, Any]:
    """Get reward configuration (cached)."""
    global _reward_config
    if _reward_config is None:
        _reward_config = get_config()["reward"]
    return _reward_config


def get_training_config(obs_type: str = "state") -> Dict[str, Any]:
    """Get training configuration for specified observation type."""
    config = get_config()["training"]
    return {
        **config[obs_type],
        "default_timesteps": config["default_timesteps"],
        "default_n_envs": config["default_n_envs"],
        "progress_update_freq": config["progress_update_freq"],
    }


def get_eval_config() -> Dict[str, Any]:
    """Get evaluation configuration (cached)."""
    global _eval_config
    if _eval_config is None:
        _eval_config = get_config()["evaluation"]
    return _eval_config


# Convenience accessors
def max_episode_steps() -> int:
    return get_env_config()["max_episode_steps"]


def success_threshold() -> float:
    return get_env_config()["success_threshold"]
