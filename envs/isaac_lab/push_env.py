"""
Isaac Lab implementation of the push task environment with SB3 support.

This module provides a wrapper that makes Isaac Lab environments compatible
with Stable Baselines 3 (SB3).
"""

import numpy as np
import torch
from typing import Dict, Optional, Tuple, Any
import gymnasium as gym
from gymnasium import spaces

from ..base import BasePushEnv, BasePushEnvConfig


def check_isaac_lab_available() -> bool:
    """Check if Isaac Lab is available."""
    try:
        import omni.isaac.lab
        return True
    except ImportError:
        return False


class IsaacLabPushEnv(BasePushEnv, gym.Env):
    """Isaac Lab implementation of push task with SB3 compatibility.

    This wrapper converts Isaac Lab's vectorized environment interface
    to work seamlessly with Stable Baselines 3.

    Key features:
    - GPU-accelerated parallel simulation
    - Automatic batching/unbatching for SB3 compatibility
    - Supports both single-env and vec-env modes

    Note: Isaac Lab requires NVIDIA Isaac Sim to be installed.
    """

    BACKEND_NAME = "isaac_lab"
    SUPPORTS_GPU_PARALLEL = True

    def __init__(
        self,
        cfg: BasePushEnvConfig,
        render_mode: Optional[str] = None,
        obs_type: str = "state",
        num_envs: int = 1,
        device: str = "cuda",
    ):
        """Initialize Isaac Lab push environment.

        Args:
            cfg: Environment configuration.
            render_mode: "human" for GUI, "rgb_array" for rendering, None for headless.
            obs_type: "state" for 19D vector (image not yet supported in Isaac Lab).
            num_envs: Number of parallel environments.
            device: "cuda" for GPU or "cpu".
        """
        super().__init__(cfg, render_mode, obs_type, num_envs, device)

        if obs_type == "image":
            raise NotImplementedError("Image observations not yet supported in Isaac Lab backend")

        # Lazy import to avoid issues when Isaac Lab is not available
        if not check_isaac_lab_available():
            raise ImportError(
                "Isaac Lab is not available. Please install Isaac Sim and Isaac Lab. "
                "See: https://isaac-sim.github.io/IsaacLab/main/source/setup/installation.html"
            )

        from omni.isaac.lab.envs import DirectRLEnv
        from .isaac_push_env import IsaacLabPushEnvInternal, create_isaac_lab_cfg

        # Create Isaac Lab config from BasePushEnvConfig
        isaac_cfg = create_isaac_lab_cfg(cfg, num_envs, render_mode == "human")

        # Create internal Isaac Lab environment
        self._isaac_env: DirectRLEnv = IsaacLabPushEnvInternal(isaac_cfg, render_mode)

        # Define spaces
        self._action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(3,),
            dtype=np.float32
        )
        self._observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(19,),
            dtype=np.float32
        )

        # For SB3 compatibility (single-env interface)
        self._is_vec_env = num_envs > 1

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment.

        For single-env mode, returns unbatched observation.
        """
        if seed is not None:
            torch.manual_seed(seed)

        obs_dict, info = self._isaac_env.reset()

        # Extract policy observation
        obs = obs_dict["policy"]

        if not self._is_vec_env:
            # Single env mode - unbatch
            obs = obs[0].cpu().numpy()
        else:
            obs = obs.cpu().numpy()

        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one environment step.

        For single-env mode, handles batching/unbatching automatically.
        """
        # Convert to tensor and batch if needed
        if not self._is_vec_env:
            action = np.expand_dims(action, 0)

        action_tensor = torch.from_numpy(action).float().to(self.device)

        # Step the environment
        obs_dict, reward, terminated, truncated, info = self._isaac_env.step(action_tensor)

        # Extract observations
        obs = obs_dict["policy"]

        if not self._is_vec_env:
            # Single env mode - unbatch
            obs = obs[0].cpu().numpy()
            reward = reward[0].item()
            terminated = terminated[0].item()
            truncated = truncated[0].item()
        else:
            obs = obs.cpu().numpy()
            reward = reward.cpu().numpy()
            terminated = terminated.cpu().numpy()
            truncated = truncated.cpu().numpy()

        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        """Render the environment."""
        if self.render_mode == "rgb_array":
            # TODO: Implement camera rendering for Isaac Lab
            return None
        return None

    def close(self):
        """Close the environment."""
        self._isaac_env.close()

    # Methods for SB3 VecEnv compatibility
    def get_attr(self, attr_name: str, indices=None):
        """Get attribute from environment."""
        return getattr(self, attr_name)

    def set_attr(self, attr_name: str, value, indices=None):
        """Set attribute on environment."""
        setattr(self, attr_name, value)

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs):
        """Call method on environment."""
        method = getattr(self, method_name)
        return method(*method_args, **method_kwargs)


class IsaacLabVecEnvWrapper(gym.vector.VectorEnv):
    """VecEnv wrapper for Isaac Lab environments.

    This wrapper provides full VecEnv compatibility for Isaac Lab,
    allowing direct use with SB3's SubprocVecEnv replacement.

    Usage:
        env = IsaacLabVecEnvWrapper(cfg, num_envs=1024, device="cuda")
        model = PPO("MlpPolicy", env)
        model.learn(total_timesteps=1000000)
    """

    def __init__(
        self,
        cfg: BasePushEnvConfig,
        num_envs: int = 1024,
        device: str = "cuda",
        render_mode: Optional[str] = None,
    ):
        """Initialize the VecEnv wrapper.

        Args:
            cfg: Environment configuration.
            num_envs: Number of parallel environments.
            device: "cuda" for GPU or "cpu".
            render_mode: Render mode (only affects first env).
        """
        self.cfg = cfg
        self.device = device
        self._num_envs = num_envs

        # Create Isaac Lab environment
        if not check_isaac_lab_available():
            raise ImportError("Isaac Lab is not available")

        from .isaac_push_env import IsaacLabPushEnvInternal, create_isaac_lab_cfg

        isaac_cfg = create_isaac_lab_cfg(cfg, num_envs, render_mode == "human")
        self._isaac_env = IsaacLabPushEnvInternal(isaac_cfg, render_mode)

        # Define spaces
        single_obs_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(19,),
            dtype=np.float32
        )
        single_action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(3,),
            dtype=np.float32
        )

        super().__init__(num_envs, single_obs_space, single_action_space)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Reset all environments."""
        if seed is not None:
            torch.manual_seed(seed)

        obs_dict, info = self._isaac_env.reset()
        obs = obs_dict["policy"].cpu().numpy()

        return obs, info

    def step(self, actions: np.ndarray):
        """Step all environments."""
        action_tensor = torch.from_numpy(actions).float().to(self.device)

        obs_dict, rewards, terminated, truncated, info = self._isaac_env.step(action_tensor)

        obs = obs_dict["policy"].cpu().numpy()
        rewards = rewards.cpu().numpy()
        terminated = terminated.cpu().numpy()
        truncated = truncated.cpu().numpy()

        return obs, rewards, terminated, truncated, info

    def close(self):
        """Close the environment."""
        self._isaac_env.close()

    def render(self):
        """Render (not implemented for vec env)."""
        return None
