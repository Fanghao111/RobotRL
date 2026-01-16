"""
Internal Isaac Lab environment implementation.

This module contains the actual Isaac Lab DirectRLEnv implementation.
It is separated from the SB3 wrapper for cleaner architecture.
"""

from __future__ import annotations

import math
import torch
from typing import Dict, Tuple, Optional

from ..base import BasePushEnvConfig


def create_isaac_lab_cfg(cfg: BasePushEnvConfig, num_envs: int = 1024, headless: bool = True):
    """Create Isaac Lab PushEnvCfg from BasePushEnvConfig.

    Args:
        cfg: Base environment configuration.
        num_envs: Number of parallel environments.
        headless: Whether to run in headless mode.

    Returns:
        PushEnvCfg for Isaac Lab.
    """
    # Import here to avoid issues when Isaac Lab is not available
    from omni.isaac.lab.envs import DirectRLEnvCfg
    from omni.isaac.lab.sim import SimulationCfg, PhysxCfg
    from omni.isaac.lab.utils import configclass
    import omni.isaac.lab.sim as sim_utils
    from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
    from omni.isaac.lab.scene import InteractiveSceneCfg
    from omni.isaac.lab_assets.franka import FRANKA_PANDA_CFG

    @configclass
    class PushSceneCfg(InteractiveSceneCfg):
        """Scene configuration."""
        ground = AssetBaseCfg(
            prim_path="/World/ground",
            spawn=sim_utils.GroundPlaneCfg(),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        )

        table = AssetBaseCfg(
            prim_path="/World/table",
            spawn=sim_utils.CuboidCfg(
                size=(1.5, 1.0, 0.05),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.5, 0.4)),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.6, 0.0, -0.025)),
        )

        robot: ArticulationCfg = FRANKA_PANDA_CFG.replace(
            prim_path="/World/envs/env_.*/Robot",
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.0),
                joint_pos={
                    "panda_joint1": 0.0,
                    "panda_joint2": -0.569,
                    "panda_joint3": 0.0,
                    "panda_joint4": -2.810,
                    "panda_joint5": 0.0,
                    "panda_joint6": 2.241,
                    "panda_joint7": 0.741,
                    "panda_finger_joint1": 0.04,
                    "panda_finger_joint2": 0.04,
                },
            ),
        )

        # Object size from config
        obj_size = (
            cfg.object_half_extents[0] * 2,
            cfg.object_half_extents[1] * 2,
            cfg.object_half_extents[2] * 2,
        )

        object: RigidObjectCfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Object",
            spawn=sim_utils.CuboidCfg(
                size=obj_size,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.0, 0.1)),
        )

        target_size = (
            cfg.target_half_extents[0] * 2,
            cfg.target_half_extents[1] * 2,
            cfg.target_half_extents[2] * 2,
        )

        target: AssetBaseCfg = AssetBaseCfg(
            prim_path="/World/envs/env_.*/Target",
            spawn=sim_utils.CuboidCfg(
                size=target_size,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 1.0, 0.0),
                    opacity=0.5,
                ),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.8, 0.0, 0.001)),
        )

        dome_light = AssetBaseCfg(
            prim_path="/World/DomeLight",
            spawn=sim_utils.DomeLightCfg(intensity=1500.0, color=(0.9, 0.9, 0.9)),
        )

    @configclass
    class ActionsCfg:
        action_scale: float = 0.1
        fixed_ee_z: float = cfg.fixed_ee_z

    @configclass
    class RewardsCfg:
        position_progress_coef: float = cfg.position_progress_coef
        orientation_progress_coef: float = cfg.orientation_progress_coef
        coupling_coef: float = cfg.coupling_coef
        alignment_coef: float = cfg.alignment_coef
        ee_approach_coef: float = cfg.ee_approach_coef
        contact_threshold: float = cfg.contact_threshold
        contact_reward: float = cfg.contact_reward
        success_bonus: float = cfg.success_bonus
        step_penalty: float = cfg.step_penalty

    @configclass
    class TerminationsCfg:
        time_out: bool = True
        success: bool = True
        position_threshold: float = cfg.success_threshold
        orientation_threshold: float = cfg.orientation_threshold

    @configclass
    class RandomizationCfg:
        object_x_range: tuple = cfg.object_x_range
        object_y_range: tuple = cfg.object_y_range
        target_x_range: tuple = cfg.target_x_range
        target_y_range: tuple = cfg.target_y_range
        target_yaw_range: tuple = (-math.pi, math.pi)

    @configclass
    class PushEnvCfg(DirectRLEnvCfg):
        sim: SimulationCfg = SimulationCfg(
            dt=1.0 / 120.0,
            render_interval=4,
            physx=PhysxCfg(
                bounce_threshold_velocity=0.2,
                gpu_found_lost_aggregate_pairs_capacity=1024 * 1024 * 4,
                gpu_total_aggregate_pairs_capacity=16 * 1024,
            ),
        )

        scene: PushSceneCfg = PushSceneCfg(num_envs=num_envs, env_spacing=2.5)
        decimation: int = 12
        episode_length_s: float = cfg.max_episode_steps * (1.0 / 10.0)  # 10Hz control

        action_space: int = 3
        observation_space: int = 19
        state_space: int = 0

        actions: ActionsCfg = ActionsCfg()
        rewards: RewardsCfg = RewardsCfg()
        terminations: TerminationsCfg = TerminationsCfg()
        randomization: RandomizationCfg = RandomizationCfg()

        def __post_init__(self):
            self.max_episode_length = int(self.episode_length_s / (self.sim.dt * self.decimation))
            self.viewer.eye = (2.0, 2.0, 2.0)
            self.viewer.lookat = (0.5, 0.0, 0.0)

    return PushEnvCfg()


class IsaacLabPushEnvInternal:
    """Internal Isaac Lab environment implementation.

    This class implements the actual environment logic using Isaac Lab's
    DirectRLEnv interface.

    Note: This is a placeholder. For full implementation, copy the logic
    from the existing isaac_lab_push/envs/push_env.py and adapt it.
    """

    def __init__(self, cfg, render_mode: Optional[str] = None):
        """Initialize the internal environment.

        Args:
            cfg: PushEnvCfg configuration.
            render_mode: Render mode.
        """
        # Import Isaac Lab components
        from omni.isaac.lab.envs import DirectRLEnv
        from omni.isaac.lab.assets import Articulation, RigidObject
        from omni.isaac.lab.utils.math import euler_xyz_from_quat, quat_from_euler_xyz

        self.cfg = cfg
        self.render_mode = render_mode

        # Initialize base environment
        # Note: Full implementation would inherit from DirectRLEnv
        # For now, we'll create a minimal wrapper

        # This is a simplified placeholder - the actual implementation
        # should use the full DirectRLEnv machinery
        self._initialized = False

    def reset(self) -> Tuple[Dict[str, torch.Tensor], Dict]:
        """Reset the environment."""
        raise NotImplementedError(
            "Full Isaac Lab implementation requires Isaac Sim. "
            "Please use the PyBullet backend for CPU-based training, "
            "or ensure Isaac Sim is properly installed."
        )

    def step(self, actions: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """Step the environment."""
        raise NotImplementedError(
            "Full Isaac Lab implementation requires Isaac Sim. "
            "Please use the PyBullet backend for CPU-based training."
        )

    def close(self):
        """Close the environment."""
        pass
