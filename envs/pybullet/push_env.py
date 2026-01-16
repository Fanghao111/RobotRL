"""
PyBullet implementation of the push task environment.

This module implements the push task using PyBullet physics engine.
It supports both state-based and image-based observations.
"""

import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np
import math
import time
from typing import Dict, Optional, Tuple, Any

from ..base import BasePushEnv, BasePushEnvConfig


class PyBulletPushEnv(BasePushEnv, gym.Env):
    """PyBullet implementation of push task environment.

    Uses Kuka iiwa 7-DOF robot arm to push a rectangular object
    to a target position and orientation.

    Physics: 120Hz simulation, 10Hz control (12 physics steps per action)
    """

    BACKEND_NAME = "pybullet"
    SUPPORTS_GPU_PARALLEL = False

    def __init__(
        self,
        cfg: BasePushEnvConfig,
        render_mode: Optional[str] = None,
        obs_type: str = "state",
        num_envs: int = 1,
        device: str = "cpu",
    ):
        """Initialize PyBullet push environment.

        Args:
            cfg: Environment configuration.
            render_mode: "human" for GUI, "rgb_array" for rendering, None for headless.
            obs_type: "state" for 19D vector, "image" for 84x84 RGB.
            num_envs: Ignored for PyBullet (always 1).
            device: Ignored for PyBullet (always CPU).
        """
        super().__init__(cfg, render_mode, obs_type, num_envs=1, device="cpu")

        # Connect to PyBullet
        if render_mode == "human":
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Define spaces
        self._action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)

        self.img_width = 84
        self.img_height = 84

        if self.obs_type == "image":
            self._observation_space = spaces.Box(
                low=0, high=255,
                shape=(self.img_height, self.img_width, 3),
                dtype=np.uint8
            )
        else:
            # 19D state observation
            self._observation_space = spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(19,),
                dtype=np.float32
            )

        # Scene objects (initialized in reset)
        self.robot_id = None
        self.plane_id = None
        self.object_id = None
        self.target_id = None

        # State tracking for incremental rewards
        self.prev_dist_obj_target = None
        self.prev_dist_ee_obj = None
        self.prev_yaw_error = None
        self.step_count = 0

        # Target pose (set in reset)
        self.target_pos = np.zeros(3)
        self.target_yaw = 0.0

        # Fixed end-effector orientation (pointing down)
        self.fixed_orientation = p.getQuaternionFromEuler([math.pi, 0, 0])

        # Camera matrices for image observation
        self._setup_cameras()

    def _setup_cameras(self):
        """Set up camera matrices for rendering."""
        # Top-down view for image observation
        self.view_matrix = p.computeViewMatrix(
            cameraEyePosition=[0.5, 0, 2.0],
            cameraTargetPosition=[0.5, 0, 0],
            cameraUpVector=[1, 0, 0]
        )
        self.proj_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=float(self.img_width) / self.img_height,
            nearVal=0.1,
            farVal=100.0
        )

        # Global camera for video recording
        self.global_view_matrix = p.computeViewMatrix(
            cameraEyePosition=[1.2, -0.5, 0.8],
            cameraTargetPosition=[0.5, 0, 0.1],
            cameraUpVector=[0, 0, 1]
        )
        self.global_proj_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=float(self.img_width) / self.img_height,
            nearVal=0.1,
            farVal=100.0
        )

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment."""
        super(BasePushEnv, self).__init__()  # gymnasium.Env.reset() for seeding
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(1.0 / 120.0)

        # Reset reward tracking
        self.prev_dist_obj_target = None
        self.prev_dist_ee_obj = None
        self.prev_yaw_error = None

        # Load plane
        self.plane_id = p.loadURDF("plane.urdf")

        # Load robot (Kuka iiwa)
        self.robot_id = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True)

        # Create object
        col_box_id = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=list(self.cfg.object_half_extents)
        )
        visual_box_id = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=list(self.cfg.object_half_extents),
            rgbaColor=[1, 0, 0, 1]
        )

        # Randomize object position
        object_x = self.np_random.uniform(*self.cfg.object_x_range)
        object_y = self.np_random.uniform(*self.cfg.object_y_range)
        self.object_pos = [object_x, object_y, 0.1]
        self.object_id = p.createMultiBody(
            baseMass=1,
            baseCollisionShapeIndex=col_box_id,
            baseVisualShapeIndex=visual_box_id,
            basePosition=self.object_pos
        )
        p.changeDynamics(self.object_id, -1, ccdSweptSphereRadius=0.002)

        # Create target visual
        visual_target_id = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=list(self.cfg.target_half_extents),
            rgbaColor=[0, 1, 0, 0.5]
        )
        target_x = self.np_random.uniform(*self.cfg.target_x_range)
        target_y = self.np_random.uniform(*self.cfg.target_y_range)
        self.target_pos = np.array([target_x, target_y, 0.0])

        # Randomize target orientation
        self.target_yaw = self.np_random.uniform(-math.pi, math.pi)
        target_orn = p.getQuaternionFromEuler([0, 0, self.target_yaw])
        self.target_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual_target_id,
            basePosition=self.target_pos,
            baseOrientation=target_orn
        )

        # Reset robot to initial pose
        target_ee_pos = [
            self.cfg.fixed_ee_initial_pos[0],
            self.cfg.fixed_ee_initial_pos[1],
            self.cfg.fixed_ee_z
        ]
        joint_poses = p.calculateInverseKinematics(
            self.robot_id, 6, target_ee_pos, self.fixed_orientation
        )
        for i in range(7):
            p.resetJointState(self.robot_id, i, joint_poses[i])

        # Let physics settle
        for _ in range(100):
            p.stepSimulation()

        self.step_count = 0

        return self._get_obs(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one environment step."""
        # Scale action
        dx, dy, _ = action * 0.1
        dz = 0

        # Get current EE position
        current_ee_state = p.getLinkState(self.robot_id, 6)
        current_ee_pos = current_ee_state[0]

        # Compute new target position
        new_ee_pos = [
            current_ee_pos[0] + dx,
            current_ee_pos[1] + dy,
            self.cfg.fixed_ee_z
        ]

        # IK control
        joint_poses = p.calculateInverseKinematics(
            self.robot_id, 6, new_ee_pos, self.fixed_orientation
        )
        for i in range(7):
            p.setJointMotorControl2(
                self.robot_id, i, p.POSITION_CONTROL, joint_poses[i]
            )

        # Step physics (12 steps for 10Hz control at 120Hz physics)
        for _ in range(12):
            p.stepSimulation()
            if self.render_mode == "human":
                time.sleep(1./120.)

        # Enforce 2D constraint on object
        pos, orn = p.getBasePositionAndOrientation(self.object_id)
        new_pos = [pos[0], pos[1], 0.1]
        euler = p.getEulerFromQuaternion(orn)
        new_orn = p.getQuaternionFromEuler([0, 0, euler[2]])
        p.resetBasePositionAndOrientation(self.object_id, new_pos, new_orn)

        self.step_count += 1

        # Compute reward
        reward, terminated, truncated = self._compute_reward()

        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self) -> np.ndarray:
        """Get current observation."""
        if self.obs_type == "image":
            return self._get_image_obs()
        else:
            return self._get_state_obs()

    def _get_state_obs(self) -> np.ndarray:
        """Get 19D state observation."""
        # End-effector position
        ee_state = p.getLinkState(self.robot_id, 6)
        ee_pos = np.array(ee_state[0][:2])

        # Object position and orientation
        obj_pos_full, obj_orn = p.getBasePositionAndOrientation(self.object_id)
        obj_pos = np.array(obj_pos_full[:2])
        obj_euler = p.getEulerFromQuaternion(obj_orn)
        obj_yaw = obj_euler[2]

        # Target position
        target_pos = np.array(self.target_pos[:2])
        target_yaw = self.target_yaw

        # Relative positions
        ee_to_obj = obj_pos - ee_pos
        obj_to_target = target_pos - obj_pos

        # Object velocity
        obj_vel_full, obj_ang_vel_full = p.getBaseVelocity(self.object_id)
        obj_vel = np.array(obj_vel_full[:2])
        obj_angular_vel = obj_ang_vel_full[2]

        # Yaw error
        yaw_error = self._normalize_angle(target_yaw - obj_yaw)

        # Sin/cos encoding
        obj_yaw_sincos = [np.sin(obj_yaw), np.cos(obj_yaw)]
        target_yaw_sincos = [np.sin(target_yaw), np.cos(target_yaw)]
        yaw_error_sincos = [np.sin(yaw_error), np.cos(yaw_error)]

        # Concatenate
        obs = np.concatenate((
            ee_pos,
            obj_pos,
            target_pos,
            ee_to_obj,
            obj_to_target,
            obj_vel,
            obj_yaw_sincos,
            target_yaw_sincos,
            yaw_error_sincos,
            [obj_angular_vel]
        ))

        return obs.astype(np.float32)

    def _get_image_obs(self) -> np.ndarray:
        """Get 84x84 RGB image observation."""
        w, h, rgb, _, _ = p.getCameraImage(
            width=self.img_width,
            height=self.img_height,
            viewMatrix=self.view_matrix,
            projectionMatrix=self.proj_matrix,
            renderer=p.ER_TINY_RENDERER
        )
        rgb = np.array(rgb, dtype=np.uint8)
        rgb = np.reshape(rgb, (h, w, 4))
        rgb = rgb[:, :, :3]
        return rgb

    def _compute_reward(self) -> Tuple[float, bool, bool]:
        """Compute reward, terminated, truncated."""
        # Get positions
        obj_pos, obj_orn = p.getBasePositionAndOrientation(self.object_id)
        obj_pos = np.array(obj_pos[:2])
        obj_euler = p.getEulerFromQuaternion(obj_orn)
        obj_yaw = obj_euler[2]

        target_pos = np.array(self.target_pos[:2])
        target_yaw = self.target_yaw

        ee_pos = np.array(p.getLinkState(self.robot_id, 6)[0][:2])

        # Distances and errors
        dist_obj_target = np.linalg.norm(obj_pos - target_pos)
        dist_ee_obj = np.linalg.norm(ee_pos - obj_pos)
        yaw_error = abs(self._normalize_angle(target_yaw - obj_yaw))

        # Initialize previous values
        if self.prev_dist_obj_target is None:
            self.prev_dist_obj_target = dist_obj_target
        if self.prev_dist_ee_obj is None:
            self.prev_dist_ee_obj = dist_ee_obj
        if self.prev_yaw_error is None:
            self.prev_yaw_error = yaw_error

        # Incremental changes
        delta_dist_target = self.prev_dist_obj_target - dist_obj_target
        delta_dist_ee_obj = self.prev_dist_ee_obj - dist_ee_obj
        delta_yaw_error = self.prev_yaw_error - yaw_error

        # Update previous values
        self.prev_dist_obj_target = dist_obj_target
        self.prev_dist_ee_obj = dist_ee_obj
        self.prev_yaw_error = yaw_error

        # Reward components
        reward = self.cfg.step_penalty

        # Position progress
        reward += delta_dist_target * self.cfg.position_progress_coef

        # Orientation progress
        reward += delta_yaw_error * self.cfg.orientation_progress_coef

        # Coupling bonus
        if delta_dist_target > 0 and delta_yaw_error > 0:
            coupling = min(delta_dist_target * 10, delta_yaw_error) * self.cfg.coupling_coef
            reward += coupling

        # Alignment reward
        vec_ee_to_obj = obj_pos - ee_pos
        vec_obj_to_target = target_pos - obj_pos
        norm_ee_obj = np.linalg.norm(vec_ee_to_obj)
        norm_obj_target = np.linalg.norm(vec_obj_to_target)

        if norm_ee_obj > 0.01 and norm_obj_target > 0.01:
            alignment = np.dot(vec_ee_to_obj, vec_obj_to_target) / (norm_ee_obj * norm_obj_target)
            if dist_ee_obj < self.cfg.contact_threshold * 2:
                reward += alignment * self.cfg.alignment_coef

        # EE approach reward
        if dist_ee_obj > self.cfg.contact_threshold:
            reward += delta_dist_ee_obj * self.cfg.ee_approach_coef
        else:
            if delta_dist_target > 0:
                reward += self.cfg.contact_reward * 2.0
            else:
                reward += self.cfg.contact_reward * 0.5

        # Success check
        terminated = False
        position_success = dist_obj_target < self.cfg.success_threshold
        orientation_success = yaw_error < self.cfg.orientation_threshold

        if position_success and orientation_success:
            reward += self.cfg.success_bonus
            terminated = True

        truncated = (not terminated) and (self.step_count >= self.cfg.max_episode_steps)

        return reward, terminated, truncated

    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def render(self) -> Optional[np.ndarray]:
        """Render the environment."""
        if self.render_mode == "rgb_array":
            w, h, rgb, _, _ = p.getCameraImage(
                width=320,
                height=240,
                viewMatrix=self.global_view_matrix,
                projectionMatrix=self.global_proj_matrix,
                renderer=p.ER_TINY_RENDERER
            )
            rgb = np.array(rgb, dtype=np.uint8)
            rgb = np.reshape(rgb, (240, 320, 4))
            return rgb[:, :, :3]
        return None

    def close(self):
        """Close the environment."""
        p.disconnect()

    # Debug methods
    def get_ee_position(self) -> np.ndarray:
        """Get current end-effector position."""
        ee_state = p.getLinkState(self.robot_id, 6)
        return np.array(ee_state[0])

    def get_object_pose(self) -> Tuple[np.ndarray, float]:
        """Get current object position and yaw angle."""
        pos, orn = p.getBasePositionAndOrientation(self.object_id)
        euler = p.getEulerFromQuaternion(orn)
        return np.array(pos), euler[2]

    def get_target_pose(self) -> Tuple[np.ndarray, float]:
        """Get target position and yaw angle."""
        return self.target_pos.copy(), self.target_yaw
