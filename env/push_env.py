import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np
import time
import math
import sys
import os

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_loader import get_env_config, get_reward_config

class PushEnv(gym.Env):
    def __init__(self, render_mode=None, obs_type="state"):
        super(PushEnv, self).__init__()
        self.render_mode = render_mode
        self.obs_type = obs_type # "state" or "image"
        
        # Load configuration
        env_cfg = get_env_config()
        self.max_steps = env_cfg["max_episode_steps"]
        self.fixed_ee_z = env_cfg["fixed_ee_z"]
        self.success_threshold = env_cfg["success_threshold"]
        self.object_x_range = env_cfg["object_x_range"]
        self.object_y_range = env_cfg["object_y_range"]
        self.target_x_range = env_cfg["target_x_range"]
        self.target_y_range = env_cfg["target_y_range"]
        
        # Load reward configuration
        reward_cfg = get_reward_config()
        self.target_progress_coef = reward_cfg["target_progress_coef"]
        self.ee_approach_coef = reward_cfg["ee_approach_coef"]
        self.contact_threshold = reward_cfg["contact_threshold"]
        self.contact_reward = reward_cfg["contact_reward"]
        self.success_bonus = reward_cfg["success_bonus"]
        self.step_penalty = reward_cfg["step_penalty"]
        
        # Connect to PyBullet
        if render_mode == "human":
            self.physicsClient = p.connect(p.GUI)
        else:
            self.physicsClient = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Action space: 3D velocity for the end effector (dx, dy, dz) + gripper (optional, but we just push)
        # Let's simplify to 2D movement (dx, dy) since it's pushing on a plane,
        # but 3D is more general. Let's do 3D for the end effector target velocity.
        # We will use inverse kinematics to control the arm.
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)

        # Observation space:
        self.img_width = 84 # Reduced resolution for faster training
        self.img_height = 84

        if self.obs_type == "image":
            # Image observation (H, W, 3) RGB from Top-down view
            self.observation_space = spaces.Box(low=0, high=255, shape=(self.img_height, self.img_width, 3), dtype=np.uint8)
        else:
            # State observation:
            # 1. ee_xy (2)
            # 2. obj_xy (2)
            # 3. target_xy (2)
            # 4. ee_to_obj_xy (2)
            # 5. obj_to_target_xy (2)
            # 6. obj_vel_xy (2)
            # Shape: (12,)
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)

        self.robotId = None
        self.planeId = None
        self.objectId = None
        self.targetId = None

        # Camera parameters
        self.view_matrix = p.computeViewMatrix(
            cameraEyePosition=[0.5, 0, 1.0],
            cameraTargetPosition=[0.5, 0, 0],
            cameraUpVector=[0, 1, 0]
        )
        self.proj_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=float(self.img_width) / self.img_height,
            nearVal=0.1,
            farVal=100.0
        )

        # Global Camera 1 parameters (Side view)
        self.global_view_matrix_1 = p.computeViewMatrix(
            cameraEyePosition=[1.5, 0, 1.5], # Further away and higher
            cameraTargetPosition=[0.5, 0, 0], # Looking at the workspace center
            cameraUpVector=[0, 0, 1]
        )

        # Global Camera 2 parameters (Top-down view)
        self.global_view_matrix_2 = p.computeViewMatrix(
            cameraEyePosition=[0.5, 0, 2.0], # Directly above
            cameraTargetPosition=[0.5, 0, 0],
            cameraUpVector=[1, 0, 0] # X-axis is up in image
        )

        # Global Camera 3 parameters (Front view)
        self.global_view_matrix_3 = p.computeViewMatrix(
            cameraEyePosition=[0.5, -1.5, 1.0], # From the front/side
            cameraTargetPosition=[0.5, 0, 0],
            cameraUpVector=[0, 0, 1]
        )

        self.global_proj_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=float(self.img_width) / self.img_height,
            nearVal=0.1,
            farVal=100.0
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        # [修改] 物理仿真频率 120Hz
        p.setTimeStep(1.0 / 120.0)

         # Reset prev_dist
        if hasattr(self, 'prev_dist_obj_target'):
            del self.prev_dist_obj_target
        if hasattr(self, 'prev_dist_ee_obj'):
            del self.prev_dist_ee_obj

        # Load plane
        self.planeId = p.loadURDF("plane.urdf")

        # Load robot (Kuka iiwa)
        self.robotId = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True)

        # Load object (cube/rectangle)
        # Create a visual and collision shape for the block
        # Increased size from 0.05 to 0.1
        colBoxId = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1])
        visualBoxId = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1], rgbaColor=[1, 0, 0, 1])

        # Randomize object start position
        # If seed is provided in reset(), self.np_random is seeded by super().reset(seed=seed)
        object_x = self.np_random.uniform(self.object_x_range[0], self.object_x_range[1])
        object_y = self.np_random.uniform(self.object_y_range[0], self.object_y_range[1])
        self.object_pos = [object_x, object_y, 0.1]
        self.objectId = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=colBoxId,
                                          baseVisualShapeIndex=visualBoxId, basePosition=self.object_pos)

        # [修改] 使用人工重置方式代替约束，避免 pybullet 版本兼容问题
        # 仅开启 CCD 防止穿模
        p.changeDynamics(self.objectId, -1, ccdSweptSphereRadius=0.002)

        # Create target visual (non-colliding)
        visualTargetId = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.001], rgbaColor=[0, 1, 0, 0.5])
        target_x = self.np_random.uniform(self.target_x_range[0], self.target_x_range[1])
        target_y = self.np_random.uniform(self.target_y_range[0], self.target_y_range[1])
        self.target_pos = np.array([target_x, target_y, 0.0])  # 随机目标位置
        self.targetId = p.createMultiBody(baseMass=0, baseVisualShapeIndex=visualTargetId, basePosition=self.target_pos)

        # Reset robot joints to a good starting configuration
        # Kuka iiwa has 7 joints
        # Instead of fixed rest poses, let's use IK to position the EE above the object
        # so the camera (eye-in-hand) sees the object immediately.

        # Target EE position: slightly behind and above the object
        # Object is at self.object_pos
        # Calculate direction from object to target to position EE behind the object aligned with the target
        vec_obj_to_target = self.target_pos - np.array(self.object_pos)
        vec_obj_to_target[2] = 0 # Ignore Z for direction
        dist = np.linalg.norm(vec_obj_to_target)
        if dist > 0:
            dir_obj_to_target = vec_obj_to_target / dist
        else:
            dir_obj_to_target = np.array([1.0, 0.0, 0.0]) # Default to X+

        # Position EE 0.2m behind the object along the line of sight to target
        target_ee_pos = np.array(self.object_pos) - dir_obj_to_target * 0.2
        target_ee_pos[2] = self.fixed_ee_z

        # Convert back to list if needed, though numpy array works for IK usually
        target_ee_pos = target_ee_pos.tolist()

        # We want the camera (z-axis of EE) to point at the object.
        # And we want the gripper to be somewhat horizontal or pointing down.
        # For simplicity, let's just set a fixed orientation that points the EE down/forward.
        # Quaternion for pointing down/forward
        # By Pybullet convention, [0, 1, 0, 0] is a 180 degree rotation around X, flipping Z to point down
        self.fixed_orientation = p.getQuaternionFromEuler([math.pi, 0, 0]) # Vertical down
        # Kuka default: z-axis is along the last link.
        # If we rotate around X by 180 deg, Z will point down.
        # Let's try to have the EE point towards the object.

        joint_poses = p.calculateInverseKinematics(self.robotId, 6, target_ee_pos, self.fixed_orientation)

        for i in range(7):
            p.resetJointState(self.robotId, i, joint_poses[i])

        # Let physics settle
        for _ in range(100):
            p.stepSimulation()

        self.step_count = 0

        return self._get_obs(), {}

    def _get_obs(self):
        if self.obs_type == "image":
            # Return Top-down view image
            w, h, rgb, depth, seg = p.getCameraImage(
                width=self.img_width,
                height=self.img_height,
                viewMatrix=self.global_view_matrix_2, # Top-down view
                projectionMatrix=self.global_proj_matrix,
                renderer=p.ER_BULLET_HARDWARE_OPENGL
            )
            rgb = np.array(rgb, dtype=np.uint8)
            rgb = np.reshape(rgb, (h, w, 4))
            rgb = rgb[:, :, :3] # Remove alpha channel
            return rgb
        else:
            # Get end effector position (XY only)
            ee_state = p.getLinkState(self.robotId, 6)
            ee_pos = np.array(ee_state[0][:2])

            # Get object position (XY only)
            obj_pos_full, _ = p.getBasePositionAndOrientation(self.objectId)
            obj_pos = np.array(obj_pos_full[:2])

            # Get target position (XY only)
            target_pos = np.array(self.target_pos[:2])

            # Relative positions
            ee_to_obj = obj_pos - ee_pos
            obj_to_target = target_pos - obj_pos

            # Object velocity (XY linear velocity)
            obj_vel_full, _ = p.getBaseVelocity(self.objectId)
            obj_vel = np.array(obj_vel_full[:2])

            # Concatenate all features
            obs = np.concatenate((
                ee_pos,
                obj_pos,
                target_pos,
                ee_to_obj,
                obj_to_target,
                obj_vel
            ))
            return obs.astype(np.float32)

    def step(self, action):
        # Action is delta position or velocity for end effector
        # Scale action
        dx, dy, _ = action * 0.1
        dz = 0

        current_ee_state = p.getLinkState(self.robotId, 6)
        current_ee_pos = current_ee_state[0]
        # current_ee_orn = current_ee_state[1] # Don't use current orientation, use fixed one

        new_ee_pos = [
            current_ee_pos[0] + dx,
            current_ee_pos[1] + dy,
            self.fixed_ee_z
        ]

        # Calculate joint angles using Inverse Kinematics
        # Use self.fixed_orientation instead of current_ee_orn to maintain orientation
        joint_poses = p.calculateInverseKinematics(self.robotId, 6, new_ee_pos, self.fixed_orientation)

        # Apply joint controls
        for i in range(7):
            p.setJointMotorControl2(self.robotId, i, p.POSITION_CONTROL, joint_poses[i])

        # [修改] 降低控制频率 (Frame Skip)
        # 物理频率为 120Hz，为了保持 10Hz 的控制频率，每次 action 执行 12 次 stepSimulation。
        for _ in range(12):
            p.stepSimulation()
            if self.render_mode == "human":
                time.sleep(1./120.)
        
        # [优化] 将强制 2D 约束的处理移到循环外以提升训练FPS
        # 只需要在每次 env.step 结束时统一修正一次即可
        pos, orn = p.getBasePositionAndOrientation(self.objectId)
        new_pos = [pos[0], pos[1], 0.1]
        euler = p.getEulerFromQuaternion(orn)
        # 保持 Z 轴旋转 (Yaw), 归零 X/Y 轴旋转 (Roll/Pitch)
        new_orn = p.getQuaternionFromEuler([0, 0, euler[2]])
        
        p.resetBasePositionAndOrientation(self.objectId, new_pos, new_orn)

        self.step_count += 1
        # Calculate reward
        reward, terminated, truncated = self._compute_reward()

        return self._get_obs(), reward, terminated, truncated, {}

    def _compute_reward(self):
        # 获取物体、目标、末端执行器的2D位置（忽略Z轴，只关注平面移动）
        obj_pos, _ = p.getBasePositionAndOrientation(self.objectId)
        obj_pos = np.array(obj_pos[:2])  # 只取XY平面
        target_pos = np.array(self.target_pos[:2])
        ee_pos = np.array(p.getLinkState(self.robotId, 6)[0][:2])

        # 1. 基础距离计算
        dist_obj_target = np.linalg.norm(obj_pos - target_pos)  # 物体到目标的距离
        dist_ee_obj = np.linalg.norm(ee_pos - obj_pos)          # 末端执行器到物体的距离

        # 初始化历史距离（首次调用时）
        if not hasattr(self, 'prev_dist_obj_target'):
            self.prev_dist_obj_target = dist_obj_target
        if not hasattr(self, 'prev_dist_ee_obj'):
            self.prev_dist_ee_obj = dist_ee_obj

        # 2. 增量计算
        delta_dist_target = self.prev_dist_obj_target - dist_obj_target  # 正值表示物体靠近目标
        delta_dist_ee_obj = self.prev_dist_ee_obj - dist_ee_obj          # 正值表示EE靠近物体

        # 更新历史距离
        self.prev_dist_obj_target = dist_obj_target
        self.prev_dist_ee_obj = dist_ee_obj

        # 3. 奖励组件设计
        reward_components = {
            "target_progress": 0.0,   # 向目标移动的增量奖励
            "ee_approach": 0.0,       # EE 靠近物体的增量奖励
            "success_bonus": 0.0,     # 成功推到目标的奖励
            "step_penalty": self.step_penalty  # 每步小惩罚（鼓励尽快完成）
        }

        # 向目标移动的增量奖励（核心奖励）
        reward_components["target_progress"] = delta_dist_target * self.target_progress_coef

        # EE 靠近物体的奖励（分阶段）
        if dist_ee_obj > self.contact_threshold:
            # 远离物体时：给予接近奖励（增量式，避免持续惩罚）
            reward_components["ee_approach"] = delta_dist_ee_obj * self.ee_approach_coef
        else:
            # 已接触物体：给予小额持续奖励，鼓励保持接触
            reward_components["ee_approach"] = self.contact_reward

        # 4. 成功奖励（物体到达目标区域）
        terminated = False
        if dist_obj_target < self.success_threshold:
            reward_components["success_bonus"] = self.success_bonus
            terminated = True

        # 5. 总奖励计算
        total_reward = sum(reward_components.values())

        # 截断判断
        truncated = (not terminated) and (self.step_count >= self.max_steps)

        return total_reward, terminated, truncated

    def close(self):
        p.disconnect()
