# Push Task RL - Unified Framework

A unified reinforcement learning framework for robot arm pushing tasks, supporting both **PyBullet** (CPU) and **Isaac Lab** (GPU) backends with **Stable Baselines 3**.

## Features

- **Unified Interface**: Same API for both PyBullet and Isaac Lab environments
- **Backend Switching**: Seamlessly switch between CPU (PyBullet) and GPU (Isaac Lab) training
- **Stable Baselines 3**: Uses PPO algorithm with identical hyperparameters across backends
- **Configurable**: YAML-based configuration for environment and training parameters
- **Curriculum Learning**: Support for multi-stage training pipelines

## Project Structure

```
RL/
├── envs/                       # Unified environment module
│   ├── __init__.py
│   ├── factory.py              # Environment factory (make_env, make_vec_env)
│   ├── base/                   # Base classes
│   │   ├── __init__.py
│   │   └── base_env.py         # BasePushEnv, BasePushEnvConfig
│   ├── pybullet/               # PyBullet backend
│   │   ├── __init__.py
│   │   └── push_env.py         # PyBulletPushEnv
│   └── isaac_lab/              # Isaac Lab backend
│       ├── __init__.py
│       ├── push_env.py         # IsaacLabPushEnv, IsaacLabVecEnvWrapper
│       └── isaac_push_env.py   # Internal Isaac Lab implementation
│
├── configs/                    # Configuration files
│   ├── config.yaml             # Default configuration
│   ├── config-s1.yaml          # Stage 1 curriculum config
│   ├── config-s2.yaml          # Stage 2 curriculum config
│   └── config-s3.yaml          # Stage 3 curriculum config
│
├── train.py                    # Unified training script
├── evaluate.py                 # Unified evaluation script
├── models/                     # Trained models output
└── requirements.txt            # Dependencies
```

## Installation

### Basic Installation (PyBullet only)

```bash
pip install -r requirements.txt
```

### Full Installation (with Isaac Lab)

Isaac Lab requires NVIDIA Isaac Sim. Follow the [official installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation.html).

## Quick Start

### Training with PyBullet (CPU)

```bash
# Basic training
python train.py --backend pybullet --timesteps 1000000

# With checkpoint saving
python train.py --backend pybullet --timesteps 1000000 --save-freq 100000

# Custom config
python train.py --backend pybullet --config config-s2.yaml --timesteps 3000000
```

### Training with Isaac Lab (GPU)

```bash
# GPU-accelerated training with 1024 parallel envs
python train.py --backend isaac_lab --timesteps 1000000 --n-envs 1024

# Headless training on server
python train.py --backend isaac_lab --n-envs 4096 --timesteps 2000000
```

### Evaluation

```bash
# Evaluate latest model
python evaluate.py --backend pybullet

# Evaluate with video recording
python evaluate.py --backend pybullet --save-video

# Evaluate specific model
python evaluate.py --model-path ./models/xxx/ppo_push_robot.zip

# Evaluate all checkpoints in directory
python evaluate.py --model-path ./models/xxx/checkpoints/
```

### Resume Training

```bash
python train.py --backend pybullet --load-checkpoint ./models/xxx/ --timesteps 500000
```

## Environment Details

### Task Description

The robot arm must push a rectangular object to a target position and orientation.

- **Robot**: Kuka iiwa 7-DOF (PyBullet) / Franka Panda (Isaac Lab)
- **Task**: Push red block to green target zone with correct orientation
- **Success**: Position error < 5cm AND orientation error < threshold

### Observation Space (19D state vector)

| Feature | Dims | Description |
|---------|------|-------------|
| ee_xy | 2 | End-effector position |
| obj_xy | 2 | Object position |
| target_xy | 2 | Target position |
| ee_to_obj_xy | 2 | Relative: EE to object |
| obj_to_target_xy | 2 | Relative: object to target |
| obj_vel_xy | 2 | Object linear velocity |
| obj_yaw_sincos | 2 | Object orientation (sin/cos) |
| target_yaw_sincos | 2 | Target orientation (sin/cos) |
| yaw_error_sincos | 2 | Orientation error (sin/cos) |
| obj_angular_vel | 1 | Object angular velocity (Z) |

### Action Space (3D)

- Continuous delta position: (dx, dy, dz)
- dz is ignored (fixed end-effector height)
- Range: [-1, 1], scaled by 0.1

### Reward Structure

1. **Position Progress**: Incremental reward for moving object closer to target
2. **Orientation Progress**: Incremental reward for reducing yaw error
3. **Coupling Bonus**: Extra reward when both improve simultaneously
4. **Alignment**: Reward for pushing from correct direction
5. **EE Approach**: Reward for end-effector approaching object
6. **Success Bonus**: Large reward upon task completion
7. **Step Penalty**: Small penalty per step

## Configuration

Configuration is done via YAML files. Key sections:

```yaml
# Environment settings
env:
  max_episode_steps: 200
  success_threshold: 0.05      # Position threshold (m)
  orientation_threshold: 0.15  # Orientation threshold (rad)

# Reward coefficients
reward:
  position_progress_coef: 30.0
  orientation_progress_coef: 15.0
  success_bonus: 100.0

# Training settings
training:
  state:
    learning_rate: 0.0001
    net_arch_pi: [256, 256, 128]
    net_arch_vf: [256, 256, 128]
```

## Backend Comparison

| Feature | PyBullet | Isaac Lab |
|---------|----------|-----------|
| Device | CPU | GPU (CUDA) |
| Parallel Envs | 12 (SubprocVecEnv) | 1024+ (native) |
| Training Speed | ~1-2 hours / 1M steps | ~5-10 min / 1M steps |
| Robot | Kuka iiwa | Franka Panda |
| Dependencies | pybullet | Isaac Sim |

## Programmatic Usage

```python
from envs import make_env, make_vec_env
from envs.base import BasePushEnvConfig

# Create single environment
env = make_env(backend="pybullet", obs_type="state")

# Create vectorized environment
vec_env = make_vec_env(backend="pybullet", n_envs=12)

# With custom config
cfg = BasePushEnvConfig(
    max_episode_steps=200,
    success_threshold=0.05,
    orientation_threshold=0.15,
)
env = make_env(backend="pybullet", cfg=cfg)
```
