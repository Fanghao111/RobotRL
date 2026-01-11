# Robot Arm Pushing Task with RL (PPO)

This project implements a Reinforcement Learning agent using PPO (Proximal Policy Optimization) to train a Kuka iiwa robot arm to push a block to a target location in a PyBullet simulation.

## Prerequisites

- Python 3.8+
- Virtual environment (recommended)

## Installation

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Project Structure

```
RL/
├── env/                    # 环境模块
│   ├── __init__.py
│   └── push_env.py         # Custom Gymnasium environment
├── scripts/                # 辅助脚本
│   ├── analyze_logs.py     # TensorBoard日志分析
│   ├── analyze_sim_logs.py # 仿真日志分析
│   └── check_model_arch.py # 检查模型架构
├── models/                 # 训练好的模型
│   ├── ppo_push_robot.zip
│   ├── ppo_push_robot_image.zip
│   └── *.pkl               # VecNormalize统计
├── logs/                   # 运行日志
├── train_unified.py        # 统一训练脚本
├── evaluate_unified.py     # 统一评估脚本
├── simple_push_test.py     # 脚本策略测试
└── requirements.txt        # 依赖列表
```

## Usage

**Note**: Please run all commands in the `fhz-rl` conda environment.

### 1. Training

To train the agent, use `train_unified.py`. It supports both observation types ('state' or 'image').

**Train using state observations (Default):**
```bash
python train_unified.py --obs-type state --timesteps 1000000
```
This trains an MLP policy and saves to `checkpoints/` and `ppo_push_robot.zip`.

**Train using image observations:**
```bash
python train_unified.py --obs-type image --timesteps 1000000
```
This trains a CNN policy and saves to `checkpoints_image/` and `ppo_push_robot_image.zip`.

**Additional Arguments:**
- `--save-freq`: Checkpoint saving frequency (default: 50000)
- `--load-checkpoint`: Path to a .zip file to resume training
- `--n-envs`: Number of parallel environments (default: 4)

### 2. Evaluation

To evaluate the trained agent, use `evaluate_unified.py`.

**Evaluate state model:**
```bash
python evaluate_unified.py --mode state
# Use --save_images to save visualization frames
python evaluate_unified.py --mode state --save_images
```

**Evaluate image model:**
```bash
python evaluate_unified.py --mode image --save_images
```

**Evaluate specific checkpoint(s):**
```bash
# Evaluate a single file
python evaluate_unified.py --model_path checkpoints/ppo_push_robot_100000_steps.zip

# Evaluate all checkpoints in a directory (sorted by steps)
python evaluate_unified.py --model_path checkpoints/
```

Results are saved to `evaluation_results/`.

## Environment Details

- **Robot**: Kuka iiwa 7-DOF arm.
- **Task**: Push a red cube to a green target zone.
- **Observation Space**: 9-dimensional vector (End effector pos, Object pos, Target pos).
- **Action Space**: 3-dimensional vector (End effector velocity dx, dy, dz).
- **Reward**: Negative distance to target + bonus for reaching target.

## Notes

- The environment uses Inverse Kinematics to control the robot arm based on desired end-effector velocity.
- Training time may vary depending on hardware. 100k timesteps is a starting point; more may be needed for optimal performance.
