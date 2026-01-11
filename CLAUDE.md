# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Note**: 
- **Install Dependencies**: `pip install -r requirements.txt`
- **Train Agent**: `python train_unified.py` (options: `--obs-type [state|image]`, `--timesteps N`, `--load-checkpoint PATH`, `--n-envs N`, `--save-freq N`)
- **Evaluate Agent**: `python evaluate_unified.py` (options: `--mode [state|image]`, `--model-path PATH`, `--save-images`)
- **Run Simple Test**: `python simple_push_test.py` - Runs a scripted policy to debug environment

## Architecture

- **Environment (`env/push_env.py`)**: Custom Gymnasium env wrapping PyBullet
  - **Robot**: Kuka iiwa 7-DOF arm controlled via Inverse Kinematics
  - **Task**: Push a red cube to a green target location
  - **Observations**:
    - State mode: 9D vector (EE pos, Object pos, Target pos)
    - Image mode: 84x84 RGB top-down view
  - **Actions**: 3D continuous velocity (dx, dy, dz) of end-effector
  - **Reward**: Dense distance improvement reward + sparse success bonus + step penalty

- **Training**
  - Uses Stable Baselines 3 PPO algorithm
  - `train_unified.py`: Unified training script (supports both state and image obs)
  - Model artifacts saved to `models/` directory as `.zip` files
  - Logs to TensorBoard directories

- **Evaluation**
  - `evaluate_unified.py` loads trained models (single file or directory of checkpoints)
  - Calculates success rates over episodes
  - Can evaluate both state-based and image-based models
  - Saves visualization frames to `evaluation_results/` if `--save_images` is set

- **Scripts (`scripts/`)**
  - `analyze_logs.py`: Analyzes TensorBoard training logs
  - `analyze_sim_logs.py`: Analyzes simulation episode logs
  - `check_model_arch.py`: Inspects model architecture
