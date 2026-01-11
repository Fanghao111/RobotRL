import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env.push_env import PushEnv
from config_loader import load_config, get_env_config, get_eval_config
import cv2
import os
import glob
import re
import numpy as np
import argparse
import matplotlib.pyplot as plt


# Load config at module level
_config = None

def get_config():
    global _config
    if _config is None:
        _config = load_config()
    return _config


def find_vecnormalize_file(model_path: str) -> str:
    """Find VecNormalize stats file corresponding to a model."""
    # Try different naming conventions
    base_path = model_path.replace('.zip', '')
    
    possible_paths = [
        base_path + '.pkl',
        base_path + '_vecnormalize.pkl',
        os.path.join(os.path.dirname(model_path), 'vecnormalize.pkl'),
    ]
    
    # Also check for similar named pkl files in same directory
    model_dir = os.path.dirname(model_path)
    if model_dir:
        pkl_files = glob.glob(os.path.join(model_dir, '*.pkl'))
        # Find pkl with matching step count
        model_steps = get_steps(model_path)
        for pkl in pkl_files:
            if str(model_steps) in pkl:
                possible_paths.insert(0, pkl)
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None


def create_eval_env(obs_type: str, vecnorm_path: str = None):
    """Create evaluation environment with optional VecNormalize."""
    env_cfg = get_env_config()
    max_steps = env_cfg["max_episode_steps"]
    
    def make_env():
        env = PushEnv(render_mode="rgb_array", obs_type=obs_type)
        env = TimeLimit(env, max_episode_steps=max_steps)
        return env
    
    # Wrap in DummyVecEnv (required for VecNormalize)
    env = DummyVecEnv([make_env])
    
    # Load VecNormalize if available (only for state mode)
    if obs_type == "state" and vecnorm_path:
        print(f"Loading VecNormalize stats from: {vecnorm_path}")
        env = VecNormalize.load(vecnorm_path, env)
        env.training = False  # Disable training mode (don't update stats)
        env.norm_reward = False  # Don't normalize rewards during eval
    elif obs_type == "state":
        print("Warning: No VecNormalize stats found. Using unnormalized observations.")
    
    return env


def get_steps(filename: str) -> int:
    """Extract step count from checkpoint filename."""
    match = re.search(r'_(\d+)_steps', filename)
    return int(match.group(1)) if match else 0


def get_model_path(args) -> str:
    """Determine model path based on arguments."""
    if args.model_path:
        return args.model_path
    
    # Auto-detect latest model in tmp/ directory (training output)
    tmp_dir = "./tmp"
    if not os.path.exists(tmp_dir):
        print(f"Tmp directory {tmp_dir} not found.")
        return None
    
    # Find all model directories with timestamp pattern (YYYYMMDD_HHMMSS_type)
    suffix = "_image" if args.mode == "image" else "_state"
    model_dirs = [d for d in os.listdir(tmp_dir) 
                  if os.path.isdir(os.path.join(tmp_dir, d)) and d.endswith(suffix)]
    
    if not model_dirs:
        print(f"No model directories found in {tmp_dir} with suffix '{suffix}'")
        return None
    
    # Sort by timestamp (format: YYYYMMDD_HHMMSS_type) - latest first
    model_dirs.sort(reverse=True)
    latest_dir = os.path.join(tmp_dir, model_dirs[0])
    
    # Look for model files directly in the model directory
    zip_files = glob.glob(os.path.join(latest_dir, "*.zip"))
    
    if zip_files:
        # Only look for files without "_steps" (final model), ignore checkpoints
        final_models = [f for f in zip_files if "_steps" not in os.path.basename(f)]
        if final_models:
            print(f"Auto-detected latest model: {final_models[0]}")
            return final_models[0]
        else:
            print(f"No final model found in {latest_dir} (only checkpoints with step counts exist)")
            return None
    
    # No zip files found
    print(f"No model files found in {latest_dir}")
    return None


def get_models_to_evaluate(model_path: str, mode: str) -> list:
    """Get list of model files to evaluate."""
    if os.path.isdir(model_path):
        checkpoint_dir = model_path
    elif not model_path.endswith('.zip') and os.path.isdir(model_path.replace('.zip', '')):
        checkpoint_dir = model_path.replace('.zip', '')
    else:
        # Single file
        if not model_path.endswith('.zip'):
            model_path += ".zip"
        if not os.path.exists(model_path):
            print(f"Model file {model_path} not found.")
            return []
        return [model_path]
    
    # Directory mode - find all checkpoints
    print(f"Looking for checkpoints in {checkpoint_dir}...")
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.zip"))
    checkpoint_files.sort(key=get_steps)
    
    if not checkpoint_files:
        print(f"No checkpoints found in {checkpoint_dir}")
        return []
    
    return checkpoint_files


def run_episode(env, model, episode_num: int, model_output_dir: str, save_video: bool):
    """Run a single evaluation episode."""
    import pybullet as p
    
    # Get config (cached, only read once per process)
    env_cfg = get_env_config()
    eval_cfg = get_eval_config()
    max_steps = env_cfg["max_episode_steps"]
    video_config = eval_cfg["video"]  # Cache video config for render_frame
    frame_skip = video_config["frame_skip"]
    
    # Set seed for reproducibility
    seed = 42 + episode_num
    np.random.seed(seed)
    
    # VecEnv reset returns only obs (no info), and obs is batched
    obs = env.reset()
    # Set seed on underlying env
    env.env_method('reset', seed=seed)
    obs = env.reset()  # Reset again with seed
    
    done = False
    step = 0
    episode_reward = 0
    
    # Trajectory data for plotting
    ee_trajectory = []
    obj_trajectory = []
    target_pos_fixed = None
    
    # Video frames
    frames = [] if save_video else None
    
    # Get underlying env for coordinate logging
    base_env = env.envs[0].unwrapped if hasattr(env, 'envs') else env.venv.envs[0].unwrapped
    
    while not done:
        # Collect coordinates BEFORE step (to avoid getting reset positions)
        ee_pos = p.getLinkState(base_env.robotId, 6)[0]
        obj_pos, _ = p.getBasePositionAndOrientation(base_env.objectId)
        target_pos = base_env.target_pos
        
        ee_trajectory.append([ee_pos[0], ee_pos[1]])
        obj_trajectory.append([obj_pos[0], obj_pos[1]])
        if target_pos_fixed is None:
            target_pos_fixed = [target_pos[0], target_pos[1]]
        
        # Capture frame for video BEFORE step (to avoid reset frames)
        if save_video and step % frame_skip == 0:
            frame = render_frame(base_env, video_config)
            if frame is not None:
                frames.append(frame)
        
        # Take action
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, dones, infos = env.step(action)
        episode_reward += reward[0]  # Unbatch reward
        done = dones[0]  # Unbatch done
        
        step += 1
        
        # Check for success in info
        if 'terminal_observation' in infos[0]:
            break
    
    # Collect final position (only if not auto-reset by VecEnv)
    # When done=True, VecEnv auto-resets, so we need to get final pos from info
    if 'terminal_observation' in infos[0]:
        # VecEnv stores the final observation before reset
        # For state mode, we can extract positions from terminal_observation
        pass  # Final position already captured in last iteration before done
    else:
        # Episode truncated, capture final position
        ee_pos = p.getLinkState(base_env.robotId, 6)[0]
        obj_pos, _ = p.getBasePositionAndOrientation(base_env.objectId)
        ee_trajectory.append([ee_pos[0], ee_pos[1]])
        obj_trajectory.append([obj_pos[0], obj_pos[1]])
    
    # Convert to numpy arrays
    ee_trajectory = np.array(ee_trajectory)
    obj_trajectory = np.array(obj_trajectory)
    
    # Check if it was a success (terminated=True) or truncation
    success = done and step < max_steps
    
    # Save episode log (txt)
    save_episode_log(episode_num, model_output_dir, success, episode_reward, step,
                     ee_trajectory, obj_trajectory, target_pos_fixed)
    
    # Plot trajectory (png)
    plot_trajectory(ee_trajectory, obj_trajectory, target_pos_fixed, 
                    episode_num, model_output_dir, success)
    
    # Save video (mp4)
    if save_video and frames:
        save_episode_video(frames, episode_num, model_output_dir, success, 
                          fps=video_config["fps"])
    
    return success, episode_reward, step


def render_frame(base_env, video_config):
    """Render a frame from the environment using PyBullet CPU renderer (TinyRenderer).
    
    Uses low resolution for faster rendering.
    
    Args:
        base_env: The unwrapped PyBullet environment
        video_config: Video configuration dict with width/height
    """
    import pybullet as p
    
    width = video_config["width"]
    height = video_config["height"]
    
    # Use a nice viewing angle
    view_matrix = p.computeViewMatrix(
        cameraEyePosition=[1.2, -0.5, 0.8],
        cameraTargetPosition=[0.5, 0, 0.1],
        cameraUpVector=[0, 0, 1]
    )
    proj_matrix = p.computeProjectionMatrixFOV(
        fov=60,
        aspect=width/height,
        nearVal=0.1,
        farVal=100.0
    )
    
    _, _, rgb, _, _ = p.getCameraImage(
        width=width,
        height=height,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix,
        renderer=p.ER_TINY_RENDERER
    )
    
    # Convert to numpy array and remove alpha channel
    frame = np.array(rgb, dtype=np.uint8)
    frame = frame.reshape(height, width, 4)
    frame = frame[:, :, :3]  # RGB only
    
    return frame


def save_episode_video(frames, episode_num, output_dir, success, fps=5):
    """Save frames as a video file using imageio for better compatibility.
    
    Args:
        frames: List of video frames
        episode_num: Episode number
        output_dir: Output directory
        success: Whether episode was successful
        fps: Frames per second
    """
    import imageio
    
    videos_dir = os.path.join(output_dir, "videos")
    os.makedirs(videos_dir, exist_ok=True)
    video_path = os.path.join(videos_dir, f"episode_{episode_num + 1}_video.mp4")
    
    # Use imageio with ffmpeg for H.264 encoding (better compatibility)
    writer = imageio.get_writer(video_path, fps=fps, codec='libx264', 
                                 pixelformat='yuv420p', quality=7)
    
    for frame in frames:
        writer.append_data(frame)
    
    writer.close()
    
    status = "SUCCESS" if success else "FAILED"
    print(f"  Saved video ({len(frames)} frames): {video_path} [{status}]")


def save_episode_log(episode_num, output_dir, success, episode_reward, steps, 
                     ee_traj, obj_traj, target_pos):
    """Save episode results to a text file."""
    logs_dir = os.path.join(output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, f"episode_{episode_num + 1}_log.txt")
    
    final_dist = np.linalg.norm(obj_traj[-1] - np.array(target_pos))
    
    with open(log_path, 'w') as f:
        f.write(f"Episode {episode_num + 1} Results\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Status: {'SUCCESS' if success else 'FAILED'}\n")
        f.write(f"Total Reward: {episode_reward:.4f}\n")
        f.write(f"Total Steps: {steps}\n")
        f.write(f"Final Distance to Target: {final_dist:.4f}m\n\n")
        
        f.write(f"Target Position: ({target_pos[0]:.4f}, {target_pos[1]:.4f})\n")
        f.write(f"Object Start: ({obj_traj[0, 0]:.4f}, {obj_traj[0, 1]:.4f})\n")
        f.write(f"Object End: ({obj_traj[-1, 0]:.4f}, {obj_traj[-1, 1]:.4f})\n")
        f.write(f"EE Start: ({ee_traj[0, 0]:.4f}, {ee_traj[0, 1]:.4f})\n")
        f.write(f"EE End: ({ee_traj[-1, 0]:.4f}, {ee_traj[-1, 1]:.4f})\n")
    
    print(f"  Saved episode log: {log_path}")


def plot_trajectory(ee_traj, obj_traj, target_pos, episode_num, output_dir, success):
    """Plot the trajectory of EE and object for a single episode."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot EE trajectory
    ax.plot(ee_traj[:, 0], ee_traj[:, 1], 'b-', linewidth=1.5, label='EE Trajectory', alpha=0.7)
    ax.scatter(ee_traj[0, 0], ee_traj[0, 1], c='blue', marker='o', s=100, zorder=5, label='EE Start')
    ax.scatter(ee_traj[-1, 0], ee_traj[-1, 1], c='blue', marker='s', s=100, zorder=5, label='EE End')
    
    # Plot Object trajectory
    ax.plot(obj_traj[:, 0], obj_traj[:, 1], 'r-', linewidth=2, label='Object Trajectory', alpha=0.7)
    ax.scatter(obj_traj[0, 0], obj_traj[0, 1], c='red', marker='o', s=150, zorder=5, label='Object Start')
    ax.scatter(obj_traj[-1, 0], obj_traj[-1, 1], c='red', marker='s', s=150, zorder=5, label='Object End')
    
    # Plot Target
    ax.scatter(target_pos[0], target_pos[1], c='green', marker='*', s=300, zorder=5, label='Target')
    # Draw target zone (5cm radius)
    circle = plt.Circle((target_pos[0], target_pos[1]), 0.05, color='green', fill=False, 
                         linestyle='--', linewidth=2, label='Target Zone (5cm)')
    ax.add_patch(circle)
    
    # Calculate final distance
    final_dist = np.linalg.norm(obj_traj[-1] - np.array(target_pos))
    
    # Labels and title
    status = "SUCCESS" if success else "FAILED"
    status_color = "green" if success else "red"
    ax.set_title(f'Episode {episode_num + 1} - {status}\nFinal Distance to Target: {final_dist:.3f}m', 
                 fontsize=14, color=status_color)
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Set axis limits with some padding
    all_x = np.concatenate([ee_traj[:, 0], obj_traj[:, 0], [target_pos[0]]])
    all_y = np.concatenate([ee_traj[:, 1], obj_traj[:, 1], [target_pos[1]]])
    padding = 0.1
    ax.set_xlim(all_x.min() - padding, all_x.max() + padding)
    ax.set_ylim(all_y.min() - padding, all_y.max() + padding)
    
    # Save figure
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    fig_path = os.path.join(plots_dir, f"episode_{episode_num + 1}_trajectory.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved trajectory plot: {fig_path}")


def evaluate_single_model(model_file: str, args) -> dict:
    """Evaluate a single model and return results."""
    step_count = get_steps(model_file)
    model_name = os.path.splitext(os.path.basename(model_file))[0]
    obs_type = "image" if args.mode == "image" else "state"
    
    print(f"\nEvaluating model: {model_file}")
    
    # Find VecNormalize stats file
    vecnorm_path = find_vecnormalize_file(model_file)
    
    # Create environment with VecNormalize
    env = create_eval_env(obs_type, vecnorm_path)
    
    try:
        model = PPO.load(model_file)
    except Exception as e:
        print(f"Failed to load model {model_file}: {e}")
        env.close()
        return None
    
    # Create output directory
    model_output_dir = os.path.join(args.output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)
    
    success_count = 0
    total_rewards = []
    total_steps = []
    
    for episode in range(args.episodes):
        print(f"  Episode {episode + 1}/{args.episodes}")
        
        success, episode_reward, steps = run_episode(
            env, model, episode, model_output_dir, args.save_video
        )
        
        total_steps.append(steps)
        total_rewards.append(episode_reward)
        
        if success:
            success_count += 1
            print(f"    Result: Success (Target Reached) - Reward: {episode_reward:.2f}, Steps: {steps}")
        else:
            print(f"    Result: Failed (Time Limit) - Reward: {episode_reward:.2f}, Steps: {steps}")
    
    env.close()
    
    success_rate = (success_count / args.episodes) * 100
    avg_reward = np.mean(total_rewards)
    avg_steps = np.mean(total_steps)
    
    print(f"  Summary for {model_name}: Success Rate: {success_rate:.2f}%, Avg Reward: {avg_reward:.2f}")
    
    return {
        "name": model_name,
        "steps": step_count,
        "success_rate": success_rate,
        "avg_reward": avg_reward,
        "avg_steps": avg_steps
    }


def print_summary(results: dict):
    """Print final evaluation summary."""
    print("\n" + "=" * 100)
    print("FINAL EVALUATION SUMMARY")
    print("=" * 100)
    
    sorted_keys = sorted(results.keys(), key=lambda k: results[k]["steps"] if results[k]["steps"] > 0 else k)
    
    print(f"{'Model Name':<50} | {'Steps':<10} | {'Success Rate':<15} | {'Avg Reward':<15}")
    print("-" * 100)
    for name in sorted_keys:
        stats = results[name]
        print(f"{name:<50} | {stats['steps']:<10} | {stats['success_rate']:6.2f}%        | {stats['avg_reward']:6.2f}")


def evaluate(args):
    """Main evaluation function."""
    obs_type = "image" if args.mode == "image" else "state"
    print(f"Evaluation mode: {args.mode} (obs_type={obs_type})")
    
    # Get models to evaluate
    model_path = get_model_path(args)
    models_to_evaluate = get_models_to_evaluate(model_path, args.mode)
    
    if not models_to_evaluate:
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Evaluate all models (each creates its own env with matching VecNormalize)
    results = {}
    for model_file in models_to_evaluate:
        result = evaluate_single_model(model_file, args)
        if result:
            results[result["name"]] = result
    
    # Print summary
    if results:
        print_summary(results)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate RL agent for Push Task")
    parser.add_argument("--mode", type=str, default="state", choices=["state", "image"],
                        help="Observation mode: 'state' or 'image'")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to model file or directory containing checkpoints")
    parser.add_argument("--episodes", type=int, default=20,
                        help="Number of episodes to evaluate per model")
    parser.add_argument("--output-dir", type=str, default="evaluation_results",
                        help="Directory to save results")
    parser.add_argument("--save-video", action="store_true",
                        help="Save video recordings of evaluation episodes")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
