import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.logger import HParam
from env.push_env import PushEnv
from config_loader import load_config, get_training_config, get_env_config
import os
import sys
import argparse
from datetime import datetime
import time
import numpy as np
import glob


def resolve_checkpoint_path(checkpoint_path: str) -> tuple:
    """
    Resolve checkpoint path to (model_path, vecnorm_path).
    
    Supports:
    - Directory: finds .zip and .pkl files inside
    - .zip file: finds corresponding .pkl file
    
    Returns:
        (model_path, vecnorm_path) tuple, vecnorm_path may be None
    """
    if checkpoint_path is None:
        return None, None
    
    # If it's a directory, find model files inside
    if os.path.isdir(checkpoint_path):
        # Find .zip files
        zip_files = glob.glob(os.path.join(checkpoint_path, "*.zip"))
        if not zip_files:
            raise ValueError(f"No .zip model files found in {checkpoint_path}")
        
        # Prefer non-checkpoint files (without _steps), otherwise use latest checkpoint
        final_models = [f for f in zip_files if "_steps" not in os.path.basename(f)]
        if final_models:
            model_path = final_models[0]
        else:
            # Sort by step count to get latest checkpoint
            def get_steps(f):
                import re
                match = re.search(r'_(\d+)_steps', f)
                return int(match.group(1)) if match else 0
            zip_files.sort(key=get_steps, reverse=True)
            model_path = zip_files[0]
        
        # Find .pkl files (VecNormalize stats)
        pkl_files = glob.glob(os.path.join(checkpoint_path, "*.pkl"))
        if pkl_files:
            # Prefer file with matching name or vecnormalize in name
            model_base = os.path.splitext(os.path.basename(model_path))[0]
            matching_pkl = [p for p in pkl_files if model_base in p or "vecnormalize" in p.lower()]
            vecnorm_path = matching_pkl[0] if matching_pkl else pkl_files[0]
        else:
            vecnorm_path = None
        
        print(f"Resolved directory {checkpoint_path}:")
        print(f"  Model: {model_path}")
        print(f"  VecNormalize: {vecnorm_path}")
        return model_path, vecnorm_path
    
    # If it's a file path
    if not checkpoint_path.endswith('.zip'):
        checkpoint_path += '.zip'
    
    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Model file not found: {checkpoint_path}")
    
    # Find corresponding .pkl file
    base_path = checkpoint_path.replace('.zip', '')
    possible_pkl_paths = [
        base_path + "_vecnormalize.pkl",
        base_path + ".pkl",
        os.path.join(os.path.dirname(checkpoint_path), "vecnormalize.pkl"),
    ]
    
    vecnorm_path = None
    for pkl_path in possible_pkl_paths:
        if os.path.exists(pkl_path):
            vecnorm_path = pkl_path
            break
    
    return checkpoint_path, vecnorm_path


class SaveVecNormalizeCallback(BaseCallback):
    """Callback to save VecNormalize statistics during training."""
    
    def __init__(self, save_freq: int, save_path: str, name_prefix: str, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps.pkl")
            if hasattr(self.training_env, 'save'):
                self.training_env.save(path)
                if self.verbose > 1:
                    print(f"Saved VecNormalize stats to {path}")
        return True


class ProgressBarCallback(BaseCallback):
    """
    Callback for displaying a progress bar with training metrics.
    Shows current step, total steps, ETA, and key training statistics.
    """
    
    def __init__(self, total_timesteps: int, update_freq: int = 1000, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.update_freq = update_freq
        self.start_time = None
        self.episode_rewards = []
        self.episode_lengths = []
        self.last_update_step = 0
        
    def _on_training_start(self) -> None:
        self.start_time = time.time()
        self.initial_timesteps = self.model.num_timesteps
        # Print initial empty lines for the status box
        print("\n" * 6)
        
    def _on_step(self) -> bool:
        # Collect episode info from infos
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])
        
        # Update progress bar at specified frequency
        if self.num_timesteps - self.last_update_step >= self.update_freq:
            self._update_progress_bar()
            self.last_update_step = self.num_timesteps
            
        return True
    
    def _update_progress_bar(self):
        current_steps = self.num_timesteps
        elapsed_time = time.time() - self.start_time
        steps_done = current_steps - self.initial_timesteps
        
        # Calculate progress
        progress = current_steps / self.total_timesteps
        progress_percent = progress * 100
        
        # Calculate ETA
        if steps_done > 0:
            steps_per_sec = steps_done / elapsed_time
            remaining_steps = self.total_timesteps - current_steps
            eta_seconds = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
            eta_str = self._format_time(eta_seconds)
            fps_str = f"{steps_per_sec:.0f}"
        else:
            eta_str = "N/A"
            fps_str = "N/A"
        
        # Calculate recent episode stats (last 100 episodes)
        if len(self.episode_rewards) > 0:
            recent_rewards = self.episode_rewards[-100:]
            recent_lengths = self.episode_lengths[-100:]
            avg_reward = np.mean(recent_rewards)
            avg_length = np.mean(recent_lengths)
            reward_str = f"{avg_reward:.1f}"
            length_str = f"{avg_length:.0f}"
        else:
            reward_str = "N/A"
            length_str = "N/A"
        
        # Create progress bar
        bar_length = 40
        filled_length = int(bar_length * progress)
        bar = "█" * filled_length + "░" * (bar_length - filled_length)
        
        # Format elapsed time
        elapsed_str = self._format_time(elapsed_time)
        
        # Clear screen and move cursor to top (using ANSI escape codes)
        # Move cursor up 6 lines and clear from cursor to end of screen
        sys.stdout.write("\033[6A\033[J")
        
        # Build multi-line status display
        status_lines = [
            f"╔{'═' * 58}╗",
            f"║  [{bar}] {progress_percent:5.1f}%  ║",
            f"╠{'═' * 58}╣",
            f"║  Steps: {current_steps:>12,} / {self.total_timesteps:<12,}          ║",
            f"║  FPS: {fps_str:>8}  │  Elapsed: {elapsed_str}  │  ETA: {eta_str}  ║",
            f"║  Reward: {reward_str:>8}  │  EpLen: {length_str:>8}  │  Episodes: {len(self.episode_rewards):<6} ║",
            f"╚{'═' * 58}╝",
        ]
        
        sys.stdout.write("\n".join(status_lines) + "\n")
        sys.stdout.flush()
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds into HH:MM:SS."""
        if seconds < 0 or seconds > 86400 * 30:  # Cap at 30 days
            return "N/A"
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def _on_training_end(self) -> None:
        # Print newline to move past progress bar
        print()
        total_time = time.time() - self.start_time
        print(f"Training completed in {self._format_time(total_time)}")
        if len(self.episode_rewards) > 0:
            print(f"Final avg reward (last 100 ep): {np.mean(self.episode_rewards[-100:]):.2f}")
            print(f"Total episodes: {len(self.episode_rewards)}")


def get_config(obs_type: str) -> dict:
    """Get training configuration based on observation type."""
    # 生成时间戳用于临时文件目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Load from config file
    train_cfg = get_training_config(obs_type)
    
    if obs_type == "image":
        base_dir = f"./tmp/{timestamp}_image"
        return {
            "base_dir": base_dir,
            "checkpoint_path": f"{base_dir}/checkpoints/",
            "tensorboard_log": f"{base_dir}/tensorboard/",
            "model_save_path": f"{base_dir}/ppo_push_robot_image",
            "model_name": "ppo_push_robot_image",
            "policy_type": "CnnPolicy",
            "progress_update_freq": train_cfg["progress_update_freq"],
            "ppo_kwargs": {
                "learning_rate": train_cfg["learning_rate"],
                "n_steps": train_cfg["n_steps"],
                "batch_size": train_cfg["batch_size"],
                "n_epochs": train_cfg["n_epochs"],
                "gamma": train_cfg["gamma"],
                "gae_lambda": train_cfg["gae_lambda"],
                "clip_range": train_cfg["clip_range"],
                "ent_coef": train_cfg["ent_coef"],
            }
        }
    else:
        base_dir = f"./tmp/{timestamp}_state"
        return {
            "base_dir": base_dir,
            "checkpoint_path": f"{base_dir}/checkpoints/",
            "tensorboard_log": f"{base_dir}/tensorboard/",
            "model_save_path": f"{base_dir}/ppo_push_robot",
            "model_name": "ppo_push_robot",
            "policy_type": "MlpPolicy",
            "progress_update_freq": train_cfg["progress_update_freq"],
            "ppo_kwargs": {
                "learning_rate": train_cfg["learning_rate"],
                "ent_coef": train_cfg["ent_coef"],
                "policy_kwargs": dict(net_arch=dict(
                    pi=train_cfg["net_arch_pi"], 
                    vf=train_cfg["net_arch_vf"]
                )),
            }
        }


def create_env(obs_type: str, n_envs: int, model_path: str = None, vecnorm_path: str = None):
    """Create and configure the training environment."""
    env_cfg = get_env_config()
    max_steps = env_cfg["max_episode_steps"]
    
    def make_env():
        env = PushEnv(render_mode=None, obs_type=obs_type)
        env = TimeLimit(env, max_episode_steps=max_steps)
        return env

    env = make_vec_env(make_env, n_envs=n_envs, vec_env_cls=SubprocVecEnv)

    # Apply VecNormalize for state observations
    if obs_type == "state":
        if vecnorm_path and os.path.exists(vecnorm_path):
            print(f"Loading VecNormalize stats from {vecnorm_path}")
            env = VecNormalize.load(vecnorm_path, env)
            env.training = True
            env.norm_reward = True
        else:
            print("Enabling fresh VecNormalize for state observation...")
            env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10., gamma=0.99)
    
    return env


def create_callbacks(args, config, env, n_envs: int) -> list:
    """Create training callbacks based on configuration."""
    callbacks = []
    
    # Add progress bar callback
    progress_callback = ProgressBarCallback(
        total_timesteps=args.timesteps,
        update_freq=config.get("progress_update_freq", 32768),
        verbose=1
    )
    callbacks.append(progress_callback)
    
    if args.save_freq is not None:
        os.makedirs(config["checkpoint_path"], exist_ok=True)
        
        checkpoint_callback = CheckpointCallback(
            save_freq=max(args.save_freq // n_envs, 1),
            save_path=config["checkpoint_path"],
            name_prefix=config["model_name"],
        )
        callbacks.append(checkpoint_callback)
        print(f"Checkpoint saving enabled: every {args.save_freq} steps to {config['checkpoint_path']}")
        
        if args.obs_type == "state" and isinstance(env, VecNormalize):
            norm_callback = SaveVecNormalizeCallback(
                save_freq=max(args.save_freq // n_envs, 1),
                save_path=config["checkpoint_path"],
                name_prefix=config["model_name"],
            )
            callbacks.append(norm_callback)
    else:
        print("Checkpoint saving disabled (use --save-freq N to enable)")
    
    return callbacks


def train(args):
    """Main training function."""
    print(f"Training with observation type: {args.obs_type}")
    
    # Get configuration
    config = get_config(args.obs_type)
    
    # Resolve checkpoint path (supports directory or file)
    model_path, vecnorm_path = resolve_checkpoint_path(args.load_checkpoint)
    
    # Create environment
    env = create_env(args.obs_type, args.n_envs, model_path, vecnorm_path)
    
    # Create callbacks
    callbacks = create_callbacks(args, config, env, args.n_envs)
    
    # Create or load model
    if model_path:
        print(f"Loading model from checkpoint: {model_path}")
        model = PPO.load(model_path, env=env, tensorboard_log=config["tensorboard_log"])
        reset_num_timesteps = False
        print("Model loaded successfully. Resuming training...")
    else:
        print(f"Creating PPO model with {config['policy_type']}...")
        model = PPO(
            config["policy_type"],
            env,
            verbose=0,  # Reduce verbosity to avoid cluttering progress bar
            tensorboard_log=config["tensorboard_log"],
            **config["ppo_kwargs"],
            device="cpu"
        )
        reset_num_timesteps = True

    # Train
    print(f"Training for {args.timesteps} timesteps...")
    print(f"Progress bar will update every ~32k steps. TensorBoard logs available at:")
    print(f"  tensorboard --logdir {config['tensorboard_log']}")
    print("-" * 80)
    model.learn(total_timesteps=args.timesteps, callback=callbacks, reset_num_timesteps=reset_num_timesteps)

    # Save final model to tmp directory (same as checkpoints)
    os.makedirs(config["base_dir"], exist_ok=True)
    model.save(config["model_save_path"])
    print(f"Model saved as {config['model_save_path']}")

    # Save VecNormalize stats
    if args.obs_type == "state" and isinstance(env, VecNormalize):
        env.save(config["model_save_path"] + "_vecnormalize.pkl")
        print(f"VecNormalize stats saved as {config['model_save_path']}_vecnormalize.pkl")

    env.close()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train RL agent for Push Task")
    parser.add_argument("--obs-type", type=str, default="state", choices=["state", "image"],
                        help="Observation type: 'state' or 'image'")
    parser.add_argument("--timesteps", type=int, default=1000000,
                        help="Total training timesteps")
    parser.add_argument("--save-freq", type=int, default=None,
                        help="Checkpoint save frequency in steps (disabled if not set)")
    parser.add_argument("--load-checkpoint", type=str, default=None,
                        help="Path to checkpoint file (.zip) or directory containing model files")
    parser.add_argument("--n-envs", type=int, default=12,
                        help="Number of parallel environments")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
