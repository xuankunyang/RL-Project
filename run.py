import argparse
import os

# Configure MuJoCo for headless rendering
if os.name == 'posix' and 'DISPLAY' not in os.environ:
    print("Headless environment detected. Setting MUJOCO_GL='egl'")
    os.environ['MUJOCO_GL'] = 'egl'

import torch
import numpy as np
import gymnasium as gym
import pickle
import time
from gymnasium.wrappers import NormalizeObservation, RecordVideo

from utils.wrappers import make_atari_env, make_mujoco_env
from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent

# Try to import BEST_MODELS configuration
try:
    from configs.best_models import BEST_MODELS
except ImportError:
    print("Warning: Could not import BEST_MODELS from configs.best_models.")
    BEST_MODELS = {}

class DummyWriter:
    """Dummy SummaryWriter to avoid errors in Agent classes."""
    def add_scalar(self, *args, **kwargs):
        pass
    def add_histogram(self, *args, **kwargs):
        pass
    def close(self):
        pass

def load_obs_rms(env, obs_rms_path):
    """Load observation normalization statistics for PPO."""
    if not os.path.exists(obs_rms_path):
        print(f"Warning: obs_rms file not found at {obs_rms_path}. Running without normalization stats.")
        return

    print(f"Loading obs_rms from {obs_rms_path}...")
    with open(obs_rms_path, 'rb') as f:
        obs_rms = pickle.load(f)
    
    # Inject into env
    current = env
    found = False
    while hasattr(current, "env"):
        if isinstance(current, NormalizeObservation):
            current.obs_rms = obs_rms
            # Freeze stats during evaluation
            current.update = lambda x: None 
            found = True
            break
        current = current.env
    
    if found:
        print("Successfully loaded obs_rms.")
    else:
        print("Warning: NormalizeObservation wrapper not found in environment stack.")

def main():
    parser = argparse.ArgumentParser(description='RL Final Project - Evaluation & Visualization')
    
    # === Basic Settings ===
    parser.add_argument('--env_name', type=str, required=True, help='Gym environment name (e.g., ALE/Breakout-v5, Hopper-v4)')
    parser.add_argument('--algo', type=str, default=None, choices=['dqn', 'ppo'], help='Algorithm used for training (Optional, inferred from env)')
    parser.add_argument('--model_path', type=str, default=None, help='Path to the .pth model file (Optional if in BEST_MODELS)')
    parser.add_argument('--dqn_type', type=str, default='rainbow', choices=['dqn', 'double', 'dueling', 'rainbow'], help='DQN Variant (default: rainbow)')
    
    # === Evaluation Settings ===
    parser.add_argument('--episodes', type=int, default=5, help='Number of episodes to evaluate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
    parser.add_argument('--render', action='store_true', help='Enable rendering (window)')
    parser.add_argument('--save_video', action='store_true', help='Enable video recording (headless friendly)')
    parser.add_argument('--video_dir', type=str, default='videos', help='Directory to save videos')
    parser.add_argument('--sleep', type=float, default=0.0, help='Sleep time between steps (s) for slower visualization')
    parser.add_argument('--obs_rms_path', type=str, default=None, help='Path to obs_rms.pkl (Required for PPO)')

    # === Dummy Hyperparams (Required to initialize Agents) ===
    # These values don't affect evaluation but are needed for __init__
    parser.add_argument('--hidden_dim_dqn', type=int, default=512)
    parser.add_argument('--hidden_dim_ppo', type=int, default=256)

    args = parser.parse_args()

    # === Infer Algorithm if not provided ===
    if args.algo is None:
        if any(x in args.env_name for x in ['ALE', 'Breakout', 'Pong', 'SpaceInvaders', 'NoFrameskip']):
            args.algo = 'dqn'
            print(f"Inferred algorithm: DQN (for {args.env_name})")
        elif any(x in args.env_name for x in ['Hopper', 'Ant', 'HalfCheetah', 'Walker', 'Swimmer', 'Humanoid']):
            args.algo = 'ppo'
            print(f"Inferred algorithm: PPO (for {args.env_name})")
        else:
            parser.error(f"Could not infer algorithm for environment {args.env_name}. Please specify --algo.")

    # === Load Configuration from BEST_MODELS ===
    if args.model_path is None:
        if args.env_name in BEST_MODELS and args.algo in BEST_MODELS[args.env_name]:
            algo_config = BEST_MODELS[args.env_name][args.algo]
            
            # Select specific config based on algorithm variant
            config = None
            if args.algo == 'dqn':
                if args.dqn_type in algo_config:
                    config = algo_config[args.dqn_type]
                    print(f"Loading best configuration for {args.env_name} ({args.algo} - {args.dqn_type})...")
                else:
                    print(f"Error: Configuration for DQN variant '{args.dqn_type}' not found in BEST_MODELS.")
                    return
            else:
                # For PPO or others without sub-variants
                config = algo_config
                print(f"Loading best configuration for {args.env_name} ({args.algo})...")

            if config:
                args.model_path = config.get('model_path')
                
                # Load other optional configs
                if 'obs_rms_path' in config:
                    args.obs_rms_path = config['obs_rms_path']
                # dqn_type is already in args, but we could enforce it from config if needed
                if 'hidden_dim_dqn' in config:
                    args.hidden_dim_dqn = config['hidden_dim_dqn']
                if 'hidden_dim_ppo' in config:
                    args.hidden_dim_ppo = config['hidden_dim_ppo']
                    
                print(f"  Model Path: {args.model_path}")
                if args.obs_rms_path:
                    print(f"  Obs RMS Path: {args.obs_rms_path}")
        else:
            print(f"Error: No model_path provided and no configuration found for {args.env_name} / {args.algo} in BEST_MODELS.")
            return

    # === Validation ===
    if args.algo == 'dqn' and 'ALE' not in args.env_name and 'Breakout' not in args.env_name and 'Pong' not in args.env_name:
        print(f"Warning: Algorithm is DQN but environment {args.env_name} does not look like Atari.")
    if args.algo == 'ppo' and ('Hopper' not in args.env_name and 'Ant' not in args.env_name and 'HalfCheetah' not in args.env_name):
        print(f"Warning: Algorithm is PPO but environment {args.env_name} does not look like MuJoCo.")

    # === Environment Setup ===
    render_mode = None
    if args.save_video:
        render_mode = 'rgb_array'
    elif args.render:
        # Check for Display on Linux
        if os.name == 'posix' and 'DISPLAY' not in os.environ:
            print("Error: --render requested but no DISPLAY environment variable found (headless server?).")
            print("Try using --save_video instead.")
            return
        render_mode = 'human'

    print(f"Creating environment: {args.env_name} with render_mode={render_mode}")
    
    if args.algo == 'dqn':
        env = make_atari_env(args.env_name, num_envs=1, is_training=False, render_mode=render_mode)
    else:
        env = make_mujoco_env(args.env_name, num_envs=1, is_training=False, render_mode=render_mode)

    if args.save_video:
        video_folder = os.path.join(args.video_dir, f"{args.env_name}_{args.algo}_{time.strftime('%Y%m%d-%H%M%S')}")
        print(f"Recording video to {video_folder}")
        env = RecordVideo(
            env, 
            video_folder=video_folder,
            episode_trigger=lambda x: True, # Record all episodes
            disable_logger=True
        )

    # === Agent Setup ===
    writer = DummyWriter()
    
    if args.algo == 'dqn':
        agent = DQNAgent(env, args, writer)
    else:
        agent = PPOAgent(env, args, writer)

    # === Load Model ===
    print(f"Loading model from {args.model_path}...")
    try:
        agent.load(args.model_path)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # === Load Obs RMS (PPO) ===
    if args.algo == 'ppo':
        if args.obs_rms_path:
             load_obs_rms(env, args.obs_rms_path)
        else:
             # Try to infer path
             inferred_path = args.model_path.replace("model_", "obs_rms_").replace(".pth", ".pkl")
             if os.path.exists(inferred_path):
                 print(f"Inferred obs_rms path: {inferred_path}")
                 load_obs_rms(env, inferred_path)
             else:
                 # Check for final_obs_rms.pkl in the same directory
                 model_dir = os.path.dirname(args.model_path)
                 final_path = os.path.join(model_dir, "final_obs_rms.pkl")
                 if os.path.exists(final_path):
                      print(f"Found final_obs_rms.pkl at {final_path}")
                      load_obs_rms(env, final_path)
                 else:
                      print("Warning: No obs_rms_path provided and could not infer one. Performance might be poor if env was normalized.")
                      print("Note: Older models might not have saved obs_rms.pkl.")

    # === Evaluation Loop ===
    print(f"Starting evaluation for {args.episodes} episodes...")
    returns = []
    
    for ep in range(args.episodes):
        state, _ = env.reset(seed=args.seed + ep) # Different seed per episode
        done = False
        truncated = False
        episode_reward = 0
        step = 0
        
        while not (done or truncated):
            if args.algo == 'dqn':
                action = agent.select_action(state, steps_done=0, eval_mode=True)
            else:
                action, _, _ = agent.select_action(state, eval_mode=True)
                if isinstance(action, np.ndarray) and len(action.shape) > 1:
                    action = action[0]

            next_state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            state = next_state
            step += 1
            
            if args.render and args.sleep > 0:
                time.sleep(args.sleep)

        returns.append(episode_reward)
        print(f"Episode {ep+1}/{args.episodes}: Reward = {episode_reward:.2f}, Steps = {step}")

    mean_return = np.mean(returns)
    std_return = np.std(returns)
    print("="*40)
    print(f"Evaluation Finished")
    print(f"Mean Reward: {mean_return:.2f} +/- {std_return:.2f}")
    print("="*40)
    
    env.close()

if __name__ == "__main__":
    main()