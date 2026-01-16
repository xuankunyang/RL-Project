import torch
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from agents.dqn_agent import DQNAgent
from utils.wrappers import make_atari_env
import argparse
from types import SimpleNamespace
import os

def get_env_config(env_name):
    """Get configuration for a specific environment."""
    configs = {
        'Breakout-v5': {
            'env_name': "ALE/Breakout-v5",
            'model_path': "results/Atari/ALE/Breakout-v5/DQN_Vanilla/lr0.0001_uf1000_sd42_hd256_bs32_env16_20260113-033247/models/final_model.pth",
            'dqn_type': "dqn",
            'hidden_dim': 256,
            'warmup_steps': 50  # Steps to play before generating saliency
        },
        'Pong-v5': {
            'env_name': "ALE/Pong-v5",
            'model_path': "results/Atari/ALE/Pong-v5/DQN_Rainbow/lr0.0001_uf2000_sd42_hd512_bs32_env16_20260115-070355/models/final_model.pth",  # TODO: Update with your Pong model path
            'dqn_type': "rainbow",
            'hidden_dim': 512,
            'warmup_steps': 100  # Pong might need more steps to get to an interesting state
        }
    }
    return configs.get(env_name, configs['Breakout-v5'])

def generate_saliency(env_name="Breakout-v5", model_path=None, device="cuda:0"):
    """Generate saliency map for a given environment."""

    # 1. Get environment configuration
    config = get_env_config(env_name)
    if model_path is None:
        model_path = config['model_path']

    print(f"Generating saliency map for {env_name}")
    print(f"Using model: {model_path}")

    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Warning: Model path {model_path} does not exist!")
        print("Please provide a valid model path using --model_path")
        return

    # Construct agent args
    args = SimpleNamespace(
        device=device, gamma=0.99, batch_size=32, lr=1e-4,
        dqn_type=config['dqn_type'], hidden_dim_dqn=config['hidden_dim'],
        epsilon_start=0.01, epsilon_final=0.01, epsilon_decay=1, # Eval mode
        learning_start=0, buffer_size=1000, update_freq=1000
    )

    # Initialize environment (single environment)
    env = make_atari_env(config['env_name'], num_envs=1, is_training=False)
    agent = DQNAgent(env, args, writer=None)
    agent.q_net.load_state_dict(torch.load(model_path, map_location=device))
    agent.q_net.eval()

    # 2. Get an interesting state
    obs, _ = env.reset()
    warmup_steps = config['warmup_steps']

    print(f"Warming up for {warmup_steps} steps...")

    # Play for warmup steps to get to an interesting game state
    for step in range(warmup_steps):
        action = agent.select_action(obs, 0, eval_mode=True)
        obs, reward, done, truncated, _ = env.step(action)
        if done or truncated:
            obs, _ = env.reset()
            print(f"Episode ended at step {step}, resetting...")

    print("Generating saliency map from current state...")

    # 3. Compute Saliency
    # obs shape: (4, 84, 84), convert to tensor and enable gradients
    state_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device) / 255.0
    state_tensor.requires_grad = True

    q_values = agent.q_net(state_tensor)
    best_action = q_values.argmax()
    best_q = q_values[0, best_action]

    # Backward pass to get gradients
    best_q.backward()

    # Take absolute value of gradients as saliency
    # shape: (1, 4, 84, 84) -> take max across channels
    saliency = state_tensor.grad.data.abs().squeeze(0) # (4, 84, 84)
    saliency, _ = torch.max(saliency, dim=0) # (84, 84) collapse channels
    saliency = saliency.cpu().numpy()

    # 4. Plotting
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Show raw observation (last frame)
    ax[0].imshow(obs[3], cmap='gray')
    ax[0].set_title(f"{env_name} - Raw Observation (Last Frame)")
    ax[0].axis('off')

    # Show Saliency Map
    ax[1].imshow(obs[3], cmap='gray', alpha=0.5)
    im = ax[1].imshow(saliency, cmap='jet', alpha=0.7)  # Overlay heatmap
    ax[1].set_title(f"{env_name} - Saliency Map (Attention)")
    ax[1].axis('off')

    # Add colorbar
    plt.colorbar(im, ax=ax[1], shrink=0.8)

    # Save plot
    output_filename = f"saliency_map_{env_name.replace('-', '_').lower()}.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Saliency map saved as: {output_filename}")

    # Clean up
    env.close()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Generate saliency maps for DQN agents in Atari environments')
    parser.add_argument('--env', type=str, default='Pong-v5',
                        choices=['Breakout-v5', 'Pong-v5'],
                        help='Environment name (default: Pong-v5)')
    parser.add_argument('--model_path', type=str, help='Path to the trained model (optional, uses default for environment)')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to run on (default: cuda:0)')

    args = parser.parse_args()

    generate_saliency(
        env_name=args.env,
        model_path=args.model_path,
        device=args.device
    )

if __name__ == "__main__":
    main()
