import argparse
import os
import torch
import numpy as np
import random
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

# 导入我们定义的模块
from utils.wrappers import make_atari_env, make_mujoco_env
from agents.dqn_agent import DQNAgent 
from agents.ppo_agent import PPOAgent 

def set_seed(seed, env):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    # env.action_space.seed(seed)
    # env.observation_space.seed(seed)

def evaluate(agent, env_name, algo, seed, episodes=5):
    """
    Evaluation loop
    """
    if algo == 'dqn':
        eval_env = make_atari_env(env_name)
    else:
        eval_env = make_mujoco_env(env_name)
    
    # set_seed(seed + 100, eval_env)
    
    returns = []
    for _ in range(episodes):
        state, _ = eval_env.reset()
        done = False
        truncated = False
        episode_reward = 0
        while not (done or truncated):
            action = agent.select_action(state, steps_done=0, eval_mode=True)
            if isinstance(action, tuple): action = action[0] # Handle PPO return format if needed
            
            # PPO select_action returns (action, log_prob, value)
            # DQN select_action returns action_int
            
            next_state, reward, done, truncated, _ = eval_env.step(action)
            episode_reward += reward
            state = next_state
        returns.append(episode_reward)
    
    eval_env.close()
    return np.mean(returns), np.std(returns)

def main():
    parser = argparse.ArgumentParser(description='RL Final Project')
    
    # === 基础设置 ===
    parser.add_argument('--env_name', type=str, default='BreakoutNoFrameskip-v4', help='Gym environment name')
    parser.add_argument('--algo', type=str, default='dqn', choices=['dqn', 'ppo'], help='Algorithm to use')
    # DQN Variants
    parser.add_argument('--dqn_type', type=str, default='dqn', choices=['dqn', 'double', 'dueling', 'rainbow'], help='DQN Variant')
    
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device (cuda:0, cuda:1, cpu)')
    parser.add_argument('--total_timesteps', type=int, default=1000000, help='Total training steps')
    parser.add_argument('--exp_name', type=str, default='default_exp', help='Experiment name for logging')
    parser.add_argument('--eval_freq', type=int, default=10000, help='Evaluation frequency')

    # === 超参数 (用于 Grid Search) ===
    parser.add_argument('--hidden_dim_dqn', type=int, default=512, help='Hidden dimension for DQNs')
    parser.add_argument('--hidden_dim_ppo', type=int, default=256, help='Hidden dimension for PPOs')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate (Shared default)')
    parser.add_argument('--lr_actor', type=float, default=None, help='Actor Learning rate (if None, use --lr)')
    parser.add_argument('--lr_critic', type=float, default=None, help='Critic Learning rate (if None, use --lr)')
    parser.add_argument('--ppo_clip', type=float, default=0.2, help='PPO Clip range')
    
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    
    args = parser.parse_args()

    # Default LRs logic
    if args.lr_actor is None: args.lr_actor = args.lr
    if args.lr_critic is None: args.lr_critic = args.lr

    # 1. 初始化 Log 目录
    # 如果是 DQN，在 log 中加入 type
    if args.algo == 'dqn':
        algo_name = f"{args.algo}_{args.dqn_type}"
    else:
        # PPO: Include Clip info if non-standard
        algo_name = f"{args.algo}"
        if args.ppo_clip > 0.5: # Assuming > 0.5 means "No Clip" experiment
             algo_name += "_noclip"
        
    log_dir = f"results/{algo_name}_{args.env_name}_{args.exp_name}_{args.seed}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    writer = SummaryWriter(log_dir)
    print(f"Starting training on {args.device} | Log dir: {log_dir}")

    # 2. 环境选择与 Agent 初始化
    if args.algo == 'dqn':
        env = make_atari_env(args.env_name)
        set_seed(args.seed, env)
        agent = DQNAgent(env, args, writer)
    elif args.algo == 'ppo':
        env = make_mujoco_env(args.env_name)
        set_seed(args.seed, env)
        agent = PPOAgent(env, args, writer)
    
    # 3. Training Loop
    state, _ = env.reset(seed=args.seed)
    current_ep_reward = 0
    
    for global_step in range(1, args.total_timesteps + 1):
        
        # --- Action Selection ---
        if args.algo == 'dqn':
            action = agent.select_action(state, global_step)
            # DQN specific: execution
            next_state, reward, done, truncated, _ = env.step(action)
            current_ep_reward += reward
            
            # Store
            agent.buffer.add(state, action, reward, next_state, done)
            
            state = next_state
            if done or truncated:
                writer.add_scalar("Train/EpisodeReward", current_ep_reward, global_step)
                current_ep_reward = 0
                state, _ = env.reset()
            
            # Train
            agent.learn()
            
        elif args.algo == 'ppo':
            # PPO specific: Rollout collection
            action, log_prob, value = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            current_ep_reward += reward
            
            # Store in RolloutBuffer
            agent.buffer.add(state, action, log_prob, reward, done, value)
            
            state = next_state
            
            if done or truncated:
                writer.add_scalar("Train/EpisodeReward", current_ep_reward, global_step)
                current_ep_reward = 0
                state, _ = env.reset()
            
            # Update if buffer full
            if agent.buffer.full:
                # Value bootstrapping for the last state
                _, _, last_val = agent.select_action(state, eval_mode=True)
                agent.learn(last_val)

        # --- Logging ---
        if global_step % 1000 == 0:
            print(f"[{args.algo.upper()}] Step {global_step}/{args.total_timesteps}")

        # --- Evaluation ---
        if global_step % args.eval_freq == 0:
            mean_ret, std_ret = evaluate(agent, args.env_name, args.algo, args.seed)
            writer.add_scalar("Eval/MeanReward", mean_ret, global_step)
            print(f"Step {global_step} | Eval Reward: {mean_ret:.2f} +/- {std_ret:.2f}")
            
            # Save Model
            torch.save(agent.__dict__.get('q_net', agent.__dict__.get('policy')).state_dict(), 
                       os.path.join(log_dir, f"model_{global_step}.pth"))

    # 4. 保存最终模型
    torch.save(agent.__dict__.get('q_net', agent.__dict__.get('policy')).state_dict(), 
               os.path.join(log_dir, "final_model.pth"))
    
    env.close()
    writer.close()
    print("Training Finished!")

if __name__ == "__main__":
    main()
