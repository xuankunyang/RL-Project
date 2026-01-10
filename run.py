import argparse
import os
import torch
import numpy as np
import random
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

# 导入我们定义的模块
from utils.wrappers import make_atari_env, make_mujoco_env
# 这里的 Agent 还是空的，等下一步填入内容
# from agents.dqn_agent import DQNAgent 
# from agents.ppo_agent import PPOAgent 

def set_seed(seed, env):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def main():
    parser = argparse.ArgumentParser(description='RL Final Project')
    
    # === 基础设置 ===
    parser.add_argument('--env_name', type=str, default='BreakoutNoFrameskip-v4', help='Gym environment name')
    parser.add_argument('--algo', type=str, default='dqn', choices=['dqn', 'ppo'], help='Algorithm to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device (cuda:0, cuda:1, cpu)')
    parser.add_argument('--total_timesteps', type=int, default=1000000, help='Total training steps')
    parser.add_argument('--exp_name', type=str, default='default_exp', help='Experiment name for logging')

    # === 超参数 (用于 Grid Search) ===
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    
    args = parser.parse_args()

    # 1. 初始化 Log 目录
    log_dir = f"results/{args.algo}_{args.env_name}_{args.exp_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    writer = SummaryWriter(log_dir)
    print(f"Starting training on {args.device} | Log dir: {log_dir}")

    # 2. 环境选择与 Agent 初始化
    if args.algo == 'dqn':
        # Value-based: Atari
        env = make_atari_env(args.env_name)
        # TODO: 初始化 DQNAgent
        # agent = DQNAgent(env, args, writer)
        print("Running DQN training loop...")
        # agent.train()
        
    elif args.algo == 'ppo':
        # Policy-based: MuJoCo
        env = make_mujoco_env(args.env_name)
        # TODO: 初始化 PPOAgent
        # agent = PPOAgent(env, args, writer)
        print("Running PPO training loop...")
        # agent.train()

    # 3. 保存最终模型
    # torch.save(agent.model.state_dict(), os.path.join(log_dir, "final_model.pth"))
    env.close()
    writer.close()

if __name__ == "__main__":
    main()
