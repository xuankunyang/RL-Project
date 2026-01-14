import torch
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

# 计算上一级文件夹的路径（os.path.dirname 向上走一级）
parent_dir = os.path.dirname(current_dir)

# 添加到 sys.path（如果有多个层级，可以多用几次 dirname）
sys.path.append(parent_dir)

from agents.ppo_agent import PPOAgent  # 确保能导入你的 Agent 类
import argparse
from types import SimpleNamespace

# === 配置 ===
ENV_NAME = "HalfCheetah-v4"
MODEL_PATH = "results/MuJoCo/HalfCheetah-v4/PPO_Standard/lra5e-05_lrc0.0002_clp0.2_sd42_hd256_env16_20260112-054627/models/final_model.pth" # 【替换】你的最佳模型路径
HIDDEN_DIM = 256  # 【替换】跟你训练时一致
SEED = 42

def plot_gait():
    # 1. 初始化环境和 Agent
    env = gym.make(ENV_NAME, render_mode="rgb_array")
    
    # 构造一个假的 args 对象来初始化 Agent
    args = SimpleNamespace(
        device="cpu", 
        gamma=0.99, lr=3e-4, lr_actor=None, lr_critic=None, 
        ppo_clip=0.2, ppo_epochs=10, mini_batch_size=64, 
        hidden_dim_ppo=HIDDEN_DIM, vf_coef=0.5, ent_coef=0.01, 
        horizon=2048, gae_lambda=0.95
    )
    
    # 初始化 Agent 并加载权重 (注意：这里只加载 Policy 部分即可，因为我们只推理)
    # 如果你的保存逻辑是保存整个 state_dict，请相应调整
    agent = PPOAgent(env, args, writer=None)
    
    # 加载模型
    # 注意：根据你的 save 代码，如果是 agent.policy.state_dict()：
    agent.policy.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    
    # 2. 运行推理
    obs, _ = env.reset(seed=SEED)
    actions = []
    
    # 只需要前 100 步就能看清步态
    steps = 100
    for _ in range(steps):
        # eval_mode=True 确保输出均值，去燥
        action, _, _ = agent.select_action(obs, eval_mode=True)
        obs, _, done, _, _ = env.step(action)
        actions.append(action)
        if done: break
            
    actions = np.array(actions)
    
    # 3. 绘图
    # HalfCheetah 动作维度通常是 6维: 
    # [bthigh, bshin, bfoot, fthigh, fshin, ffoot] (顺序可能随版本微调，但通常前3后3)
    # 我们取 Index 0 (后大腿) 和 Index 3 (前大腿) 来对比
    
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 5), dpi=150)
    
    x = np.arange(len(actions))
    plt.plot(x, actions[:, 0], label='Back Thigh Torque', linewidth=2, color='#d62728') # 红色
    plt.plot(x, actions[:, 3], label='Front Thigh Torque', linewidth=2, color='#1f77b4') # 蓝色
    
    plt.title(f"Learned Gait Pattern (Action Trajectory) - {ENV_NAME}", fontsize=14)
    plt.xlabel("Timesteps", fontsize=12)
    plt.ylabel("Action Value (Torque)", fontsize=12)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    plt.savefig("results/MuJoCo/HalfCheetah-v4/PPO_Standard/analysis/gait_analysis.png")
    plt.show()

if __name__ == "__main__":
    plot_gait()