import torch
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from agents.dqn_agent import DQNAgent
from utils.wrappers import make_atari_env
import argparse
from types import SimpleNamespace

def generate_saliency():
    # 1. 配置 & 加载模型
    env_name = "ALE/Breakout-v5"
    model_path = "results/Atari/ALE/Breakout-v5/DQN_Rainbow/lr0.0001_uf2000_sd42_hd512_bs32_env16_20260114-165851/models/final_model.pth" # 替换你的模型路径
    
    # 构造假 args
    args = SimpleNamespace(
        device="cuda:0", gamma=0.99, batch_size=32, lr=1e-4,
        dqn_type="rainbow", hidden_dim_dqn=256, 
        epsilon_start=0.01, epsilon_final=0.01, epsilon_decay=1, # Eval mode
        learning_start=0, buffer_size=1000, update_freq=1000
    )
    
    # 初始化环境 (单环境)
    env = make_atari_env(env_name, num_envs=1, is_training=False)
    agent = DQNAgent(env, args, writer=None)
    agent.q_net.load_state_dict(torch.load(model_path))
    agent.q_net.eval()
    
    # 2. 获取一个状态
    obs, _ = env.reset()
    # 玩几步直到发球
    for _ in range(50): 
        action = agent.select_action(obs, 0, eval_mode=True)
        obs, _, done, _, _ = env.step(action)
        if done: obs, _ = env.reset()
        
    # 3. 计算 Saliency
    # obs shape: (4, 84, 84), 需要转 tensor 并这就梯度
    state_tensor = torch.FloatTensor(obs).unsqueeze(0).to(args.device) / 255.0
    state_tensor.requires_grad = True
    
    q_values = agent.q_net(state_tensor)
    best_action = q_values.argmax()
    best_q = q_values[0, best_action]
    
    # 反向传播求梯度
    best_q.backward()
    
    # 取梯度的绝对值最大值作为显著性
    # shape: (1, 4, 84, 84) -> 取这一帧里4个channel的最大值，或者取最新的 channel
    saliency = state_tensor.grad.data.abs().squeeze(0) # (4, 84, 84)
    saliency, _ = torch.max(saliency, dim=0) # (84, 84) collapse channels
    saliency = saliency.cpu().numpy()
    
    # 4. 绘图
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
    # 显示原始画面 (取最后一帧)
    ax[0].imshow(obs[3], cmap='gray')
    ax[0].set_title("Raw Observation (Last Frame)")
    ax[0].axis('off')
    
    # 显示 Saliency
    ax[1].imshow(obs[3], cmap='gray', alpha=0.5)
    ax[1].imshow(saliency, cmap='jet', alpha=0.7) # 叠加热力图
    ax[1].set_title("Saliency Map (Attention)")
    ax[1].axis('off')
    
    plt.savefig("saliency_map.png")
    plt.show()

if __name__ == "__main__":
    generate_saliency()