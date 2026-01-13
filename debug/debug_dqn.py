import torch
import numpy as np
import gymnasium as gym
from utils.wrappers import make_atari_env
from agents.dqn_agent import DQNAgent

# 伪造一个 args 类
class Args:
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    dqn_type = 'dqn'
    gamma = 0.99
    batch_size = 32  # 必须是 32
    update_freq = 1000
    epsilon_start = 1.0
    epsilon_final = 0.01
    epsilon_decay = 10000
    lr = 1e-4
    hidden_dim_dqn = 512

def check_dqn():
    print("=== 开始 DQN 诊断程序 ===")
    
    # 1. 环境检查
    env_name = "ALE/Breakout-v5"
    # 强制单环境同步模式，方便调试
    env = make_atari_env(env_name, num_envs=1, seed=42, is_training=True)
    
    print(f"[Check 1] Environment: {env_name}")
    obs, _ = env.reset()
    print(f"  Obs Shape: {obs.shape}") # 预期 (1, 4, 84, 84) 或 (4, 84, 84)
    print(f"  Obs Dtype: {obs.dtype}") # 预期 uint8
    print(f"  Obs Max: {np.max(obs)}") # 预期 0-255 (通常是0，因为刚开始是黑屏)
    
    # 2. Agent 初始化检查
    writer = None # Mock writer
    agent = DQNAgent(env, Args(), writer)
    print("[Check 2] Agent Initialized")
    
    # 3. 前向传播检查 (Forward Pass)
    state = torch.FloatTensor(obs).to(Args.device)
    # 如果是 (4, 84, 84) 补充 batch 维度
    if len(state.shape) == 3:
        state = state.unsqueeze(0)
    
    # 模拟数据预处理
    state_norm = state / 255.0
    
    with torch.no_grad():
        q_values = agent.q_net(state_norm)
    
    print(f"[Check 3] Q-Network Output")
    print(f"  Q-Values Shape: {q_values.shape}") # 预期 (1, 4)
    print(f"  Q-Values Mean: {q_values.mean().item():.4f}") # 预期 接近0 (初始化)
    
    # 4. 发球检查 (Fire Check)
    print("[Check 4] Fire Reset Logic")
    # 强制执行几个 FIRE 动作
    total_reward = 0
    for i in range(100):
        # 动作 1 是 FIRE
        # 【修正】对于单环境，直接传整数，不要传数组
        action = 1 if i < 2 else agent.env.action_space.sample()
        next_obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        if done or truncated:
            env.reset()
            
    print(f"  100步随机游走 (含Fire) 总奖励: {total_reward}")
    if total_reward == 0:
        print("  [WARNING] 警告：跑了100步全是0分，FireReset可能没生效，或者脸太黑。")
    else:
        print("  [OK] 能够获得奖励，环境正常。")

    # 5. 学习步检查 (Learn Step)
    print("[Check 5] Learning Step")
    # 强行塞数据进 Buffer
    for _ in range(50):
        agent.buffer.add(obs[0], 1, 1.0, obs[0], 0) # 假数据
    
    # 运行一次 learn
    try:
        agent.learn()
        print("  [OK] agent.learn() 运行成功，没有报错。")
    except Exception as e:
        print(f"  [ERROR] agent.learn() 报错: {e}")

    env.close()
    print("=== 诊断结束 ===")

if __name__ == "__main__":
    check_dqn()