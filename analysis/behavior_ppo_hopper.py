import torch
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import argparse
import pickle
from types import SimpleNamespace

# Configure MuJoCo for headless rendering (Server support)
if os.name == 'posix' and 'DISPLAY' not in os.environ:
    print("Headless environment detected. Setting MUJOCO_GL='egl'")
    os.environ['MUJOCO_GL'] = 'egl'

# === 路径设置 ===
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from agents.ppo_agent import PPOAgent
from gymnasium.wrappers import NormalizeObservation

# Try to import BEST_MODELS
try:
    from configs.best_models import BEST_MODELS
except ImportError:
    BEST_MODELS = {}

# === 配置 ===
ENV_NAME = "Hopper-v4"
HIDDEN_DIM = 256 # Default
MODEL_PATH = ""
OBS_RMS_PATH = None

# 尝试自动从 BEST_MODELS 加载配置
if ENV_NAME in BEST_MODELS and "ppo" in BEST_MODELS[ENV_NAME]:
    print(f"Loading configuration from BEST_MODELS for {ENV_NAME}...")
    cfg = BEST_MODELS[ENV_NAME]["ppo"]
    MODEL_PATH = os.path.join(parent_dir, cfg.get("model_path", ""))
    _obs_rms = cfg.get("obs_rms_path", None)
    if _obs_rms:
        OBS_RMS_PATH = os.path.join(parent_dir, _obs_rms)
    HIDDEN_DIM = cfg.get("hidden_dim_ppo", 256)
else:
    # Fallback / Manual Config
    # TODO: 如果 BEST_MODELS 中没有，请在此处手动指定
    MODEL_PATH = "results/MuJoCo/Hopper-v4/PPO_Standard/models/final_model.pth" 
    OBS_RMS_PATH = None 
    HIDDEN_DIM = 512

SEED = 42

def load_obs_rms(env, obs_rms_path):
    """Load observation normalization statistics."""
    if not obs_rms_path or not os.path.exists(obs_rms_path):
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

def plot_hopper_gait():
    print(f"Analyzing Behavior for {ENV_NAME}...")
    
    # 1. 初始化环境
    env = gym.make(ENV_NAME, render_mode="rgb_array")
    
    # 添加 NormalizeObservation wrapper
    env = NormalizeObservation(env)
    
    # 加载 Obs RMS 统计数据 (关键步骤)
    if OBS_RMS_PATH:
        load_obs_rms(env, OBS_RMS_PATH)
    else:
        print("Warning: No OBS_RMS_PATH configured. Results might be incorrect if the model was trained with normalization.")

    # 2. 初始化 Agent
    args = SimpleNamespace(
        device="cpu", 
        gamma=0.99, lr=3e-4, lr_actor=None, lr_critic=None, 
        ppo_clip=0.2, ppo_epochs=10, mini_batch_size=64, 
        hidden_dim_ppo=HIDDEN_DIM, vf_coef=0.5, ent_coef=0.01, 
        horizon=2048, gae_lambda=0.95,
        algo='ppo', env_name=ENV_NAME # 补全 PPOAgent 可能需要的参数
    )
    
    agent = PPOAgent(env, args, writer=None)
    
    # 3. 加载模型
    if os.path.exists(MODEL_PATH):
        print(f"Loading model from {MODEL_PATH}")
        state_dict = torch.load(MODEL_PATH, map_location='cpu')
        agent.policy.load_state_dict(state_dict)
    else:
        print(f"Warning: Model path {MODEL_PATH} does not exist! Using random weights.")

    # 4. 运行推理
    obs, _ = env.reset(seed=SEED)
    actions = []
    
    # Hopper 跳跃周期较短，100步足够看清几个周期
    steps = 150
    for i in range(steps):
        action, _, _ = agent.select_action(obs, eval_mode=True)
        obs, _, done, truncated, _ = env.step(action)
        actions.append(action)
        
        if done or truncated:
            print(f"Hopper fell over at step {i+1}")
            break
            
    actions = np.array(actions)
    
    # 5. 绘图分析
    # Hopper Action Space (3维):
    # 0: Thigh (大腿)
    # 1: Leg (小腿)
    # 2: Foot (脚)
    
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 6), dpi=150)
    
    x = np.arange(len(actions))
    
    # 绘制三条关节曲线
    plt.plot(x, actions[:, 0], label='Thigh Joint', linewidth=2.5, color='#d62728', alpha=0.9) # 红
    plt.plot(x, actions[:, 1], label='Leg Joint', linewidth=2.5, color='#1f77b4', alpha=0.9)   # 蓝
    plt.plot(x, actions[:, 2], label='Foot Joint', linewidth=2.5, color='#2ca02c', alpha=0.9)  # 绿
    
    plt.title("Learned Gait Pattern (Action Trajectory) - Hopper-v4", fontsize=16, pad=15)
    plt.xlabel("Timesteps (Control Cycles)", fontsize=14)
    plt.ylabel("Action Value (Torque)", fontsize=14)
    
    # 标记周期性区域
    if len(actions) > 50:
        plt.axvspan(20, 80, color='gray', alpha=0.1, label='Stable Hopping Phase')
    
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 保存图片
    save_dir = "results/MuJoCo/Hopper-v4/PPO_Standard/analysis"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "hopper_gait.png")
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Analysis saved to {save_path}")
    # plt.show() # 如果在服务器上运行，注释掉这一行

if __name__ == "__main__":
    plot_hopper_gait()