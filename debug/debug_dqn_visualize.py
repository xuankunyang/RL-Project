import gymnasium as gym
import numpy as np
import cv2  # 如果没有 cv2，用 matplotlib 也可以，这里默认用 cv2 保存图片
from utils.wrappers import make_atari_env
import os

def check_visual():
    print("=== Breakout 视觉诊断 ===")
    
    # 1. 创建环境 (强制使用训练配置，包含 FireReset)
    # is_training=True 会开启 FireReset 和 LifeLoss
    env = make_atari_env("ALE/Breakout-v5", num_envs=1, seed=42, is_training=True)
    
    print(f"Action Meanings: {env.unwrapped.get_action_meanings()}")
    # 预期输出: ['NOOP', 'FIRE', 'RIGHT', 'LEFT']
    
    obs, _ = env.reset()
    
    # 保存初始画面
    # obs 是 (4, 84, 84)，我们需要取最后一帧
    frame_0 = obs[-1, :, :] 
    cv2.imwrite("debug_step_0.png", frame_0)
    print("已保存: debug_step_0.png (重置后的画面)")
    
    # 2. 运行 50 步 (约 200 帧)
    # 如果球发出来了，这时候应该能看到球在空中
    total_reward = 0
    print("开始运行 50 步...")
    
    for i in range(1, 51):
        # 随机动作
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        
        # 保存第 20 步和第 50 步的画面
        if i == 20:
            frame_20 = obs[-1, :, :]
            cv2.imwrite("debug_step_20.png", frame_20)
            print("已保存: debug_step_20.png")
        
        if i == 50:
            frame_50 = obs[-1, :, :]
            cv2.imwrite("debug_step_50.png", frame_50)
            print("已保存: debug_step_50.png")

        if done:
            print(f"在第 {i} 步检测到 Done (掉命)！说明球肯定发出来了！")
            break

    print(f"50步总奖励: {total_reward}")
    env.close()

if __name__ == "__main__":
    check_visual()