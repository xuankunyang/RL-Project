import gymnasium as gym
import numpy as np
import ale_py  # Register ALE environments
from gymnasium.wrappers import (
    AtariPreprocessing,
    TransformReward,
    FrameStackObservation
)


def make_atari_env(env_name):
    """
    创建一个经过预处理的 Atari 环境。
    包含：灰度化、缩放 (84x84)、4帧叠加、NoOp启动等。
    """
    env = gym.make(env_name, render_mode="rgb_array")
    
    # 官方推荐的 Atari 预处理
    env = AtariPreprocessing(
        env, 
        noop_max=30, 
        frame_skip=4, 
        screen_size=84, 
        terminal_on_life_loss=False,  # DQN 学完整 episode
        grayscale_obs=True,
        scale_obs=True  # 归一化到 [0, 1]
    )
    
    # 堆叠4帧，让Agent感知速度（FrameStack 的新替代）
    env = FrameStackObservation(env, stack_size=4)
    
    return env


def make_mujoco_env(env_name):
    # MuJoCo 通常需要 Observation 和 Reward 的归一化以获得更好的 PPO 性能
    env = gym.make(env_name, render_mode="rgb_array")
    
    # 归一化 Observation
    env = gym.wrappers.NormalizeObservation(env)
    
    # 归一化 Reward 和 Clip
    env = gym.wrappers.NormalizeReward(env)
    env = gym.wrappers.TransformReward(env, lambda r: np.clip(r, -10, 10))
    
    return env
