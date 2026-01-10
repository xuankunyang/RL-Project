import gymnasium as gym
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
    # MuJoCo 通常只需要归一化 Observation，这里简单起见直接返回
    # 如果要做得更好，可以加 NormalizeObservation wrapper
    env = gym.make(env_name, render_mode="rgb_array")
    return env
