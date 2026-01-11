import gymnasium as gym
import numpy as np
import ale_py  # Register ALE environments
from gymnasium.wrappers import (
    AtariPreprocessing,
    TransformReward,
    FrameStackObservation
)


from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv

def make_atari_env(env_name, num_envs=1, seed=42):
    """
    创建一个经过预处理的 Atari 环境。
    支持向量化环境 (VectorEnv)。
    """
    def make_env(rank):
        def _thunk():
            # Atari v5: frameskip handled by env, not AtariPreprocessing
            # Use frameskip parameter in gym.make() or let v5 defaults handle it
            env = gym.make(env_name, frameskip=4)  # v5 parameter
            env = AtariPreprocessing(
                env, 
                noop_max=30, 
                frame_skip=1,  # v5 already handles frame skip, set to 1
                screen_size=84, 
                terminal_on_life_loss=False,  
                grayscale_obs=True,
                scale_obs=False  # Return uint8 (0-255) to save RAM/Bandwidth
            )
            env = FrameStackObservation(env, stack_size=4)
            return env
        return _thunk

    if num_envs > 1:
        # 使用 AsyncVectorEnv 并行运行多个环境
        return AsyncVectorEnv([make_env(i) for i in range(num_envs)])
    else:
        return make_env(0)()


def make_mujoco_env(env_name, num_envs=1, seed=42):
    def make_env(rank):
        def _thunk():
            env = gym.make(env_name)  # Remove render_mode for speed
            env = gym.wrappers.NormalizeObservation(env)
            env = gym.wrappers.NormalizeReward(env)
            env = gym.wrappers.TransformReward(env, lambda r: np.clip(r, -10, 10))
            return env
        return _thunk

    if num_envs > 1:
        return AsyncVectorEnv([make_env(i) for i in range(num_envs)])
    else:
        return make_env(0)()
