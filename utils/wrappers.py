import gymnasium as gym
import numpy as np
import ale_py  # Register ALE environments
from gymnasium.wrappers import (
    RecordEpisodeStatistics, 
    ClipAction, 
    NormalizeObservation, 
    TransformObservation, 
    NormalizeReward, 
    TransformReward, 
    AtariPreprocessing
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
    """
    创建一个经过预处理的 MuJoCo 环境 (PPO 专用)
    """
    def make_env():
        env = gym.make(env_name, render_mode="rgb_array")
        
        # 1. 记录真实分数 (必须在 Normalize 之前)
        env = RecordEpisodeStatistics(env)
        
        # 2. Clip Action
        env = ClipAction(env)
        
        # 3. Normalize Observation
        env = NormalizeObservation(env)
        
        # === 修复点：显式传入 observation_space ===
        # TransformObservation(env, func, observation_space)
        env = TransformObservation(
            env, 
            lambda obs: np.clip(obs, -10, 10), 
            env.observation_space
        )
        
        # 4. Normalize Reward
        env = NormalizeReward(env)
        env = TransformReward(
            env, 
            lambda r: np.clip(r, -10, 10)
        )
        # TransformReward 通常不需要传入 space，因为 reward 在 gym 里没有严格的 space 定义
        
        return env

    if num_envs > 0:
        return gym.vector.SyncVectorEnv([make_env for _ in range(num_envs)])
    else:
        return make_env()