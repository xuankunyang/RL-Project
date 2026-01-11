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
    def make_env():
        # render_mode 设为 rgb_array 是为了防止服务器报错，同时允许后续 eval 时存视频
        env = gym.make(env_name, render_mode="rgb_array")
        
        # 1. 最先加 RecordEpisodeStatistics，记录原始分数为 "episode" -> "r"
        env = RecordEpisodeStatistics(env)
        
        # 2. ClipAction: 确保动作在 [-1, 1] 之间 (PPO 输出通常是高斯，需要截断)
        env = ClipAction(env)
        
        # 3. Normalize Observation (PPO 核心) + Clip
        env = NormalizeObservation(env)
        env = TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        
        # 4. Normalize Reward (PPO 核心) + Clip
        # 这一步之后，env.step() 返回的 reward 就会变成很小的值
        env = NormalizeReward(env)
        env = TransformReward(env, lambda r: np.clip(r, -10, 10))
        
        return env

    if num_envs > 1:
        # 推荐使用 SyncVectorEnv，MuJoCo 下速度通常优于 Async
        return gym.vector.SyncVectorEnv([make_env for _ in range(num_envs)])
    else:
        return make_env()
