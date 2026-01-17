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
    AtariPreprocessing, 
    FrameStackObservation
)


from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv

class FireResetWrapper(gym.Wrapper):
    """Breakout 专用：Reset 后自动 FIRE 发球"""
    def __init__(self, env):
        super().__init__(env)
        action_meanings = env.unwrapped.get_action_meanings()
        if len(action_meanings) < 3:
            self.has_fire = False
        else:
            self.has_fire = action_meanings[1] == 'FIRE'

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self.has_fire:
            obs, _, done, truncated, _ = self.env.step(1)
            if done or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info

def make_atari_env(env_name, num_envs=1, seed=42, is_training=True):
    """
    针对 Breakout：
    is_training=True:  开启 LifeLoss (掉命即Done), 开启 RewardClip (训练稳定)
    is_training=False: 关闭 LifeLoss (玩满全场), 关闭 RewardClip (看真实分数)
    """
    def make_env(rank):
        def _thunk():
            # 1. 禁用 v5 原生跳帧
            env = gym.make(
                env_name, 
                frameskip=1,                 
                repeat_action_probability=0.0 
            )
            
            # === 【关键点 1】记录原始分数 ===
            # 将 RecordEpisodeStatistics 放在最前面（所有处理之前）
            # 这样 info["episode"]["r"] 记录的就是 Atari 的原始真实得分（比如 400 分）
            # 注意：如果 terminal_on_life_loss=True，这里记录的是“单条命”的真实得分
            env = RecordEpisodeStatistics(env)
            
            # 2. Atari 预处理
            if 'Breakout' in env_name:
                env = AtariPreprocessing(
                    env, 
                    noop_max=30, 
                    frame_skip=4, 
                    screen_size=84, 
                    # 【关键点 2】Breakout 训练时掉命算死，评估时掉命不算死
                    terminal_on_life_loss=is_training, 
                    grayscale_obs=True,
                    scale_obs=False
                )
                env = FireResetWrapper(env)
            elif 'Pong' in env_name:
                env = AtariPreprocessing(
                    env, 
                    noop_max=30, 
                    frame_skip=4, 
                    screen_size=84, 
                    grayscale_obs=True,
                    scale_obs=False
                )
            
            else:
                raise ValueError(f"Unsupported env: {env_name}")
            
            # === 【关键点 3】只在训练时 Clip Reward ===
            # 评估时我们需要 agent 跑出真实分数，虽然 agent 内部还是基于 clipped 经验学的
            # 但这里环境返回的 reward 我们希望是真实的（方便 evaluate 函数累加）
            # 对于 Pong 不影响
            if is_training:
                env = TransformReward(env, lambda r: np.sign(r))
            
            env = FrameStackObservation(env, stack_size=4)
            return env
        return _thunk

    if num_envs > 1:
        return gym.vector.AsyncVectorEnv([make_env(i) for i in range(num_envs)])
    else:
        return make_env(0)()


def make_mujoco_env(env_name, num_envs=1, seed=42):
    """
    创建一个经过预处理的 MuJoCo 环境 (PPO 专用)
    这里需要修改一下！！！！！统一一下！！！！！
    """
    def make_env(rank):
        def _thunk():
            env = gym.make(env_name)  # 移除 render_mode 提速
            
            # 1. 记录真实分数 (必须在 Normalize 之前)
            env = RecordEpisodeStatistics(env)
            
            # 2. Clip Action
            env = ClipAction(env)

            if env_name.startswith("HalfCheetah"):
                # HalfCheetah: 强烈推荐 obs normalize + clip
                env = NormalizeObservation(env)
                env = TransformObservation(
                    env,
                    lambda obs: np.clip(obs, -10.0, 10.0),
                    env.observation_space,
                )
                
            elif env_name.startswith("Ant"):
                # Stable but high-dimensional, no obs clipping
                env = NormalizeObservation(env)
                # 【关键修正】Ant 的 Reward 很大 (1000~6000)，必须加 Reward Normalization
                # 否则 Critic 的 Loss 会极其巨大，导致梯度爆炸，Actor 根本学不到东西
                env = NormalizeReward(env)
                env = TransformReward(env, lambda r: np.clip(r, -10, 10))

            elif env_name.startswith("Hopper"):
                # Unstable system: normalize only, be conservative
                env = NormalizeObservation(env)

            else:
                raise ValueError(f"Unsupported env: {env_name}")

            # # 3. Normalize Observation
            # env = NormalizeObservation(env)
            # env = TransformObservation(
            #     env,
            #     lambda obs: np.clip(obs, -10, 10),
            #     env.observation_space
            # )

            return env
        return _thunk

    if num_envs > 1:
        # 使用 AsyncVectorEnv 更快
        return gym.vector.AsyncVectorEnv([make_env(i) for i in range(num_envs)])
    else:
        return make_env(0)()
