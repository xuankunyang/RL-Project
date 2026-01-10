import torch
import gymnasium as gym
import mujoco
import ale_py

def check_setup():
    print("="*30)
    print("环境自检开始...")
    
    # 1. 检查 GPU
    if torch.cuda.is_available():
        print(f"[SUCCESS] GPU 检测成功: {torch.cuda.get_device_name(0)}")
        print(f"          GPU 数量: {torch.cuda.device_count()}")
    else:
        print("[WARNING] 未检测到 GPU，将使用 CPU 训练 (速度会很慢)!")

    # 2. 检查 Atari 环境 (Breakout-v5)
    try:
        env = gym.make("BreakoutNoFrameskip-v4", render_mode="rgb_array")
        env.reset()
        env.step(env.action_space.sample())
        env.close()
        print("[SUCCESS] Atari 环境 (Breakout) 加载成功")
    except Exception as e:
        print(f"[ERROR] Atari 环境加载失败: {e}")

    # 3. 检查 MuJoCo 环境 (HalfCheetah-v4)
    try:
        env = gym.make("HalfCheetah-v4", render_mode="rgb_array")
        env.reset()
        env.step(env.action_space.sample())
        env.close()
        print("[SUCCESS] MuJoCo 环境 (HalfCheetah) 加载成功")
    except Exception as e:
        print(f"[ERROR] MuJoCo 环境加载失败: {e}")
        print("提示: 可能是缺少系统库，尝试安装 libgl1-mesa-glx 或 libosmesa6-dev")

    print("="*30)

if __name__ == "__main__":
    check_setup()