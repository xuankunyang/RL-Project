import argparse
import os
import torch
import numpy as np
import random
from datetime import datetime
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter

# 导入我们定义的模块
from utils.wrappers import make_atari_env, make_mujoco_env
from agents.dqn_agent import DQNAgent 
from agents.ppo_agent import PPOAgent 

def set_seed(seed, env):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    # env.action_space.seed(seed)
    # env.observation_space.seed(seed)

def evaluate(agent, env_name, algo, seed, episodes=5):
    """
    Evaluation loop
    """
    if algo == 'dqn':
        eval_env = make_atari_env(env_name)
    else:
        # 使用未经 normalize 的环境进行评估
        eval_env = gym.make(env_name)
    
    # set_seed(seed + 100, eval_env)
    
    returns = []
    for _ in range(episodes):
        state, _ = eval_env.reset()
        done = False
        truncated = False
        episode_reward = 0
        while not (done or truncated):
            # Different select_action signatures for DQN vs PPO
            if algo == 'dqn':
                action = agent.select_action(state, steps_done=0, eval_mode=True)
            else:  # PPO
                action, _, _ = agent.select_action(state, eval_mode=True)
            
            next_state, reward, done, truncated, _ = eval_env.step(action)
            episode_reward += reward
            state = next_state
        returns.append(episode_reward)
    
    eval_env.close()
    return np.mean(returns), np.std(returns)

def main():
    parser = argparse.ArgumentParser(description='RL Final Project')
    
    # === 基础设置 ===
    parser.add_argument('--env_name', type=str, default='ALE/Pong-v5', help='Gym environment name')
    parser.add_argument('--algo', type=str, default='dqn', choices=['dqn', 'ppo'], help='Algorithm to use')
    # DQN Variants
    parser.add_argument('--dqn_type', type=str, default='dqn', choices=['dqn', 'double', 'dueling', 'rainbow'], help='DQN Variant')
    
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device (cuda:0, cuda:1, cpu)')
    parser.add_argument('--total_timesteps', type=int, default=1000000, help='Total training steps')
    parser.add_argument('--exp_name', type=str, default='default_exp', help='Experiment name for logging')
    parser.add_argument('--eval_freq', type=int, default=50000, help='Evaluation frequency')

    # === 超参数 (用于 Grid Search) ===
    # DQN
    parser.add_argument('--update_freq', type=int, default=1000, help='Target network update frequency')
    parser.add_argument('--train_freq', type=int, default=4, help='Training frequency (train every N steps)')
    parser.add_argument('--hidden_dim_dqn', type=int, default=512, help='Hidden dimension for DQNs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate (Shared default)')
    parser.add_argument('--epsilon_decay', type=float, default=500000, help='Epsilon decay steps')
    parser.add_argument('--epsilon_final', type=float, default=0.01, help='Minimum epsilon value')
    parser.add_argument('--epsilon_start', type=float, default=1.0, help='Initial epsilon value')
    parser.add_argument('--learning_start', type=int, default=10000, help='Learning start steps')

    # PPO
    parser.add_argument('--hidden_dim_ppo', type=int, default=256, help='Hidden dimension for PPOs')
    parser.add_argument('--lr_actor', type=float, default=3e-4, help='Actor Learning rate (if None, use --lr)')
    parser.add_argument('--lr_critic', type=float, default=1e-3, help='Critic Learning rate (if None, use --lr)')
    parser.add_argument('--ppo_clip', type=float, default=0.2, help='PPO Clip range')
    parser.add_argument('--ppo_epochs', type=int, default=10, help='Number of PPO epochs')
    parser.add_argument('--mini_batch_size', type=int, default=64, help='Mini batch size for PPO')
    parser.add_argument('--vf_coef', type=float, default=0.5, help='Value function coefficient for PPO')
    parser.add_argument('--ent_coef', type=float, default=0.01, help='Entropy coefficient for PPO')
    parser.add_argument('--horizon', type=int, default=2048, help='Number of steps to run for each environment')
    parser.add_argument('--gae_lambda', type=float, default=0.95, help='GAE lambda for PPO')

    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--num_envs', type=int, default=16, help='Number of parallel environments')
    
    args = parser.parse_args()

    # Default LRs logic
    if args.lr_actor is None: args.lr_actor = args.lr
    if args.lr_critic is None: args.lr_critic = args.lr

    # === Log Directory Construction ===
    # 1. Domain & Envnironment
    if args.algo == 'dqn':
        domain = "Atari"
        # Variant Name (Folder Level)
        if args.dqn_type == 'dqn':
             variant = "DQN_Vanilla"
        elif args.dqn_type == 'double':
             variant = "DQN_Double"
        elif args.dqn_type == 'dueling':
             variant = "DQN_Dueling"
        elif args.dqn_type == 'rainbow':
             variant = "DQN_Rainbow"
        else:
             variant = f"DQN_{args.dqn_type}"
             
        # Key Hyperparams for run folder
        hp_str = f"lr{args.lr}_uf{args.update_freq}_sd{args.seed}_bs{args.batch_size}_env{args.num_envs}"
    else:
        domain = "MuJoCo"
        # Variant Name (Folder Level)
        if args.ppo_clip > 0.5: 
             variant = "PPO_NoClip"
        else:
             variant = "PPO_Standard"
        
        hp_str = f"lra{args.lr_actor}_lrc{args.lr_critic}_clp{args.ppo_clip}_sd{args.seed}_env{args.num_envs}"

    # 2. Timestamp
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    
    # 3. Final Path: results/Domain/Env/Variant/Params_Timestamp
    log_dir = os.path.join("results", domain, args.env_name, variant, f"{hp_str}_{timestamp}")
    model_dir = os.path.join(log_dir, "models")
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # === Logging Setup ===
    # Setup simple logging to file and console
    import logging
    log_file = os.path.join(log_dir, "log.txt")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()
    
    # Log Arguments
    logger.info(f"Arguments: {args}")
    logger.info(f"Log Directory: {log_dir}")
    logger.info(f"Model Directory: {model_dir}")
    
    writer = SummaryWriter(log_dir)

    # 2. 环境选择与 Agent 初始化
    if args.algo == 'dqn':
        env = make_atari_env(args.env_name, num_envs=args.num_envs, seed=args.seed)
        # VectorEnv seeding handled inside or via seed arg if passed (AsyncVectorEnv doesn't inherently take seed in init list easily without wrapper, 
        # but our make_env(rank) architecture handles it if we passed seed, actually we didn't pass seed to make_env yet in wrappers.py properly?
        # Let's re-verify wrappers.py logic. make_env(rank) doesn't use seed. 
        # But gym.make usually seeds on reset. VectorEnv reset takes seed.
        # We will seed on reset.
        agent = DQNAgent(env, args, writer)
    elif args.algo == 'ppo':
        env = make_mujoco_env(args.env_name, num_envs=args.num_envs, seed=args.seed)
        agent = PPOAgent(env, args, writer)
    
    # 3. Training Loop
    state, _ = env.reset(seed=args.seed)
    current_ep_reward = np.zeros(args.num_envs)
    
    global_step = 0
    while global_step < args.total_timesteps:
        
        # --- Action Selection ---
        if args.algo == 'dqn':
            # DQN: vector or single env step
            action = agent.select_action(state, global_step)
            if args.num_envs == 1 and np.isscalar(action):
                 action = np.array([action])
            
            # RE-WRITING LOGIC TO HANDLE BOTH SINGLE AND VECTOR SAFELY
            
            if args.num_envs == 1:
                # Single Env flow
                # action is scalar (from updated select_action logic)
                 # Wait, updated select_action returns scalar if input (C, H, W).
                ns, r, d, t, _ = env.step(action)
                current_ep_reward[0] += r
                
                agent.buffer.add(state, action, r, ns, d or t)
                
                state = ns
                if d or t:
                    writer.add_scalar("Train/EpisodeReward", current_ep_reward[0], global_step)
                    logger.info(f"Step {global_step} | Train Reward: {current_ep_reward[0]:.2f}")
                    current_ep_reward[0] = 0
                    state, _ = env.reset()
                
                global_step += 1
            else:
                # Vector Flow
                # action is array
                next_state, reward, done, truncated, info = env.step(action)
                
                # Optimized add_batch
                real_next_states = next_state.copy()
                
                # Fix next_state for done envs AND log rewards
                for i in range(args.num_envs):
                    # Accumulate reward for this step
                    step_reward = reward[i]
                    current_ep_reward[i] += step_reward
                    
                    if done[i] or truncated[i]:
                        # Try fetch final obs
                        if "final_observation" in info:
                             real_next_states[i] = info["final_observation"][i]
                        elif "_final_observation" in info and info["_final_observation"][i]:
                             real_next_states[i] = info["final_observation"][i]
                        
                        # Log episode reward
                        ep_reward = current_ep_reward[i]
                        writer.add_scalar("Train/EpisodeReward", ep_reward, global_step + i)
                        
                        # Enhanced logging for debugging
                        if 'Pong' in args.env_name:
                            if abs(ep_reward) > 25:
                                logger.warning(f"[ANOMALY] Step {global_step} | Env {i} | Episode Reward: {ep_reward:.2f} | Last Step Reward: {step_reward:.2f}")
                            else:
                                logger.info(f"Step {global_step} | Env {i} | Episode Reward: {ep_reward:.2f}")
                        else:
                            logger.info(f"Step {global_step} | Env {i} | Episode Reward: {ep_reward:.2f}")
                        
                        # Clear for next episode
                        current_ep_reward[i] = 0
                
                # Bulk add
                agent.buffer.add_batch(state, action, reward, real_next_states, done | truncated)
                
                state = next_state
                global_step += args.num_envs
            
            # Train every train_freq steps (for efficiency)
            if global_step % args.train_freq == 0 and global_step >= args.learning_start:
                agent.learn()
            
        elif args.algo == 'ppo':
            # 1. Select Action
            action, log_prob, value = agent.select_action(state)
            
            # 2. Step Environment
            next_state, reward, done, truncated, info = env.step(action)
            
            # 3. 手动累积和记录真实 reward（类似 DQN 的方式）
            ep_rewards = []
            for i in range(args.num_envs):
                # 累积 reward 用于其他目的，但记录原始 episode reward
                current_ep_reward[i] += reward[i]

                if done[i] or truncated[i]:
                    # 使用 RecordEpisodeStatistics 记录的原始 episode reward
                    if "episode" in info and info["episode"]["r"][i] is not None:
                        ep_reward = info["episode"]["r"][i]
                    else:
                        # fallback to accumulated (should not happen with RecordEpisodeStatistics)
                        ep_reward = current_ep_reward[i]
                    ep_rewards.append(ep_reward)
                    logger.info(f"Step {global_step} | Env {i} | Episode Reward: {ep_reward:.2f}")
                    current_ep_reward[i] = 0

            # 记录平均 episode reward 到 TensorBoard
            if ep_rewards:
                avg_ep_reward = np.mean(ep_rewards)
                writer.add_scalar("Train/EpisodeReward", avg_ep_reward, global_step)
            
            # 4. 存储到 Buffer (只存 done，不存 truncated)
            agent.buffer.add_batch(state, action, log_prob, reward, done, value)
            
            state = next_state
            global_step += args.num_envs
            
            # 5. Update if buffer full
            if agent.buffer.full:
                _, _, last_values = agent.select_action(state, eval_mode=True)
                agent.learn(last_values)

        # --- Logging ---
        if global_step % 1000 == 0:
            pass

        # --- Evaluation (Linear check might skip if jumping steps) ---
        # Using >= last_eval_step + freq logic is better, but % works if we check range or allow jitter
        if global_step % args.eval_freq < args.num_envs: 
            # roughly triggers close to boundary
            # Evaluate using fresh single env to ensure consistent metrics?
            # Existing evaluate function creates its own env.
            mean_ret, std_ret = evaluate(agent, args.env_name, args.algo, args.seed)
            writer.add_scalar("Eval/MeanReward", mean_ret, global_step)
            logger.info(f"Step {global_step} | Eval Reward: {mean_ret:.2f} +/- {std_ret:.2f}")
            
            # Save Model
            torch.save(agent.__dict__.get('q_net', agent.__dict__.get('policy')).state_dict(), 
                       os.path.join(model_dir, f"model_{global_step}.pth"))

    # 4. 保存最终模型
    torch.save(agent.__dict__.get('q_net', agent.__dict__.get('policy')).state_dict(), 
               os.path.join(model_dir, "final_model.pth"))
    
    env.close()
    writer.close()
    logger.info("Training Finished!")

if __name__ == "__main__":
    main()
