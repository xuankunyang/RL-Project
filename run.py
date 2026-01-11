import argparse
import os
import torch
import numpy as np
import random
from datetime import datetime
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
        eval_env = make_mujoco_env(env_name)
    
    # set_seed(seed + 100, eval_env)
    
    returns = []
    for _ in range(episodes):
        state, _ = eval_env.reset()
        done = False
        truncated = False
        episode_reward = 0
        while not (done or truncated):
            action = agent.select_action(state, steps_done=0, eval_mode=True)
            if isinstance(action, tuple): action = action[0] # Handle PPO return format if needed
            
            # PPO select_action returns (action, log_prob, value)
            # DQN select_action returns action_int
            
            next_state, reward, done, truncated, _ = eval_env.step(action)
            episode_reward += reward
            state = next_state
        returns.append(episode_reward)
    
    eval_env.close()
    return np.mean(returns), np.std(returns)

def main():
    parser = argparse.ArgumentParser(description='RL Final Project')
    
    # === 基础设置 ===
    parser.add_argument('--env_name', type=str, default='ALE/Breakout-v5', help='Gym environment name')
    parser.add_argument('--algo', type=str, default='dqn', choices=['dqn', 'ppo'], help='Algorithm to use')
    # DQN Variants
    parser.add_argument('--dqn_type', type=str, default='dqn', choices=['dqn', 'double', 'dueling', 'rainbow'], help='DQN Variant')
    
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device (cuda:0, cuda:1, cpu)')
    parser.add_argument('--total_timesteps', type=int, default=1000000, help='Total training steps')
    parser.add_argument('--exp_name', type=str, default='default_exp', help='Experiment name for logging')
    parser.add_argument('--eval_freq', type=int, default=50000, help='Evaluation frequency')

    # === 超参数 (用于 Grid Search) ===
    parser.add_argument('--update_freq', type=int, default=1000, help='Target network update frequency')
    parser.add_argument('--train_freq', type=int, default=4, help='Training frequency (train every N steps)')
    parser.add_argument('--hidden_dim_dqn', type=int, default=512, help='Hidden dimension for DQNs')
    parser.add_argument('--hidden_dim_ppo', type=int, default=256, help='Hidden dimension for PPOs')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate (Shared default)')
    parser.add_argument('--lr_actor', type=float, default=None, help='Actor Learning rate (if None, use --lr)')
    parser.add_argument('--lr_critic', type=float, default=None, help='Critic Learning rate (if None, use --lr)')
    parser.add_argument('--ppo_clip', type=float, default=0.2, help='PPO Clip range')
    
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
        # Workaround: Disable VectorEnv for PPO to avoid GAE bug (as discussed in plan logic)
        if args.num_envs > 1:
            logger.warning("PPO does not currently support VectorEnv (GAE implementation limitation). Forcing num_envs=1.")
            args.num_envs = 1
            
        env = make_mujoco_env(args.env_name, num_envs=args.num_envs, seed=args.seed)
        agent = PPOAgent(env, args, writer)
    
    # 3. Training Loop
    state, _ = env.reset(seed=args.seed)
    current_ep_reward = np.zeros(args.num_envs)
    
    global_step = 0
    while global_step < args.total_timesteps:
        
        # --- Action Selection ---
        if args.algo == 'dqn':
            # vector step
            action = agent.select_action(state, global_step) # Returns array if num_envs > 1
            if args.num_envs == 1 and np.isscalar(action):
                 action = np.array([action])
            
            # DQN specific: execution
            next_state, reward, done, truncated, info = env.step(action)
            current_ep_reward += reward
            
            # Iterate over batch
            for i in range(args.num_envs):
                real_next_state = next_state[i]
                is_done = done[i] or truncated[i]
                
                # Check for final observation if done
                if is_done:
                    # AsyncVectorEnv logic for final observation
                    if "final_observation" in info:
                         # final_observation is usually a list of arrays or array of arrays
                         real_next_state = info["final_observation"][i]
                    elif "_final_observation" in info and info["_final_observation"][i]:
                         # Some gym versions
                         real_next_state = info["final_observation"][i]

                # If num_envs=1, state is (C, H, W). If > 1, state is (N, C, H, W).
                # But buffer expects (C, H, W).
                s_i = state[i] if args.num_envs > 1 else state
                a_i = action[i] if args.num_envs > 1 else action
                r_i = reward[i] if args.num_envs > 1 else reward
                ns_i = real_next_state if args.num_envs > 1 else real_next_state # handled above
                d_i = is_done
                
                # Handling single env case where indexing might be weird if we squeezed it?
                # Actually, make_atari_env(num=1) returns non-vector env.
                # So state is (C, H, W). len(state.shape)=3.
                # If num=2, state is (2, C, H, W).
                # My loop implies I treat single env as vector of size 1?
                # No, make_atari_env returns DummyVecEnv or similar only if num > 1?
                # In wrappers.py: if num_envs > 1: return AsyncVectorEnv. else: return make_env(0)().
                # So if num_envs=1, it is a STANDARD Gym Env.
                # Standard Env step returns scalar reward, int action.
                # My logic `current_ep_reward = np.zeros(args.num_envs)` implies array.
                # `action = agent.select_action(...)` returns scalar if single env.
                # `env.step(action)` returns `state` (C,H,W), `reward` (float), `done` (bool).
                # So I need to differentiate Single vs Vector in this loop OR force VectorEnv for N=1 too?
                # Forcing VectorEnv for N=1 is cleaner but might add overhead.
                # I will handle the difference.
                pass 

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
                current_ep_reward += reward
                
                # Optimized add_batch
                real_next_states = next_state.copy()
                
                # Fix next_state for done envs
                for i in range(args.num_envs):
                    if done[i] or truncated[i]:
                        # Try fetch final obs
                        if "final_observation" in info:
                             real_next_states[i] = info["final_observation"][i]
                        elif "_final_observation" in info and info["_final_observation"][i]:
                             real_next_states[i] = info["final_observation"][i]
                        
                        writer.add_scalar("Train/EpisodeReward", current_ep_reward[i], global_step + i)
                        logger.info(f"Global Step {global_step} | Env {i} Reward: {current_ep_reward[i]:.2f}")
                        current_ep_reward[i] = 0
                
                # Bulk add
                agent.buffer.add_batch(state, action, reward, real_next_states, done | truncated)
                
                state = next_state
                global_step += args.num_envs
            
            # Train every train_freq steps (for efficiency)
            if global_step % args.train_freq == 0:
                agent.learn()
            
        elif args.algo == 'ppo':
            # PPO (Forces num_envs=1 currently)
            action, log_prob, value = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            current_ep_reward[0] += reward
            
            # Store in RolloutBuffer
            agent.buffer.add(state, action, log_prob, reward, done, value)
            
            state = next_state
            global_step += 1
            
            if done or truncated:
                writer.add_scalar("Train/EpisodeReward", current_ep_reward[0], global_step)
                logger.info(f"Step {global_step} | Train Reward: {current_ep_reward[0]:.2f}")
                current_ep_reward[0] = 0
                state, _ = env.reset()
            
            # Update if buffer full
            if agent.buffer.full:
                _, _, last_val = agent.select_action(state, eval_mode=True)
                agent.learn(last_val)

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
