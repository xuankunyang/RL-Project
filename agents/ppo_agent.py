import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
from models.networks import GaussianPolicy
from utils.buffers import RolloutBuffer

class PPOAgent:
    def __init__(self, env, args, writer):
        self.device = torch.device(args.device)
        self.writer = writer
        self.env = env
        
        # Hyperparameters
        self.gamma = args.gamma
        self.lr = args.lr
        self.clip_param = 0.2
        self.ppo_epoch = 10
        self.batch_size = 64
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.01
        self.max_grad_norm = 0.5
        self.rollout_len = 2048 # PPO typically collects a large batch
        
        # Model
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        
        self.policy = GaussianPolicy(self.state_dim, self.action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        
        # Buffer
        self.buffer = RolloutBuffer(self.rollout_len, (self.state_dim,), self.action_dim, self.device, gamma=self.gamma)
        
        self.update_step = 0

    def select_action(self, state, eval_mode=False):
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            dist = self.policy.get_action(state_t)
            value = self.policy(state_t)
            
            if eval_mode:
                action = dist.mean
            else:
                action = dist.sample()
                
            log_prob = dist.log_prob(action).sum(dim=-1)
            
        return action.cpu().numpy()[0], log_prob.cpu().item(), value.item()

    def update(self):
        # 1. Compute Returns and Advantages
        with torch.no_grad():
            # 需要最后一个 state 的 value 来 bootstrapping
            # 这里简单起见，假设最后一步 done=False 的 value 预估。
            # 实际上在 run.py 收集循环结束时，我们应该传入最后一步的 observation
            pass 
        
        # 注意: PPO 的 update 通常是在外部收集完数据后调用。
        # 这里我们假设外部已经调用了 buffer.compute_gae_and_returns(last_value)
        
        # 2. Training Loop
        advantages, returns = self.buffer.compute_gae_and_returns(0) # Placeholder: last_value should be passed in
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for _ in range(self.ppo_epoch):
            data_generator = self.buffer.get_batches(self.batch_size)
            
            for sample in data_generator:
                states, actions, old_log_probs, values, _ = sample
                
                # Calculate new log probs and values
                dist = self.policy.get_action(states)
                new_log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)
                entropy = dist.entropy().mean()
                new_values = self.policy(states)
                
                # Probs ratio
                ratio = torch.exp(new_log_probs - old_log_probs.view(-1, 1))
                
                # Surrogate Loss
                curr_advantages = advantages[_] # This is wrong, need to index correctly
                # Re-indexing inside the loop
                # The generator yields sliced batches, so we need to slice advantages & returns too
                pass 

        # 由于 PPO logic 比较依赖 batch loop，下面重写一下完整的 update 逻辑，不依赖 generator 的 slice 隐式匹配
        pass
    
    def learn(self, last_v):
        """
        PPO Main Learning Loop
        last_v: Value of the next state (after the last step in buffer)
        """
        self.update_step += 1
        
        # 1. Compute GAE
        advantages, returns = self.buffer.compute_gae_and_returns(last_v)
        # Normalize AGV
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Flatten everything to verify shapes
        # Buffer stores (T, ...)
        b_states = self.buffer.states
        b_actions = self.buffer.actions
        b_log_probs = self.buffer.log_probs
        b_returns = returns
        b_advantages = advantages
        b_values = self.buffer.values
        
        # Optimizing
        for _ in range(self.ppo_epoch):
            sampler = self.buffer.get_batches(self.batch_size)
            for batch_data in sampler:
                states, actions, old_log_probs, old_values, indices = batch_data
                
                # Target values
                batch_returns = b_returns[indices]
                batch_advantages = b_advantages[indices]
                
                # New prediction
                dist = self.policy.get_action(states)
                new_log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)
                entropy = dist.entropy().mean()
                new_values = self.policy(states)
                
                # Ratio
                ratio = torch.exp(new_log_probs - old_log_probs.view(-1, 1))
                
                # Surrogate
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value Loss
                value_loss = F.mse_loss(new_values, batch_returns)
                
                # Total Loss
                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

        # Logging
        self.writer.add_scalar("Loss/Policy", policy_loss.item(), self.update_step)
        self.writer.add_scalar("Loss/Value", value_loss.item(), self.update_step)
        self.writer.add_scalar("Value/MeanAdvantage", advantages.mean().item(), self.update_step)
        self.writer.add_scalar("Entropy", entropy.item(), self.update_step)

        # Clear buffer
        self.buffer.clear()

    def save(self, path):
        torch.save(self.policy.state_dict(), path)

    def load(self, path):
        self.policy.load_state_dict(torch.load(path))
