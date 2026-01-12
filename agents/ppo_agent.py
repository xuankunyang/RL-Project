import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from models.networks import GaussianPolicy
from utils.buffers import RolloutBuffer

class PPOAgent:
    def __init__(self, env, args, writer):
        self.device = torch.device(args.device)
        self.writer = writer
        self.env = env
        
        # Hyperparameters
        self.gamma = args.gamma
        self.learning_rate = args.lr
        self.lr_actor = args.lr_actor if args.lr_actor is not None else args.lr
        self.lr_critic = args.lr_critic if args.lr_critic is not None else args.lr
        self.clip_range = args.ppo_clip
        self.ppo_epochs = args.ppo_epochs
        self.mini_batch_size = args.mini_batch_size
        self.hidden_dim = args.hidden_dim_ppo
        self.vf_coef = args.vf_coef
        self.ent_coef = args.ent_coef
        self.horizon = args.horizon
        self.gae_lambda = args.gae_lambda
        
        # Model
        if hasattr(env, "single_observation_space"):
             self.state_dim = env.single_observation_space.shape[0]
             self.action_dim = env.single_action_space.shape[0]
        else:
             self.state_dim = env.observation_space.shape[0]
             self.action_dim = env.action_space.shape[0]
        
        self.policy = GaussianPolicy(self.state_dim, self.action_dim, hidden_dim=self.hidden_dim).to(self.device)
        
        # Separate Parameter Groups
        actor_params = list(self.policy.actor_net.parameters()) + \
                       list(self.policy.mean_layer.parameters()) + \
                       [self.policy.log_std_layer]
        critic_params = list(self.policy.critic_net.parameters())
        
        self.optimizer = optim.Adam([
            {'params': actor_params, 'lr': self.lr_actor},
            {'params': critic_params, 'lr': self.lr_critic}
        ])
        
        # Buffer (vectorized for multi-env)
        if hasattr(env, 'num_envs'):
            self.num_envs = env.num_envs
        else:
            self.num_envs = 1
        
        self.rollout_len = self.horizon // self.num_envs  # Adjust steps per env  
        self.buffer = RolloutBuffer(
            buffer_size=self.rollout_len, 
            state_shape=(self.state_dim,), 
            action_dim=self.action_dim,
            num_envs=self.num_envs,
            device=self.device, 
            gamma=self.gamma,
            gae_lambda=self.gae_lambda
        )
        
        self.update_step = 0

    def select_action(self, state, eval_mode=False):
        # Check for batch dim
        if len(state.shape) == 1:  # Single env (dim,)
             is_batched = False
             state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        else:  # Batched (N, dim)
             is_batched = True
             state_t = torch.FloatTensor(state).to(self.device)

        with torch.no_grad():
            dist = self.policy.get_action(state_t)
            value = self.policy(state_t)
            
            if eval_mode:
                action = dist.mean
            else:
                action = dist.sample()
                
            log_prob = dist.log_prob(action).sum(dim=-1)
        
        if is_batched:
            # Return arrays: action (N, A), log_prob (N,), value (N,)
            return action.cpu().numpy(), log_prob.cpu().numpy(), value.cpu().numpy().flatten()
        else:
            # Return scalars
            return action.cpu().numpy()[0], log_prob.cpu().item(), value.item()

    def learn(self, last_values):
        """
        PPO Main Learning Loop
        last_values: (num_envs,) - bootstrap values for each env
        """
        self.update_step += 1
        
        # 1. Compute GAE
        advantages, returns = self.buffer.compute_gae_and_returns(last_values)
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 2. PPO epochs
        for epoch in range(self.ppo_epochs):
            # Get mini-batches
            for states_batch, actions_batch, old_log_probs, _, advantages_batch, returns_batch in \
                self.buffer.get_batches(self.mini_batch_size, advantages, returns):
                
                # Forward pass
                dist = self.policy.get_action(states_batch)
                values = self.policy(states_batch)
                new_log_probs = dist.log_prob(actions_batch).sum(dim=-1, keepdim=True)
                entropy = dist.entropy().sum(dim=-1, keepdim=True)
                
                # Ratio for clipping
                ratio = torch.exp(new_log_probs - old_log_probs)
                
                # Policy loss
                surr1 = ratio * advantages_batch
                surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * advantages_batch
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.smooth_l1_loss(values, returns_batch)
                
                # Total loss
                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy.mean()
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.optimizer.step()
                
                # Logging
                if self.update_step % 10 == 0:
                    self.writer.add_scalar("Loss/Policy", policy_loss.item(), self.update_step)
                    self.writer.add_scalar("Loss/Value", value_loss.item(), self.update_step)
                    self.writer.add_scalar("Loss/Entropy", entropy.mean().item(), self.update_step)
                    self.writer.add_scalar("Ratio/Mean", ratio.mean().item(), self.update_step)
                    self.writer.add_scalar("Ratio/Max", ratio.max().item(), self.update_step)
        
        # Clear buffer
        self.buffer.clear()
    
    def save(self, path):
        torch.save(self.policy.state_dict(), path)
    
    def load(self, path):
        self.policy.load_state_dict(torch.load(path))
