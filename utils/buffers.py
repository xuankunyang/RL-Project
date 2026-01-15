import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, capacity, state_shape, device, state_dtype=np.uint8):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0
        
        self.states = np.zeros((capacity, *state_shape), dtype=state_dtype)
        self.actions = np.zeros((capacity, 1), dtype=np.int64)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, *state_shape), dtype=state_dtype)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_batch(self, states, actions, rewards, next_states, dones):
        """
        Add a batch of transitions to the buffer.
        states: (B, ...)
        actions: (B,) or (B, 1)
        rewards: (B,) or (B, 1)
        next_states: (B, ...)
        dones: (B,) or (B, 1)
        """
        batch_size = len(states)
        if self.ptr + batch_size <= self.capacity:
            # No wrap-around needed
            self.states[self.ptr : self.ptr + batch_size] = states
            self.actions[self.ptr : self.ptr + batch_size] = actions.reshape(-1, 1)
            self.rewards[self.ptr : self.ptr + batch_size] = rewards.reshape(-1, 1)
            self.next_states[self.ptr : self.ptr + batch_size] = next_states
            self.dones[self.ptr : self.ptr + batch_size] = dones.reshape(-1, 1)
            
            self.ptr = (self.ptr + batch_size) % self.capacity
            self.size = min(self.size + batch_size, self.capacity)
        else:
            # Wrap-around needed
            space_left = self.capacity - self.ptr
            # First part
            self.states[self.ptr:] = states[:space_left]
            self.actions[self.ptr:] = actions[:space_left].reshape(-1, 1)
            self.rewards[self.ptr:] = rewards[:space_left].reshape(-1, 1)
            self.next_states[self.ptr:] = next_states[:space_left]
            self.dones[self.ptr:] = dones[:space_left].reshape(-1, 1)
            
            # Second part (start from 0)
            rem = batch_size - space_left
            self.states[:rem] = states[space_left:]
            self.actions[:rem] = actions[space_left:].reshape(-1, 1)
            self.rewards[:rem] = rewards[space_left:].reshape(-1, 1)
            self.next_states[:rem] = next_states[space_left:]
            self.dones[:rem] = dones[space_left:].reshape(-1, 1)
            
            self.ptr = rem
            self.size = min(self.size + batch_size, self.capacity)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        
        # Note: If uint8, conversion to float happens here or inside agent?
        # Usually standard is converts to float inside agent or here. 
        # ReplayBuffer just returns what is stored. Agent handles normalization.
        
        return (
            torch.FloatTensor(self.states[ind]).to(self.device), # Converts uint8 to float32
            torch.LongTensor(self.actions[ind]).to(self.device),
            torch.FloatTensor(self.rewards[ind]).to(self.device),
            torch.FloatTensor(self.next_states[ind]).to(self.device),
            torch.FloatTensor(self.dones[ind]).to(self.device)
        )
        

class RolloutBuffer:
    """
    Vectorized Rollout Buffer for PPO with multiple parallel environments.
    Shape: (num_steps, num_envs, ...)
    """
    def __init__(self, buffer_size, state_shape, action_dim, num_envs, device, gamma=0.99, gae_lambda=0.95):
        self.max_size = buffer_size
        self.num_envs = num_envs
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        # Storage: (num_steps, num_envs, ...)
        self.states = torch.zeros((buffer_size, num_envs, *state_shape)).to(device)
        self.actions = torch.zeros((buffer_size, num_envs, action_dim)).to(device)
        
        # === FIX 1: Log Prob 应该存 (N, 1)，而不是 (N, action_dim) ===
        # 因为我们在 select_action 里已经 sum(dim=-1) 了
        self.log_probs = torch.zeros((buffer_size, num_envs, 1)).to(device)
        self.rewards = torch.zeros((buffer_size, num_envs, 1)).to(device)
        self.dones = torch.zeros((buffer_size, num_envs, 1)).to(device)
        self.values = torch.zeros((buffer_size, num_envs, 1)).to(device)
        
        self.ptr = 0
        self.full = False

    def add_batch(self, states, actions, log_probs, rewards, dones, values):
        """
        Add a batch of transitions from all environments.
        states: (num_envs, state_dim) or (num_envs, *state_shape)
        actions: (num_envs, action_dim)
        log_probs: (num_envs, action_dim) or (num_envs,)
        rewards: (num_envs,)
        dones: (num_envs,)
       values: (num_envs,)
        """
        if self.ptr >= self.max_size:
            raise ValueError("Buffer is full!")
            
        self.states[self.ptr] = torch.FloatTensor(states).to(self.device)
        self.actions[self.ptr] = torch.FloatTensor(actions).to(self.device)
        
        if len(log_probs.shape) == 1:
            log_probs = log_probs.reshape(-1, 1)
        self.log_probs[self.ptr] = torch.FloatTensor(log_probs).to(self.device)
        
        self.rewards[self.ptr] = torch.FloatTensor(rewards).reshape(-1, 1).to(self.device)
        self.dones[self.ptr] = torch.FloatTensor(dones).reshape(-1, 1).to(self.device)
        self.values[self.ptr] = torch.FloatTensor(values).reshape(-1, 1).to(self.device)
        
        self.ptr += 1
        if self.ptr == self.max_size:
            self.full = True

    def compute_gae_and_returns(self, last_values):
        """
        Compute GAE (Generalized Advantage Estimation) and Returns.
        last_values: (num_envs,) - bootstrap values for final state
        
        Handles multiple environments independently.
        """
        last_values = torch.FloatTensor(last_values).reshape(-1, 1).to(self.device)
        
        advantages = torch.zeros_like(self.rewards).to(self.device)
        last_gae_lam = torch.zeros((self.num_envs, 1)).to(self.device)
        
        # Backward pass for each timestep
        for t in reversed(range(self.max_size)):
            if t == self.max_size - 1:
                next_non_terminal = 1.0 - self.dones[t]
                next_value = last_values
            else:
                next_non_terminal = 1.0 - self.dones[t]
                next_value = self.values[t + 1]
            
            delta = self.rewards[t] + self.gamma * next_value * next_non_terminal - self.values[t]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            advantages[t] = last_gae_lam
            
        returns = advantages + self.values
        return advantages, returns

    def clear(self):
        self.ptr = 0
        self.full = False

    def get_batches(self, batch_size, advantages, returns):
        """
        Get mini-batches for training.
        Flatten (num_steps, num_envs) to (num_steps * num_envs) and shuffle.
        """
        # Flatten all buffers
        total_samples = self.max_size * self.num_envs
        
        states_flat = self.states.reshape(total_samples, -1)
        actions_flat = self.actions.reshape(total_samples, -1)
        log_probs_flat = self.log_probs.reshape(total_samples, -1)
        values_flat = self.values.reshape(total_samples, -1)
        advantages_flat = advantages.reshape(total_samples, -1)
        returns_flat = returns.reshape(total_samples, -1)
        
        indices = np.arange(total_samples)
        np.random.shuffle(indices)
        
        for start_idx in range(0, total_samples, batch_size):
            idx = indices[start_idx : start_idx + batch_size]
            yield (
                states_flat[idx],
                actions_flat[idx],
                log_probs_flat[idx],
                values_flat[idx],
                advantages_flat[idx],
                returns_flat[idx]
            )