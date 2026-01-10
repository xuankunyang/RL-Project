import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, capacity, state_shape, device):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0
        
        self.states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.actions = np.zeros((capacity, 1), dtype=np.int64)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        
        return (
            torch.FloatTensor(self.states[ind]).to(self.device),
            torch.LongTensor(self.actions[ind]).to(self.device),
            torch.FloatTensor(self.rewards[ind]).to(self.device),
            torch.FloatTensor(self.next_states[ind]).to(self.device),
            torch.FloatTensor(self.dones[ind]).to(self.device)
        )

class RolloutBuffer:
    def __init__(self, buffer_size, state_shape, action_dim, device, gamma=0.99, gae_lambda=0.95):
        self.max_size = buffer_size
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.states = torch.zeros((buffer_size, *state_shape)).to(device)
        self.actions = torch.zeros((buffer_size, action_dim)).to(device)
        self.log_probs = torch.zeros((buffer_size, action_dim)).to(device)
        self.rewards = torch.zeros((buffer_size, 1)).to(device)
        self.dones = torch.zeros((buffer_size, 1)).to(device)
        self.values = torch.zeros((buffer_size, 1)).to(device)
        
        self.ptr = 0
        self.full = False

    def add(self, state, action, log_prob, reward, done, value):
        if self.ptr >= self.max_size:
            raise ValueError("Buffer is full!")
            
        self.states[self.ptr] = torch.FloatTensor(state).to(self.device)
        self.actions[self.ptr] = torch.FloatTensor(action).to(self.device)
        self.log_probs[self.ptr] = torch.FloatTensor([log_prob]).to(self.device)
        self.rewards[self.ptr] = torch.FloatTensor([reward]).to(self.device)
        self.dones[self.ptr] = torch.FloatTensor([done]).to(self.device)
        self.values[self.ptr] = torch.FloatTensor([value]).to(self.device)
        
        self.ptr += 1
        if self.ptr == self.max_size:
            self.full = True

    def compute_gae_and_returns(self, last_value):
        """
        计算 GAE (Generalized Advantage Estimation) 和 Returns
        """
        advantages = torch.zeros_like(self.rewards).to(self.device)
        last_gae_lam = 0
        
        # 从后往前计算
        for t in reversed(range(self.max_size)):
            if t == self.max_size - 1:
                next_non_terminal = 1.0 - self.dones[t] # if done, 0
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t]
                next_value = self.values[t+1]
            
            delta = self.rewards[t] + self.gamma * next_value * next_non_terminal - self.values[t]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            advantages[t] = last_gae_lam
            
        returns = advantages + self.values
        return advantages, returns

    def clear(self):
        self.ptr = 0
        self.full = False

    def get_batches(self, batch_size):
        indices = np.arange(self.max_size)
        np.random.shuffle(indices)
        
        for start_idx in range(0, self.max_size, batch_size):
            idx = indices[start_idx : start_idx + batch_size]
            yield (
                self.states[idx],
                self.actions[idx],
                self.log_probs[idx],
                self.values[idx],
                idx # 返回索引以便外部使用 (如重新计算 returns)
            )