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

class PrioritizedReplayBuffer(ReplayBuffer):
    """
    简易版 Prioritized Replay Buffer (PER)
    为了保持简单，这里没有使用 SumTree 优化 (O(N) 采样)，
    而是使用概率数组采样 (O(N))。对于小 Batch 还可以接受。
    如果追求效率，需要引入 SumTree。
    """
    def __init__(self, capacity, state_shape, device, alpha=0.6):
        super(PrioritizedReplayBuffer, self).__init__(capacity, state_shape, device)
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.alpha = alpha
        self.pos = 0

    def add(self, state, action, reward, next_state, done):
        max_prio = self.priorities.max() if self.size > 0 else 1.0
        
        super().add(state, action, reward, next_state, done)
        
        # 修正指针 (因为 super.add 已经移了 ptr)
        # super.add: ptr = (ptr + 1) % cap
        # so self.pos should track the index that was just added
        idx = (self.ptr - 1 + self.capacity) % self.capacity
        self.priorities[idx] = max_prio

    def sample(self, batch_size, beta=0.4):
        if self.size == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.size]
            
        probs = prios ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(prios), batch_size, p=probs)
        
        # Importance Sampling Weights
        total = self.size
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        return (
            torch.FloatTensor(self.states[indices]).to(self.device),
            torch.LongTensor(self.actions[indices]).to(self.device),
            torch.FloatTensor(self.rewards[indices]).to(self.device),
            torch.FloatTensor(self.next_states[indices]).to(self.device),
            torch.FloatTensor(self.dones[indices]).to(self.device),
            torch.FloatTensor(weights).unsqueeze(1).to(self.device),
            indices
        )

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio


class NStepReplayBuffer:
    """
    用于 Rainbow 的 Multi-step Learning
    """
    def __init__(self, capacity, state_shape, device, target_buffer=None, n_step=3, gamma=0.99):
        self.capacity = capacity
        self.device = device
        self.n_step = n_step
        self.gamma = gamma
        self.n_step_buffer = [] # 暂存 n 步的数据
        
        if target_buffer is not None:
             self.buffer = target_buffer
        else:
             self.buffer = ReplayBuffer(capacity, state_shape, device)
    
    def add(self, state, action, reward, next_state, done):
        self.n_step_buffer.append((state, action, reward, next_state, done))
        
        if len(self.n_step_buffer) < self.n_step:
            return
            
        # 计算 N-step Return
        R = sum([self.n_step_buffer[i][2] * (self.gamma ** i) for i in range(self.n_step)])
        state_0, action_0 = self.n_step_buffer[0][:2]
        next_state_n, done_n = self.n_step_buffer[-1][3:]
        
        self.buffer.add(state_0, action_0, R, next_state_n, done_n)
        self.n_step_buffer.pop(0)
    
    def sample(self, batch_size):
        return self.buffer.sample(batch_size)
    
    @property
    def size(self):
        return self.buffer.size

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