import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import os
from models.networks import QNetwork
from utils.buffers import ReplayBuffer, PrioritizedReplayBuffer, NStepReplayBuffer

class DQNAgent:
    def __init__(self, env, args, writer):
        self.device = torch.device(args.device)
        self.writer = writer
        self.env = env
        self.dqn_type = args.dqn_type
        
        # Hyperparameters
        self.gamma = args.gamma
        self.batch_size = args.batch_size
        self.target_update_freq = args.update_freq
        self.epsilon_start = args.epsilon_start
        self.epsilon_final = args.epsilon_final
        self.epsilon_decay = args.epsilon_decay
        self.learning_rate = args.lr
        self.learn_step_counter = 0
        self.hidden_dim = args.hidden_dim_dqn
        self.learning_start = args.learning_start
        self.buffer_size = args.buffer_size
        
        # Flags based on dqn_type
        self.use_double = self.dqn_type in ['double', 'rainbow']
        self.use_dueling = self.dqn_type in ['dueling', 'rainbow']
        
        # Model
        if hasattr(env, "single_action_space"):
            self.action_dim = env.single_action_space.n
            input_shape = env.single_observation_space.shape
        else:
            self.action_dim = env.action_space.n
            input_shape = env.observation_space.shape # (4, 84, 84)
        
        self.q_net = QNetwork(input_shape, self.action_dim, use_dueling=self.use_dueling, hidden_dim=self.hidden_dim).to(self.device)
        self.target_net = QNetwork(input_shape, self.action_dim, use_dueling=self.use_dueling, hidden_dim=self.hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.learning_rate, eps=1e-4)
        
        self.buffer = ReplayBuffer(capacity=self.buffer_size, state_shape=input_shape, device=self.device)
    
    def get_epsilon(self, step):
        if step >= self.epsilon_decay:
            return self.epsilon_final
        else:
            return self.epsilon_start - (self.epsilon_start - self.epsilon_final) * (step / self.epsilon_decay)

    def select_action(self, state, steps_done, eval_mode=False):
        # Epsilon-Greedy
        if eval_mode:
            epsilon = 0.0
        else:
            epsilon = self.get_epsilon(step=steps_done - self.learning_start)
        
        # Check if state is batched (N, C, H, W) or single (C, H, W)
        # We assume VectorEnv returns (N, C, H, W) as numpy array
        if len(state.shape) == 3:
             # Single env case: add batch dim -> (1, C, H, W)
             state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
             scalar_output = True
        else:
             # Batched case: (N, C, H, W)
             state_t = torch.FloatTensor(state).to(self.device)
             scalar_output = False

        state_t = state_t / 255.0
        
        if not eval_mode and random.random() < epsilon:
            # Random Action
            if scalar_output:
                return self.env.action_space.sample() # Scalar
            else:
                batch_size = state.shape[0]
                return np.random.randint(0, self.action_dim, size=(batch_size,)) # Array (N,)
        else:
            with torch.no_grad():
                q_values = self.q_net(state_t)
                actions = q_values.max(1)[1].cpu().numpy() # Return (N,)
            
            if scalar_output:
                return actions[0] # Return scalar int
            else:
                return actions # Return batch array (N,)

    def learn(self):
        if self.buffer.size < self.batch_size:
            return
        
        self.learn_step_counter += 1
        

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        
        # Normalize batch (if not already float)
        # ReplayBuffer currently returns FloatTensor even if stored as uint8 (it converts in sample)
        # Wait, my ReplayBuffer.sample() code says: "torch.FloatTensor(self.states[ind])"
        # This casts uint8 to float32 but preserves values [0, 255].
        # So we MUST divide by 255.0 here.
        
        states = states / 255.0
        next_states = next_states / 255.0
        
        # Current Q
        q_values = self.q_net(states)
        current_q = q_values.gather(1, actions)
        
        # Target Q
        with torch.no_grad():
            if self.use_double:
                # Double DQN: action from Student, value from Teacher
                next_actions = self.q_net(next_states).argmax(dim=1, keepdim=True)
                next_target_q = self.target_net(next_states).gather(1, next_actions)
            else:
                # Vanilla DQN
                next_target_q = self.target_net(next_states).max(dim=1, keepdim=True)[0]
                
            target_q = rewards + (1 - dones) * self.gamma * next_target_q
            
        # Loss
        loss_elementwise = F.smooth_l1_loss(current_q, target_q, reduction='none')
        loss = loss_elementwise.mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Log Gradient Norm
        total_norm = 0.0
        for p in self.q_net.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10)
        self.optimizer.step()

        # Logging
        if self.learn_step_counter % 100 == 0:
            self.writer.add_scalar("Loss/DQN", loss.item(), self.learn_step_counter)
            self.writer.add_scalar("Value/MeanQ", current_q.mean().item(), self.learn_step_counter)
            self.writer.add_scalar("Gradients/Norm", total_norm, self.learn_step_counter)

        # Log Custom: Network Weights (Histogram) - REDUCED FREQ
        if self.learn_step_counter % 10000 == 0:
            for name, param in self.q_net.named_parameters():
                self.writer.add_histogram(f"Weights/{name}", param, self.learn_step_counter)

        # Update Target Net
        if self.learn_step_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

    def save(self, path):
        torch.save(self.q_net.state_dict(), path)

    def load(self, path):
        self.q_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.q_net.state_dict())
