import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import os
from models.networks import DuelingCNN
from utils.buffers import ReplayBuffer

class DQNAgent:
    def __init__(self, env, args, writer):
        self.device = torch.device(args.device)
        self.writer = writer
        self.env = env
        
        # Hyperparameters
        self.gamma = args.gamma
        self.batch_size = args.batch_size
        self.target_update_freq = 1000
        self.epsilon_start = 1.0
        self.epsilon_final = 0.05
        self.epsilon_decay = 100000
        self.learning_rate = args.lr
        self.learn_step_counter = 0
        
        # Model
        self.action_dim = env.action_space.n
        input_shape = env.observation_space.shape # (4, 84, 84)
        
        self.q_net = DuelingCNN(input_shape, self.action_dim).to(self.device)
        self.target_net = DuelingCNN(input_shape, self.action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.learning_rate)
        
        # Buffer
        self.buffer = ReplayBuffer(capacity=100000, state_shape=input_shape, device=self.device)

    def select_action(self, state, steps_done, eval_mode=False):
        # Epsilon-Greedy
        if eval_mode:
            epsilon = 0.001
        else:
            epsilon = self.epsilon_final + (self.epsilon_start - self.epsilon_final) * \
                      np.exp(-1. * steps_done / self.epsilon_decay)
        
        if not eval_mode and random.random() < epsilon:
            return self.env.action_space.sample()
        else:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_net(state_t)
                return q_values.argmax(dim=1).item()

    def learn(self):
        if self.buffer.size < self.batch_size:
            return
        
        self.learn_step_counter += 1
        
        # Sample
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        
        # Current Q
        q_values = self.q_net(states)
        current_q = q_values.gather(1, actions)
        
        # Double DQN Target
        with torch.no_grad():
            next_actions = self.q_net(next_states).argmax(dim=1, keepdim=True)
            next_target_q = self.target_net(next_states).gather(1, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * next_target_q
            
        # Loss
        loss = F.smooth_l1_loss(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10)
        self.optimizer.step()
        
        # Logging
        if self.learn_step_counter % 100 == 0:
            self.writer.add_scalar("Loss/DQN", loss.item(), self.learn_step_counter)
            self.writer.add_scalar("Value/MeanQ", current_q.mean().item(), self.learn_step_counter)

        # Update Target Net
        if self.learn_step_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

    def save(self, path):
        torch.save(self.q_net.state_dict(), path)

    def load(self, path):
        self.q_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.q_net.state_dict())
