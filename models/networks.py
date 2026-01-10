import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --- 1. Atari Visual Network (Dueling Architecture) ---
class DuelingCNN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DuelingCNN, self).__init__()
        c, h, w = input_shape
        
        # 经典的 Nature DQN 卷积层
        self.features = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # 计算卷积后的特征维度
        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1
        
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w, 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h, 8, 4), 4, 2), 3, 1)
        linear_input_size = convw * convh * 64
        
        # Dueling Network: 分为 Value 和 Advantage 两路
        self.fc_value = nn.Sequential(
            nn.Linear(linear_input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        self.fc_advantage = nn.Sequential(
            nn.Linear(linear_input_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        
        value = self.fc_value(x)
        advantage = self.fc_advantage(x)
        
        # Q = V + (A - mean(A))
        return value + advantage - advantage.mean(1, keepdim=True)

# --- 2. MuJoCo Policy Network (Continuous) ---
class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(GaussianPolicy, self).__init__()
        
        # Actor 网络
        self.actor_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Parameter(torch.zeros(1, action_dim)) # 可学习的标准差

        # Critic 网络 (Value Function)
        self.critic_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        # 返回 Value
        return self.critic_net(state)

    def get_action(self, state):
        # 返回 Action 分布
        x = self.actor_net(state)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer.expand_as(mean)
        std = torch.exp(log_std)
        
        dist = torch.distributions.Normal(mean, std)
        return dist