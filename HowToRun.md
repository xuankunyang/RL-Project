# RL Project Guide

I have integrated the RL framework, implemented DQN (for Atari) and PPO (for MuJoCo), and provided scripts for parameter search.

## 1. Project Structure
```
RL-Project/
├── agents/
│   ├── dqn_agent.py       # [NEW] DQN Agent (Dueling, Double DQN)
│   └── ppo_agent.py       # [NEW] PPO Agent (Gaussian Policy, GAE)
├── models/
│   └── networks.py        # Neural Networks (DuelingCNN, GaussianPolicy)
├── utils/
│   ├── buffers.py         # ReplayBuffer & RolloutBuffer
│   └── wrappers.py        # Environment Wrappers (Atari & MuJoCo)
├── scripts/
│   └── run_param_search.ps1 # [NEW] Distributed Parameter Search Script
├── run.py                 # [MODIFIED] Main Training Entry Point
└── requirements.txt       # Dependencies
```

## 2. How to Run

### Single Experiment (Debug / Test)
To run a single experiment on your local machine or server:
```bash
# DQN on Atari
python run.py --algo dqn --env_name BreakoutNoFrameskip-v4 --seed 42 --device cuda:0

# PPO on MuJoCo
python run.py --algo ppo --env_name HalfCheetah-v4 --seed 42 --device cuda:0
```

### Parameter Search (Multi-GPU)
I provided a PowerShell script `scripts/run_param_search.ps1` to automatically distribute experiments across your 4 GPUs (cuda:0 to cuda:3).
```powershell
# Run from the project root
./scripts/run_param_search.ps1
```
This script will launch 8 experiments (2 algos * 2 LRs * 2 Seeds) in parallel.

## 3. Algorithm Details

### DQN (Deep Q-Network)
- **Architecture**: Dueling DQN (Separate Value and Advantage streams).
- **Improvements**: Double DQN (Decouples selection and evaluation of actions).
- **Buffer**: Standard Replay Buffer.
- **Preprocessing**: Atari Preprocessing (Gray-scale, Frame Stacking, etc.).

### PPO (Proximal Policy Optimization)
- **Architecture**: Actor-Critic with Gaussian Head for continuous control.
- **Advantage**: GAE (Generalized Advantage Estimation) for low variance.
- **Objective**: Clipped Surrogate Objective to prevent catastrophic updates.
- **Normalization**: Observation and Reward normalization included for MuJoCo.

## 4. Key Parameters
| Parameter | Default | Description |
| :--- | :--- | :--- |
| `--algo` | `dqn` | Algorithm to use (`dqn` or `ppo`). |
| `--env_name` | `Breakout...` | Gymnasium environment ID. |
| `--total_timesteps` | `1e6` | Total interaction steps with environment. |
| `--lr` | `3e-4` | Learning Rate. |
| `--gamma` | `0.99` | Discount factor for future rewards. |
| `--batch_size` | `64` | Batch size for updates. |
| `--device` | `cuda:0` | Compute device. |

## 5. Verification
I have verified the code on your local machine (CPU mode):
- **DQN Test**: `BreakoutNoFrameskip-v4` (200 steps) -> **PASSED**
- **PPO Test**: `HalfCheetah-v4` (200 steps) -> **PASSED**

You are ready to upload to your server and run!
