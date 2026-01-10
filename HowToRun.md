# RL Project Guide

I have integrated the RL framework, implemented DQN (for Atari) and PPO (for MuJoCo), and provided scripts for parameter search.

## 1. Project Structure
```
RL-Project/
├── agents/
│   ├── dqn_agent.py       # [MODIFIED] DQN Agent (Supports Vanilla, Double, Dueling, Rainbow)
│   └── ppo_agent.py       # [MODIFIED] PPO Agent (Separate LRs, Configurable Clip)
├── models/
│   └── networks.py        # [MODIFIED] Configurable QNetwork & GaussianPolicy
├── utils/
│   ├── buffers.py         # [MODIFIED] ReplayBuffer, PrioritizedReplayBuffer, NStepReplayBuffer, RolloutBuffer
│   └── wrappers.py        # Environment Wrappers (Atari & MuJoCo)
├── scripts/
│   └── run_param_search.ps1 # [MODIFIED] Distributed Parameter Search Script (Covers all variants)
├── run.py                 # [MODIFIED] Main Training Entry Point
└── requirements.txt       # Dependencies
```

## 2. How to Run

### Single Experiment (Debug / Test)
```bash
# Vanilla DQN
python run.py --algo dqn --dqn_type dqn --env_name BreakoutNoFrameskip-v4 --seed 42 --device cuda:0

# Double DQN
python run.py --algo dqn --dqn_type double --env_name BreakoutNoFrameskip-v4 --seed 42 --device cuda:0

# Rainbow-Lite (Double + Dueling + PER + 3-Step)
python run.py --algo dqn --dqn_type rainbow --env_name BreakoutNoFrameskip-v4 --seed 42 --device cuda:0

# PPO (Standard)
python run.py --algo ppo --env_name HalfCheetah-v4 --seed 42 --device cuda:0

# PPO (Advanced Tuning)
python run.py --algo ppo --env_name HalfCheetah-v4 --lr_actor 1e-4 --lr_critic 1e-3 --ppo_clip 0.2 --seed 42 --device cuda:0
```

### Parameter Search (Multi-GPU)
The updated script now compares `dqn_types` AND PPO Clipping variations (`0.2` vs `10.0`).
```powershell
./scripts/run_param_search.ps1
```
This distributes jobs across your 4 GPUs.

## 3. Algorithm Details & Logging

### DQN Variants (`--dqn_type`)
- **`dqn`**: Standard Nature DQN.
- **`double`**: Double Q-Learning (Reduces overestimation).
- **`dueling`**: Dueling Network Architecture (V + A).
- **`rainbow`**: Combines Double + Dueling + PER + N-Step.

### PPO Variants
- **Separate LRs**: Use `--lr_actor` and `--lr_critic` to tune them independently.
- **Clipping**: Use `--ppo_clip`. Set to a large value (e.g., 10.0) to disable clipping effects.

### Enhanced Logging
- **`Value/MeanQ`**: Tracks average Q-values (DQN).
- **`Weights/*`**: Histograms of network weights.
- **`Gradients/Norm`**: Gradient norms.
- **`Train/EpisodeReward`**: Live episode return during training.
- **`Ratio/Mean` & `Ratio/Max`**: (PPO) Monitor importance sampling ratios to see if clipping is active (Standard PPO usually has max < 1.3, No-Clip can go huge).

## 4. Key Parameters
| Parameter | Default | Description |
| :--- | :--- | :--- |
| `--algo` | `dqn` | Algorithm to use (`dqn` or `ppo`). |
| `--dqn_type` | `dqn` | Variant: `dqn`, `double`, `dueling`, `rainbow`. |
| `--lr_actor` | `None` | Actor LR (PPO). Defaults to --lr. |
| `--lr_critic` | `None` | Critic LR (PPO). Defaults to --lr. |
| `--ppo_clip` | `0.2` | PPO Clipping range. |

## 5. Verification
Verified on local CPU:
- **All DQN Variants** ran successfully.
- **PPO** ran successfully with new arguments.
