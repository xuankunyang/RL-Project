# RL Project Guide

I have integrated the RL framework, implemented DQN (for Atari) and PPO (for MuJoCo), and provided scripts for parameter search.

## 1. Project Structure
```
RL-Project/
├── agents/ ...
├── models/ ...
├── utils/ ...
├── scripts/ ...
├── results/               
│   ├── Atari/ ...
│   └── MuJoCo/ ...
├── run.py                 
└── requirements.txt       
```

## 2. How to Run

### Single Experiment (Debug / Test)
```bash
# Vanilla DQN (Single Core)
python run.py --algo dqn --dqn_type dqn --env_name BreakoutNoFrameskip-v4 --seed 42 --device cuda:0

# DQN with Vectorized Environments (Speed Up!)
# Uses 8 CPU cores to collect data in parallel
python run.py --algo dqn --dqn_type dqn --env_name BreakoutNoFrameskip-v4 --num_envs 8 --device cuda:0

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

### Output & Logging
- **Directory Structure**: `results/Region/Environment/Variant/Hyperparams_Timestamp/`
- **Log File**: `log.txt` inside the run folder.
- **Models**: `models/` subdirectory.

### DQN Variants (`--dqn_type`)
- **`dqn`**: Standard Nature DQN.
- **`double`**: Double Q-Learning (Reduces overestimation).
- **`dueling`**: Dueling Network Architecture (V + A).
- **`rainbow`**: Combines Double + Dueling + PER + N-Step.

### PPO Variants
- **Separate LRs**: Use `--lr_actor` and `--lr_critic`.
- **Clipping**: Use `--ppo_clip`.

### Performance Optimization
- **`--num_envs N`**: Use N parallel environments to speed up data collection on multi-core CPUs.
  - Highly recommended for **DQN** on Atari (e.g., set to 8 or 16).
  - Note: PPO typically works better with synced vector envs which support GAE properly. My simplified PPO buffer currently supports single-stream data, so stick to `num_envs=1` for PPO correctness or upgrade buffer logic.

## 4. Key Parameters
| Parameter | Default | Description |
| :--- | :--- | :--- |
| `--algo` | `dqn` | Algorithm to use (`dqn` or `ppo`). |
| `--dqn_type` | `dqn` | Variant: `dqn`, `double`, `dueling`, `rainbow`. |
| `--lr_actor` | `None` | Actor LR (PPO). Defaults to --lr. |
| `--lr_critic` | `None` | Critic LR (PPO). Defaults to --lr. |
| `--ppo_clip` | `0.2` | PPO Clipping range. |
| `--num_envs` | `1` | Number of parallel envs (DQN). |
| `--hidden_dim_dqn` | `512` | Hidden dim for DQN. |
| `--hidden_dim_ppo` | `256` | Hidden dim for PPO. |

## 5. Verification
Verified on local CPU:
- **All DQN Variants** ran successfully.
- **PPO** ran successfully.
