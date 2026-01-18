# Reinforcement Learning Project

## 1. Project Overview
This final project implements a comprehensive Reinforcement Learning framework capable of solving both discrete control tasks (Atari games using DQN) and continuous control tasks (MuJoCo robotics using PPO). It is designed for modularity, scalability, and ease of experimentation, featuring automated parallel training, configuration-driven evaluation, and robust headless visualization support.

**Key Features:**
*   **DQN (Deep Q-Network):** Supports Vanilla, Double, Dueling, and Rainbow variants.
*   **PPO (Proximal Policy Optimization):** Optimized for continuous control with observation normalization and reward clipping.
*   **Parallel Training:** Efficient data collection using vectorized environments.
*   **Automated Evaluation:** `run.py` for rendering, video recording, and performance metrics.
*   **Configuration Registry:** Centralized management of best model checkpoints via `configs/best_models.py`.

## 2. Project Structure

```text
RL-Project/
├── agents/                 # Agent implementations
│   ├── dqn_agent.py        # DQN logic (inc. Double, Dueling, Rainbow)
│   └── ppo_agent.py        # PPO logic
├── configs/                # Configuration files
│   └── best_models.py      # Registry for best trained models
├── models/                 # Neural network architectures
│   └── networks.py         # CNNs (Atari) and MLPs (MuJoCo)
├── utils/                  # Utility functions
│   ├── wrappers.py         # Environment wrappers (FrameStack, Normalize, etc.)
│   └── buffers.py          # Replay buffers
├── scripts/                # Shell scripts for batch experiments
├── analysis/               # Analysis and plotting scripts
├── results/                # Training logs and checkpoints (auto-generated)
├── videos/                 # Recorded evaluation videos (auto-generated)
├── train.py                # Main training script
├── run.py                  # Evaluation and visualization script
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## 3. Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/xuankunyang/RL-Project.git
    cd RL-Project
    ```

2.  **Create a virtual environment (Recommended):**
    ```bash
    conda create -n rl_project python=3.10
    conda activate rl_project
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

    *Note: For MuJoCo support, ensure you have the necessary system libraries installed (e.g., `libgl1-mesa-glx` on Linux).*

## 4. Usage of `run.py`

The `run.py` script is the primary tool for evaluating and visualizing trained agents. It automatically infers the algorithm and loads the best model configuration if available.

### Available Algorithms

*   **DQN Variants:** `dqn`, `double`, `dueling`, `rainbow`
*   **PPO:** `ppo`

### Available Environments

*   **Atari:** `ALE/Breakout-v5`, `ALE/Pong-v5`
*   **MuJoCo:** `HalfCheetah-v4`, `Hopper-v4`, `Ant-v4`

### Basic Usage
Evaluate the best available model for a specific environment:

```bash
# Evaluate Hopper (PPO) - Renders to window
python run.py --env_name Hopper-v4 --render

# Evaluate Breakout (DQN) - Renders to window
python run.py --env_name ALE/Breakout-v5 --render
```

### Visualization & Evaluation
*   **--render:** Opens a window showing the agent playing the game.
*   **--episodes N:** Runs evaluation for N episodes (default: 5) and reports Mean/Std Reward.
*   **--sleep S:** Adds a delay of S seconds between steps for slow-motion viewing.

### Server Environment (Headless)
On a server without a display, use `--save_video` to record MP4 videos instead of opening a window.

```bash
# Record video for Ant-v4 (Headless safe)
python run.py --env_name Ant-v4 --save_video
```
*   Videos are saved to the `videos/` directory.
*   The script automatically handles OpenGL backend (EGL) for headless rendering.

### Local Environment
On a local machine with a display, `--render` provides real-time feedback. You can use `--sleep 0.05` to slow down fast environments like Atari.

### Model Loading
You can manually specify a model path if you don't want to use the default "best" model:

```bash
python run.py --env_name Hopper-v4 --algo ppo --model_path path/to/model.pth
```

## 5. Model Training with `train.py`

The `train.py` script handles the training loop, data collection, and logging.

### Training Process
1.  **Initialization:** Sets up the environment (vectorized for speed) and agent networks.
2.  **Collection:** Agent interacts with the environment to fill the replay buffer (DQN) or collect trajectories (PPO).
3.  **Update:** Performs gradient descent updates on the networks.
4.  **Logging:** Saves TensorBoard logs to `results/` and checkpoints to `results/.../models/`.

### DQN Model Training (Atari)
Train a Rainbow DQN agent on Breakout:

```bash
python train.py --env_name ALE/Breakout-v5 --algo dqn --dqn_type rainbow --num_envs 16 --total_timesteps 5000000
```

### PPO Model Training (MuJoCo)
Train a PPO agent on Hopper:

```bash
python train.py --env_name Hopper-v4 --algo ppo --num_envs 16 --total_timesteps 2000000
```

### Training Parameters Table

| Parameter Name | Description | Default Value | Type | Example Value |
| :--- | :--- | :--- | :--- | :--- |
| `--env_name` | Gymnasium environment ID | `ALE/Breakout-v5` | `str` | `Hopper-v4` |
| `--algo` | Algorithm to use | `dqn` | `str` | `ppo` |
| `--dqn_type` | DQN variant (dqn, double, dueling, rainbow) | `dqn` | `str` | `rainbow` |
| `--seed` | Random seed for reproducibility | `42` | `int` | `101` |
| `--num_envs` | Number of parallel environments | `16` | `int` | `32` |
| `--total_timesteps`| Total environment steps to train | `5000000` | `int` | `10000000` |
| `--device` | Computing device | `cuda:0` | `str` | `cpu` |
| `--lr` | Learning rate (DQN / Shared) | `1e-4` | `float` | `2.5e-4` |
| `--lr_actor` | PPO Actor learning rate | `3e-4` | `float` | `1e-4` |
| `--lr_critic` | PPO Critic learning rate | `1e-3` | `float` | `1e-3` |
| `--batch_size` | Batch size for DQNs | `32` | `int` | `64` |
| `--mini_batch_size` | Batch size for PPO | `32` | `int` | `64` |
| `--hidden_dim_dqn`| Hidden layer size for DQN | `512` | `int` | `256` |
| `--hidden_dim_ppo`| Hidden layer size for PPO | `256` | `int` | `512` |

---
**Note:** For advanced usage (hyperparameter search), refer to the scripts in the `scripts/` directory.