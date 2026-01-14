import os
import re
import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Set white background style
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['savefig.edgecolor'] = 'white'
plt.rcParams['figure.figsize'] = (16, 8)
plt.rcParams['font.size'] = 12

def parse_exp_dir(exp_dir):
    """Parse experiment directory name to extract parameters."""
    pattern = r'lra([0-9e.-]+)_lrc([0-9e.-]+)_clp([0-9.]+)_sd(\d+)_hd(\d+)_env(\d+)_(\d+-\d+)'
    match = re.search(pattern, exp_dir)
    if match:
        lr_actor = float(match.group(1))
        lr_critic = float(match.group(2))
        clip = float(match.group(3))
        seed = int(match.group(4))
        hidden_dim = int(match.group(5))
        num_envs = int(match.group(6))
        timestamp = match.group(7)
        return {
            'lr_actor': lr_actor,
            'lr_critic': lr_critic,
            'clip': clip,
            'seed': seed,
            'hidden_dim': hidden_dim,
            'num_envs': num_envs,
            'timestamp': timestamp,
            'exp_dir': exp_dir
        }
    return None

def load_experiment_data(exp_path):
    """Load scalar data from an experiment."""
    event_files = [f for f in os.listdir(exp_path) if f.startswith('events.out.tfevents')]
    if not event_files:
        return None

    event_file = os.path.join(exp_path, event_files[0])

    try:
        ea = EventAccumulator(event_file)
        ea.Reload()

        # Get specific scalar tags
        tags = ['Train/EpisodeReward', 'Eval/MeanReward', 'Ratio/Max']
        data = {}

        for tag in tags:
            try:
                scalars = ea.Scalars(tag)
                # Store as list of (step, value)
                data[tag] = [(s.step, s.value) for s in scalars]
            except:
                data[tag] = []

        return data
    except Exception as e:
        print(f"Error loading {exp_path}: {e}")
        return None

def apply_ema(data, alpha=0.1):
    """Apply exponential moving average to (step, value) list."""
    if not data:
        return []

    steps, values = zip(*data)
    steps = np.array(steps)
    values = np.array(values)

    # Apply EMA
    ema_values = np.zeros_like(values)
    ema_values[0] = values[0]
    for i in range(1, len(values)):
        ema_values[i] = alpha * values[i] + (1 - alpha) * ema_values[i-1]

    return list(zip(steps, ema_values))

def find_clip_experiments(results_dir):
    """Find experiments with different clip values but fixed other parameters."""
    clip_experiments = {}

    target_lr_actor = 5e-5
    target_lr_critic = 2e-4
    target_hidden_dim = 256
    target_seed = 42

    for exp_dir in os.listdir(results_dir):
        exp_path = os.path.join(results_dir, exp_dir)
        if not os.path.isdir(exp_path):
            continue

        params = parse_exp_dir(exp_dir)
        if not params:
            continue

        # Check if matches target parameters
        if (params['lr_actor'] == target_lr_actor and
            params['lr_critic'] == target_lr_critic and
            params['hidden_dim'] == target_hidden_dim and
            params['seed'] == target_seed):
            clip = params['clip']
            clip_experiments[clip] = exp_path

    return clip_experiments

def plot_reward_curve(data_dict, tag, ema_alpha=0.1, save_path=None):
    """Plot reward curve (Train or Eval) with EMA smoothing."""
    fig, ax = plt.subplots(figsize=(16, 8))

    colors = ['blue', 'red', 'green', 'orange']

    for i, (clip, data) in enumerate(sorted(data_dict.items())):
        label = f'Clip={clip}'

        if tag in data and data[tag]:
            raw_data = data[tag]
            smoothed_data = apply_ema(raw_data, alpha=ema_alpha)
            steps, values = zip(*smoothed_data)
            ax.plot(steps, values, color=colors[i], linewidth=2, label=label)

    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel(tag, fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Ensure white background
    ax.set_facecolor('white')
    fig.set_facecolor('white')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {tag} plot: {save_path}")

def plot_ratio_max_curve(data_dict, max_value_threshold=None, save_path=None):
    """Plot Ratio/Max with scatter points and mean line (no EMA on mean)."""
    fig, ax = plt.subplots(figsize=(16, 8))

    colors = ['blue', 'red', 'green', 'orange']
    tag = 'Ratio/Max'

    for i, (clip, data) in enumerate(sorted(data_dict.items())):
        label = f'Clip={clip}'

        if tag in data and data[tag]:
            raw_data = data[tag]

            # Filter out values above threshold if specified
            if max_value_threshold is not None:
                filtered_data = [(step, value) for step, value in raw_data if value <= max_value_threshold]
            else:
                filtered_data = raw_data

            if not filtered_data:
                print(f"Warning: No data left for clip={clip} after filtering")
                continue

            # Scatter all points
            steps_all, values_all = zip(*filtered_data)

            steps_all = list(steps_all)
            steps_all = [x * 200 for x in steps_all]

            ax.scatter(steps_all, values_all, color=colors[i], alpha=0.3, s=2, label=f'{label} (points)')

            # Compute mean per step (no EMA)
            step_values = defaultdict(list)
            for step, value in filtered_data:
                step_values[step].append(value)

            mean_steps = []
            mean_values = []
            for step in sorted(step_values.keys()):
                mean_steps.append(step * 200)
                mean_values.append(np.mean(step_values[step]))

            # Plot mean line
            ax.plot(mean_steps, mean_values, color=colors[i], linewidth=2, label=f'{label} (mean)')

    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel(f'Ratio/Max (filtered: value ≤ {max_value_threshold})', fontsize=12)
    # if max_value_threshold is not None:
    #     ax.set_title(f'Ratio/Max (filtered: value ≤ {max_value_threshold})', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Ensure white background
    ax.set_facecolor('white')
    fig.set_facecolor('white')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Ratio/Max plot: {save_path}")

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Plot clip comparison curves')
    parser.add_argument('--ema_alpha', type=float, default=1.0, help='EMA smoothing alpha for reward curves')
    parser.add_argument('--ratio_max_threshold', type=float, default=None, help='Maximum value threshold for Ratio/Max filtering')
    parser.add_argument('--reload_cache', action='store_true', help='Reload data from TensorBoard instead of using cache')

    args = parser.parse_args()

    results_dir = 'results/MuJoCo/HalfCheetah-v4/PPO_Standard'
    save_dir = os.path.join(results_dir, 'analysis')
    cache_file = os.path.join(save_dir, 'clip_comparison_cache.pkl')

    if args.reload_cache or not os.path.exists(cache_file):
        print("Finding clip experiments...")
        clip_experiments = find_clip_experiments(results_dir)

        print(f"Found experiments for clips: {sorted(clip_experiments.keys())}")

        # Load and cache data
        print("Loading and caching data...")
        cached_data = {}
        for clip, exp_path in clip_experiments.items():
            print(f"Loading clip={clip}...")
            data = load_experiment_data(exp_path)
            if data:
                cached_data[clip] = data
            else:
                print(f"Failed to load data for clip={clip}")

        # Save cache
        os.makedirs(save_dir, exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(cached_data, f)
        print(f"Cached data saved to {cache_file}")
    else:
        print("Loading cached data...")
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)
        print(f"Cached data loaded from {cache_file}")

    print(f"Available clips: {sorted(cached_data.keys())}")

    # Plot Train/EpisodeReward
    print("Plotting Train/EpisodeReward...")
    save_path = os.path.join(save_dir, 'Train_EpisodeReward_clip_comparison.png')
    plot_reward_curve(cached_data, 'Train/EpisodeReward', ema_alpha=args.ema_alpha, save_path=save_path)

    # Plot Eval/MeanReward
    print("Plotting Eval/MeanReward...")
    save_path = os.path.join(save_dir, 'Eval_MeanReward_clip_comparison.png')
    plot_reward_curve(cached_data, 'Eval/MeanReward', ema_alpha=args.ema_alpha, save_path=save_path)

    # Plot Ratio/Max
    print("Plotting Ratio/Max...")
    save_path = os.path.join(save_dir, 'Ratio_Max_clip_comparison.png')
    plot_ratio_max_curve(cached_data, max_value_threshold=args.ratio_max_threshold, save_path=save_path)

    print("All plots generated!")

if __name__ == "__main__":
    main()
