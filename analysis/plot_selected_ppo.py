import os
import re
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pickle

# Set style
plt.style.use('seaborn-v0_8')
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['savefig.edgecolor'] = 'white'
plt.rcParams['figure.figsize'] = (12, 8)
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
    """Load all scalar data from an experiment."""
    event_files = [f for f in os.listdir(exp_path) if f.startswith('events.out.tfevents')]
    if not event_files:
        return None

    event_file = os.path.join(exp_path, event_files[0])

    try:
        ea = EventAccumulator(event_file)
        ea.Reload()

        # Get all scalar tags
        tags = ea.Tags()['scalars']
        data = {}

        for tag in tags:
            scalars = ea.Scalars(tag)
            # Store as list of (step, value)
            data[tag] = [(s.step, s.value) for s in scalars]

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

def find_experiment_dirs(results_dir, selected_configs):
    """Find directories matching the selected configurations."""
    exp_dirs = {}

    for exp_dir in os.listdir(results_dir):
        exp_path = os.path.join(results_dir, exp_dir)
        if not os.path.isdir(exp_path):
            continue

        params = parse_exp_dir(exp_dir)
        if not params:
            continue

        key = (params['lr_actor'], params['lr_critic'], params['hidden_dim'], params['seed'])
        if key in selected_configs:
            exp_dirs[key] = exp_path

    return exp_dirs

def plot_curve(data_dict, tag, ema_alpha=0.1, save_dir='results/MuJoCo/HalfCheetah-v4/PPO_Standard/analysis'):
    """Plot a single curve for all configurations."""
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 8))

    colors = ['blue', 'red', 'green']

    is_reward_curve = tag in ['Train/EpisodeReward', 'Eval/MeanReward']

    for i, (config, data) in enumerate(data_dict.items()):
        lr_a, lr_c, hd, sd = config
        label = f'lr_a{lr_a:.0e}_lr_c{lr_c:.0e}_H{hd}_Seed{sd}'

        if tag in data:
            raw_data = data[tag]

            if is_reward_curve:
                # For reward curves: plot smoothed line
                smoothed_data = apply_ema(raw_data, alpha=ema_alpha)
                steps, values = zip(*smoothed_data)
                ax.plot(steps, values, color=colors[i], linewidth=2, label=label)
            else:
                # For loss/ratio curves: scatter plot + mean line
                # Scatter all points
                # steps_all, values_all = zip(*raw_data)
                # ax.scatter(steps_all, values_all, color=colors[i], alpha=0.1, s=1, label=f'{label} (scatter)')

                # Compute mean per step
                from collections import defaultdict
                step_values = defaultdict(list)
                for step, value in raw_data:
                    step_values[step].append(value)

                mean_data = [(step * 200, np.mean(values)) for step, values in sorted(step_values.items())]

                # Apply EMA to means
                smoothed_mean = apply_ema(mean_data, alpha=ema_alpha)
                steps_mean, values_mean = zip(*smoothed_mean)
                ax.plot(steps_mean, values_mean, color=colors[i], linewidth=2, label=f'{label}')

    ax.set_xlabel('Training Steps')
    ax.set_ylabel(tag)
    title_suffix = f'(EMA α={ema_alpha})' if is_reward_curve else f'(EMA α={ema_alpha}, scatter + mean)'
    # ax.set_title(f'{tag} {title_suffix}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{tag.replace("/", "_")}_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {save_path}")

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Plot PPO curves for selected configurations')
    parser.add_argument('--ema_alpha', type=float, default=0.5, help='EMA smoothing alpha')
    parser.add_argument('--reload_cache', action='store_true', help='Reload data from TensorBoard instead of using cache')

    args = parser.parse_args()

    results_dir = 'results/MuJoCo/HalfCheetah-v4/PPO_Standard'

    # Selected configurations: (lr_actor, lr_critic, hidden_dim, seed)
    selected_configs = [
        (1e-4, 1e-4, 256, 42),
        (5e-5, 2e-4, 256, 42),
        (5e-5, 2e-4, 512, 42)
    ]

    cache_file = os.path.join(results_dir, 'analysis', 'ppo_selected_cache.pkl')

    if args.reload_cache or not os.path.exists(cache_file):
        print("Finding experiment directories...")
        exp_dirs = find_experiment_dirs(results_dir, selected_configs)

        if len(exp_dirs) != len(selected_configs):
            print(f"Warning: Found {len(exp_dirs)}/{len(selected_configs)} expected experiments")
            for config in selected_configs:
                if config not in exp_dirs:
                    print(f"Missing: {config}")

        # Load and cache raw data
        print("Loading and caching raw data...")
        cached_data = {}
        for config, exp_path in exp_dirs.items():
            print(f"Loading {config}...")
            data = load_experiment_data(exp_path)
            if data:
                cached_data[config] = data
            else:
                print(f"Failed to load data for {config}")

        # Save cache to disk
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(cached_data, f)
        print(f"Cached data saved to {cache_file}")
    else:
        print("Loading cached data...")
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)
        print(f"Cached data loaded from {cache_file}")

    # Get all available tags
    all_tags = set()
    for data in cached_data.values():
        all_tags.update(data.keys())
    all_tags = sorted(all_tags)

    print(f"Available tags: {all_tags}")

    # Plot each tag
    print(f"Plotting with EMA alpha={args.ema_alpha}...")
    for tag in all_tags:
        print(f"Plotting {tag}...")
        plot_curve(cached_data, tag, ema_alpha=args.ema_alpha)

    print("All plots generated!")

if __name__ == "__main__":
    main()
