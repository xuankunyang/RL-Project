import os
import re
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pickle
import argparse

# Set style
plt.style.use('seaborn-v0_8')
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['savefig.edgecolor'] = 'white'
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

def parse_dqn_exp_dir(exp_dir):
    """Parse DQN experiment directory name to extract parameters."""
    # Pattern: lr{lr}_uf{update_freq}_sd{seed}_hd{hidden_dim}_bs{batch_size}_env{num_envs}_{timestamp}
    pattern = r'lr([0-9e.-]+)_uf(\d+)_sd(\d+)_hd(\d+)_bs(\d+)_env(\d+)_(\d+-\d+)'
    match = re.search(pattern, exp_dir)
    if match:
        lr = float(match.group(1))
        update_freq = int(match.group(2))
        seed = int(match.group(3))
        hidden_dim = int(match.group(4))
        batch_size = int(match.group(5))
        num_envs = int(match.group(6))
        timestamp = match.group(7)
        return {
            'lr': lr,
            'update_freq': update_freq,
            'seed': seed,
            'hidden_dim': hidden_dim,
            'batch_size': batch_size,
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

def find_experiment_dirs(base_results_dir, selected_configs):
    """Find directories matching the selected configurations for each variant."""
    exp_dirs = {}

    for variant, config in selected_configs.items():
        variant_dir = os.path.join(base_results_dir, variant)
        if not os.path.exists(variant_dir):
            print(f"Warning: Variant directory {variant_dir} does not exist")
            continue

        found = False
        for exp_dir in os.listdir(variant_dir):
            exp_path = os.path.join(variant_dir, exp_dir)
            if not os.path.isdir(exp_path):
                continue

            params = parse_dqn_exp_dir(exp_dir)
            if not params:
                continue

            # Check if params match the selected config
            if (params['lr'] == config['lr'] and
                params['update_freq'] == config['update_freq'] and
                params['hidden_dim'] == config['hidden_dim'] and
                params['seed'] == config['seed']):
                exp_dirs[variant] = exp_path
                found = True
                break

        if not found:
            print(f"Warning: No experiment found for {variant} with config {config}")

    return exp_dirs

def plot_curve(data_dict, tag, ema_alpha=0.1, save_dir='results/Atari/ALE/Breakout-v5/analysis', env_name='Breakout-v5'):
    """Plot a single curve for all DQN variants."""
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 8))

    # Colors for different variants
    colors = ['blue', 'red', 'green', 'orange']
    variant_names = {
        'DQN_Vanilla': 'Vanilla DQN',
        'DQN_Double': 'Double DQN',
        'DQN_Rainbow': 'Rainbow DQN',
        'DQN_Dueling': 'Dueling DQN'
    }

    is_reward_curve = tag in ['Train/EpisodeReward', 'Eval/MeanReward']
    is_loss_curve = tag in ['Loss/DQN', 'Value/MeanQ', 'Gradients/Norm']  # These need step scaling

    # Define plotting order: vanilla (bottom), double, rainbow, dueling (top)
    plot_order = ['DQN_Vanilla', 'DQN_Double', 'DQN_Rainbow', 'DQN_Dueling']

    for i, variant in enumerate(plot_order):
        if variant not in data_dict:
            continue

        data = data_dict[variant]
        label = variant_names.get(variant, variant)

        if tag in data:
            raw_data = data[tag]

            # Scale steps for loss/Q-value/gradient curves (multiply by 4 due to frame skip)
            if is_loss_curve:
                raw_data = [(step * 4, value) for step, value in raw_data]

            if is_reward_curve:
                # For reward curves: plot smoothed line
                smoothed_data = apply_ema(raw_data, alpha=ema_alpha)
                steps, values = zip(*smoothed_data)
                ax.plot(steps, values, color=colors[i], linewidth=2, label=label)
            else:
                # For loss/other curves: compute mean per step and smooth
                from collections import defaultdict
                step_values = defaultdict(list)
                for step, value in raw_data:
                    step_values[step].append(value)

                mean_data = [(step, np.mean(values)) for step, values in sorted(step_values.items())]

                # Apply EMA to means
                smoothed_mean = apply_ema(mean_data, alpha=ema_alpha)
                steps_mean, values_mean = zip(*smoothed_mean)
                ax.plot(steps_mean, values_mean, color=colors[i], linewidth=2, label=label)

    ax.set_xlabel('Training Steps')
    ax.set_ylabel(tag.replace('/', ' ').replace('Train', '').replace('Eval', 'Evaluation').strip())
    # ax.set_title(f'{env_name} - {tag.replace("/", " - ")} Comparison (EMA α={ema_alpha})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{tag.replace("/", "_")}_dqn_{env_name.replace("-", "_")}_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {save_path}")

def get_selected_configs_for_env(env_name):
    """Get selected configurations for a specific environment."""
    # Define configurations for each environment
    env_configs = {
        'Breakout-v5': {
            'DQN_Vanilla': {'lr': 1e-4, 'update_freq': 1000, 'hidden_dim': 256, 'seed': 42},
            'DQN_Double': {'lr': 1e-4, 'update_freq': 1000, 'hidden_dim': 512, 'seed': 42},
            'DQN_Dueling': {'lr': 1e-4, 'update_freq': 5000, 'hidden_dim': 512, 'seed': 42},
            'DQN_Rainbow': {'lr': 1e-4, 'update_freq': 2000, 'hidden_dim': 512, 'seed': 42}
        },
        'Pong-v5': {
            # TODO: Replace with your selected configurations for Pong
            'DQN_Vanilla': {'lr': 1e-4, 'update_freq': 2000, 'hidden_dim': 512, 'seed': 42},  # Example - replace with your chosen params
            'DQN_Double': {'lr': 1e-4, 'update_freq': 1000, 'hidden_dim': 256, 'seed': 42},   # Example - replace with your chosen params
            'DQN_Dueling': {'lr': 5e-5, 'update_freq': 1000, 'hidden_dim': 256, 'seed': 42},  # Example - replace with your chosen params
            'DQN_Rainbow': {'lr': 1e-4, 'update_freq': 2000, 'hidden_dim': 512, 'seed': 42}   # Example - replace with your chosen params
        }
    }

    return env_configs.get(env_name, {})

def plot_environment_curves(env_name, ema_alpha=0.5, reload_cache=False):
    """Plot curves for a specific environment."""
    print(f"\n{'='*60}")
    print(f"Processing environment: {env_name}")
    print(f"{'='*60}")

    base_results_dir = f'results/Atari/ALE/{env_name}'
    selected_configs = get_selected_configs_for_env(env_name)

    if not selected_configs:
        print(f"No configurations defined for {env_name}")
        return

    cache_file = os.path.join(base_results_dir, 'analysis', 'dqn_selected_cache.pkl')

    if reload_cache or not os.path.exists(cache_file):
        print("Finding experiment directories...")
        exp_dirs = find_experiment_dirs(base_results_dir, selected_configs)

        if len(exp_dirs) != len(selected_configs):
            print(f"Warning: Found {len(exp_dirs)}/{len(selected_configs)} expected experiments")
            for variant in selected_configs:
                if variant not in exp_dirs:
                    print(f"Missing: {variant}")

        # Load and cache raw data
        print("Loading and caching raw data...")
        cached_data = {}
        for variant, exp_path in exp_dirs.items():
            print(f"Loading {variant}...")
            data = load_experiment_data(exp_path)
            if data:
                cached_data[variant] = data
            else:
                print(f"Failed to load data for {variant}")

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
    save_dir = os.path.join(base_results_dir, 'analysis')
    print(f"Plotting with EMA alpha={ema_alpha}...")
    for tag in all_tags:
        print(f"Plotting {tag}...")
        plot_curve(cached_data, tag, ema_alpha=ema_alpha, save_dir=save_dir, env_name=env_name)

    print(f"All plots for {env_name} generated!")

def main():
    parser = argparse.ArgumentParser(description='Plot DQN curves for selected configurations across environments')
    parser.add_argument('--env', '--environments', type=str, default='Pong-v5',
                        help='Environment name(s), comma-separated (e.g., "Breakout-v5,Pong-v5")')
    parser.add_argument('--ema_alpha', type=float, default=1.0, help='EMA smoothing alpha')
    parser.add_argument('--reload_cache', action='store_true', help='Reload data from TensorBoard instead of using cache')

    args = parser.parse_args()

    # Parse environment list
    environments = [env.strip() for env in args.env.split(',')]

    print(f"Plotting DQN curves for environments: {', '.join(environments)}")
    print(f"EMA alpha: {args.ema_alpha}")

    # Plot for each environment
    for env_name in environments:
        plot_environment_curves(env_name, args.ema_alpha, args.reload_cache)

    print("\n" + "="*80)
    print("All environments processed!")

if __name__ == "__main__":
    main()
