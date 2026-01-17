import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import argparse

# Set white background style
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['savefig.edgecolor'] = 'white'

def plot_ratio_max(cached_data, save_path='results/MuJoCo/HalfCheetah-v4/PPO_Standard/analysis/Ratio_Max_detailed.png', env_name='HalfCheetah-v4'):
    """Plot Ratio_Max with scatter points and mean line (no EMA)."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 8))

    colors = ['blue', 'red', 'green']
    tag = 'Ratio/Max'

    for i, (config, data) in enumerate(cached_data.items()):
        lr_a, lr_c, hd, sd = config
        label = f'lr_a{lr_a:.0e}_lr_c{lr_c:.0e}_H{hd}_Seed{sd}'

        if tag in data:
            raw_data = data[tag]

            # Scatter all points
            steps_all, values_all = zip(*raw_data)

            steps_all = list(steps_all)
            steps_all = [x * 200 for x in steps_all]

            ax.scatter(steps_all, values_all, color=colors[i], alpha=0.3, s=2, label=f'{label} (points)')

            # Compute mean per step (no EMA)
            step_values = defaultdict(list)
            for step, value in raw_data:
                step_values[step].append(value)

            mean_steps = []
            mean_values = []
            N = len(step_values.keys())
            for step in sorted(step_values.keys()):
                mean_steps.append(step / 10000 * 2000000)
                mean_values.append(np.mean(step_values[step]))

            # Plot mean line
            ax.plot(mean_steps, mean_values, color=colors[i], linewidth=2, label=f'{label} (mean)')

    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Ratio/Max', fontsize=12)
    # ax.set_title(f'{env_name} - Ratio/Max: Scatter Plot + Mean Line', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Ensure white background
    ax.set_facecolor('white')
    fig.set_facecolor('white')

    plt.tight_layout()
    plt.show()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved detailed Ratio_Max plot: {save_path}")

def get_env_config(env_name):
    """Get configuration for a specific environment."""
    configs = {
        'HalfCheetah-v4': {
            'results_dir': 'results/MuJoCo/HalfCheetah-v4/PPO_Standard'
        },
        'Hopper-v4': {
            'results_dir': 'results/MuJoCo/Hopper-v4/PPO_Standard'
        }
    }
    return configs.get(env_name, configs['HalfCheetah-v4'])

def analyze_environment_ratio_max(env_name):
    """Analyze Ratio/Max for a specific environment."""
    print(f"\n{'='*60}")
    print(f"Analyzing Ratio/Max for {env_name}")
    print(f"{'='*60}")

    config = get_env_config(env_name)
    results_dir = config['results_dir']
    cache_file = os.path.join(results_dir, 'analysis', 'ppo_selected_cache.pkl')

    if not os.path.exists(cache_file):
        print(f"Cache file not found: {cache_file}")
        print(f"Please run 'python analysis/plot_selected_ppo.py --env {env_name} --reload_cache' first")
        return

    print("Loading cached data...")
    with open(cache_file, 'rb') as f:
        cached_data = pickle.load(f)

    # Check if Ratio/Max data exists
    has_ratio_data = False
    for data in cached_data.values():
        if 'Ratio/Max' in data:
            has_ratio_data = True
            break

    if not has_ratio_data:
        print("No Ratio/Max data found in cached data")
        return

    save_path = os.path.join(results_dir, 'analysis', f'Ratio_Max_{env_name.replace("-", "_")}_detailed.png')

    print("Plotting Ratio_Max...")
    plot_ratio_max(cached_data, save_path, env_name)

    print(f"Ratio/Max analysis completed for {env_name}!")

def main():
    parser = argparse.ArgumentParser(description='Plot Ratio/Max analysis for PPO across environments')
    parser.add_argument('--env', '--environments', type=str, default='HalfCheetah-v4',
                        help='Environment name(s), comma-separated (e.g., "HalfCheetah-v4,Hopper-v4")')

    args = parser.parse_args()

    # Parse environment list
    environments = [env.strip() for env in args.env.split(',')]

    print(f"Analyzing Ratio/Max for environments: {', '.join(environments)}")
    print()

    # Analyze each environment
    for env_name in environments:
        analyze_environment_ratio_max(env_name)

    print("\n" + "="*80)
    print("All Ratio/Max analyses completed!")

if __name__ == "__main__":
    main()
