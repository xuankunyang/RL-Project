import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Set white background style
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['savefig.edgecolor'] = 'white'

def plot_ratio_max(cached_data, save_path='results/MuJoCo/HalfCheetah-v4/PPO_Standard/analysis/Ratio_Max_detailed.png'):
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
    # ax.set_title('Ratio/Max: Scatter Plot + Mean Line', fontsize=14, fontweight='bold')
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

def main():
    cache_file = 'results/MuJoCo/HalfCheetah-v4/PPO_Standard/analysis/ppo_selected_cache.pkl'

    if not os.path.exists(cache_file):
        print(f"Cache file not found: {cache_file}")
        print("Please run analysis/plot_selected_ppo.py --reload_cache first")
        return

    print("Loading cached data...")
    with open(cache_file, 'rb') as f:
        cached_data = pickle.load(f)

    print("Plotting Ratio_Max...")
    plot_ratio_max(cached_data)

    print("Done!")

if __name__ == "__main__":
    main()
