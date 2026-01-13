import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def parse_exp_dir(exp_dir):
    """Parse experiment directory name to extract parameters."""
    # lra0.0001_lrc0.0001_clp0.2_sd42_hd256_env16_20260112-030438
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

def extract_scalars_from_event_file(event_file, scalar_name='Train/EpisodeReward'):
    """Extract scalars from TensorBoard event file."""
    try:
        ea = EventAccumulator(event_file)
        ea.Reload()
        scalars = ea.Scalars(scalar_name)
        return [(s.step, s.value) for s in scalars]
    except Exception as e:
        print(f"Error reading {event_file}: {e}")
        return []

def load_all_experiments(results_dir='results/MuJoCo/HalfCheetah-v4/PPO_Standard'):
    """Load all experiment data."""
    all_data = []

    for exp_dir in os.listdir(results_dir):
        exp_path = os.path.join(results_dir, exp_dir)
        if not os.path.isdir(exp_path):
            continue

        params = parse_exp_dir(exp_dir)
        if not params:
            continue

        event_file = os.path.join(exp_path, 'events.out.tfevents.1768145273.DVLab-7.4119890.0')  # Adjust pattern if needed
        # Actually, find the event file
        event_files = [f for f in os.listdir(exp_path) if f.startswith('events.out.tfevents')]
        if not event_files:
            continue
        event_file = os.path.join(exp_path, event_files[0])

        scalars = extract_scalars_from_event_file(event_file)
        if not scalars:
            continue

        for step, value in scalars:
            all_data.append({
                'step': step,
                'value': value,
                **params
            })

    return pd.DataFrame(all_data)

def plot_selected_curves(df, perf_df, top_n=3):
    """Plot curves for top N configurations."""
    top_configs = perf_df.nlargest(top_n, 'final_reward')

    fig, ax = plt.subplots(figsize=(12, 8))

    colors = ['blue', 'red', 'green', 'orange', 'purple'][:top_n]

    for i, (_, config) in enumerate(top_configs.iterrows()):
        subset = df[(df['lr_actor'] == config['lr_actor']) &
                   (df['lr_critic'] == config['lr_critic']) &
                   (df['hidden_dim'] == config['hidden_dim']) &
                   (df['seed'] == config['seed'])]

        if not subset.empty:
            label = f'A{config["lr_actor"]:.0e}_C{config["lr_critic"]:.0e}_H{int(config["hidden_dim"])}_S{int(config["seed"])}'
            ax.plot(subset['step'], subset['value'], color=colors[i], linewidth=2, label=label)

    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Episode Reward')
    ax.set_title(f'PPO Training Curves - Top {top_n} Configurations')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/MuJoCo/HalfCheetah-v4/PPO_Standard/analysis/ppo_top_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_lr_comparison(df):
    """Plot average curves by LR pair."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    lr_pairs = sorted(df[['lr_actor', 'lr_critic']].drop_duplicates().values.tolist())

    for i, (lr_a, lr_c) in enumerate(lr_pairs):
        ax = axes[i]

        # Average over seeds and hidden_dims
        subset = df[(df['lr_actor'] == lr_a) & (df['lr_critic'] == lr_c)]
        if subset.empty:
            continue

        # Group by step, average value
        avg_curve = subset.groupby('step')['value'].mean().reset_index()

        ax.plot(avg_curve['step'], avg_curve['value'], linewidth=2, label=f'A{lr_a:.0e}_C{lr_c:.0e}')
        ax.fill_between(avg_curve['step'],
                        subset.groupby('step')['value'].min(),
                        subset.groupby('step')['value'].max(),
                        alpha=0.2)

        ax.set_title(f'LR: A{lr_a:.0e} C{lr_c:.0e}')
        ax.set_xlabel('Steps')
        ax.set_ylabel('Reward')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/MuJoCo/HalfCheetah-v4/PPO_Standard/analysis/ppo_lr_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_hidden_dim_comparison(df):
    """Compare hidden dimensions."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    for hd in [256, 512]:
        subset = df[df['hidden_dim'] == hd]
        if subset.empty:
            continue

        # Average over all configs
        avg_curve = subset.groupby('step')['value'].mean().reset_index()

        ax1.plot(avg_curve['step'], avg_curve['value'], linewidth=2, label=f'Hidden {hd}')
        ax1.fill_between(avg_curve['step'],
                        subset.groupby('step')['value'].min(),
                        subset.groupby('step')['value'].max(),
                        alpha=0.2)

    ax1.set_title('Hidden Dimension Comparison (All Configs)')
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Episode Reward')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Final performance box plot
    final_rewards = []
    labels = []
    for hd in [256, 512]:
        subset = df[df['hidden_dim'] == hd]
        if not subset.empty:
            # Take last 100k steps average per experiment
            late_rewards = subset[subset['step'] > 1500000]['value']
            if not late_rewards.empty:
                final_rewards.append(late_rewards.values)
                labels.append(f'Hidden {hd}')

    ax2.boxplot(final_rewards, labels=labels)
    ax2.set_title('Final Performance Comparison')
    ax2.set_ylabel('Episode Reward')

    plt.tight_layout()
    plt.savefig('ppo_hidden_dim_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_parameter_effects(df):
    """Analyze parameter effects on final performance."""
    # Calculate final performance (average reward in last 500k steps)
    final_perf = []
    for _, group in df.groupby(['lr_actor', 'lr_critic', 'hidden_dim', 'seed']):
        late_rewards = group[group['step'] > 1500000]['value']
        if not late_rewards.empty:
            final_perf.append({
                'lr_actor': group['lr_actor'].iloc[0],
                'lr_critic': group['lr_critic'].iloc[0],
                'hidden_dim': group['hidden_dim'].iloc[0],
                'seed': group['seed'].iloc[0],
                'final_reward': late_rewards.mean(),
                'convergence_speed': group['step'].min() if late_rewards.mean() > 1000 else float('inf')  # Rough estimate
            })

    perf_df = pd.DataFrame(final_perf)

    # Print analysis
    print("\n=== Parameter Analysis ===")

    # LR effects
    print("\nLR Effects:")
    lr_summary = perf_df.groupby(['lr_actor', 'lr_critic'])['final_reward'].agg(['mean', 'std', 'count']).round(2)
    print(lr_summary)

    # Hidden dim effects
    print("\nHidden Dimension Effects:")
    hd_summary = perf_df.groupby('hidden_dim')['final_reward'].agg(['mean', 'std', 'count']).round(2)
    print(hd_summary)

    # Stability (variance across seeds)
    print("\nStability Analysis (variance across seeds):")
    stability = perf_df.groupby(['lr_actor', 'lr_critic'])['final_reward'].std().round(2)
    print(stability)

    return perf_df

def main():
    print("Loading PPO experiment data...")
    df = load_all_experiments()

    if df.empty:
        print("No data found!")
        return

    print(f"Loaded {len(df)} data points from {len(df.groupby(['lr_actor', 'lr_critic', 'hidden_dim', 'seed']))} experiments")

    print("Analyzing parameter effects...")
    perf_df = analyze_parameter_effects(df)

    print("Plotting selected curves...")
    plot_selected_curves(df, perf_df, top_n=3)

    print("Plotting hidden dimension comparison...")
    plot_hidden_dim_comparison(df)

    print("Analysis complete! Check analysis folder for CSV and PNG files.")

if __name__ == "__main__":
    main()
