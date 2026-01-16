import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import argparse
from data_processor import LogLoader

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

def plot_ppo_curve(df, env_name, metric='train_reward', output_dir='plots'):
    subset = df[(df['env'] == env_name) & (df['metric'] == metric)]
    
    if subset.empty:
        return

    plt.figure(figsize=(10, 6))
    
    # For PPO, we might compare different clip ratios or LRs
    # Let's try to create a 'Configuration' column for hue
    # Example: "Clip=0.2"
    
    subset = subset.copy()
    if 'clip' in subset.columns:
        subset['Config'] = subset['clip'].apply(lambda x: f"Clip={x}")
    else:
        subset['Config'] = "Standard"

    sns.lineplot(data=subset, x='step', y='value', hue='Config', style='variant')
    
    plt.title(f"{env_name} (PPO): Performance")
    plt.xlabel("Timesteps")
    plt.ylabel(metric)
    plt.tight_layout()
    
    filename = f"{env_name}_ppo_{metric}.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()
    print(f"Saved {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="results", help="Root results directory")
    parser.add_argument("--out_dir", type=str, default="report/Figs_Unified", help="Output directory")
    parser.add_argument("--env", type=str, default="HalfCheetah-v4", help="Environment name")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    loader = LogLoader(args.data_dir, cache_file="all_data_cache.pkl")
    df = loader.get_dataframe()
    
    plot_ppo_curve(df, args.env, metric='train_reward', output_dir=args.out_dir)
    plot_ppo_curve(df, args.env, metric='eval_reward', output_dir=args.out_dir)