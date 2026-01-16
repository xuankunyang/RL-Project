import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import argparse
from data_processor import LogLoader

# --- Style Configuration ---
sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
PALETTE = {
    'DQN_Vanilla': '#1f77b4', # Blue
    'DQN_Double': '#d62728',  # Red
    'DQN_Dueling': '#2ca02c', # Green
    'DQN_Rainbow': '#9467bd'  # Purple
}
VARIANT_NAMES = {
    'DQN_Vanilla': 'Vanilla DQN',
    'DQN_Double': 'Double DQN',
    'DQN_Dueling': 'Dueling DQN',
    'DQN_Rainbow': 'Rainbow DQN'
}

def plot_learning_curve(df, env_name, metric='eval_reward', output_dir='plots'):
    """
    Plots the learning curve for all variants in the given environment.
    """
    # Filter data
    subset = df[(df['env'] == env_name) & (df['metric'] == metric)]
    
    if subset.empty:
        print(f"No data found for {env_name} - {metric}")
        return

    plt.figure(figsize=(10, 6))
    
    # Map friendly names
    subset = subset.copy()
    subset['Variant'] = subset['variant'].map(VARIANT_NAMES).fillna(subset['variant'])
    
    # Plot
    sns.lineplot(
        data=subset, 
        x='step', 
        y='value', 
        hue='variant', 
        palette=PALETTE, 
        linewidth=2
    )
    
    plt.title(f"{env_name}: Learning Performance")
    plt.xlabel("Timesteps")
    plt.ylabel("Average Reward" if 'reward' in metric else metric)
    plt.legend(title="Algorithm")
    plt.tight_layout()
    
    filename = f"{env_name}_{metric}_comparison.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()
    print(f"Saved {filename}")

def plot_sensitivity(df, env_name, param='lr', metric='train_reward', output_dir='plots'):
    """
    Plots sensitivity analysis (Box Plot) for a specific parameter.
    """
    subset = df[(df['env'] == env_name) & (df['metric'] == metric)]
    
    # Filter for last 20% steps
    if subset.empty:
        return
        
    max_step = subset['step'].max()
    threshold = max_step * 0.8
    final_subset = subset[subset['step'] >= threshold]
    
    if final_subset.empty:
        print(f"No final data found for {env_name} sensitivity analysis")
        return

    plt.figure(figsize=(10, 6))
    
    if param not in final_subset.columns:
        print(f"Parameter {param} not found in data columns")
        return

    sns.boxplot(data=final_subset, x=param, y='value', hue='variant', palette=PALETTE)
    
    plt.title(f"{env_name}: Sensitivity to {param.upper()}")
    plt.ylabel("Final Average Reward")
    plt.xlabel(param.upper())
    plt.legend(title="Algorithm", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    filename = f"{env_name}_sensitivity_{param}.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()
    print(f"Saved {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="results", help="Root results directory")
    parser.add_argument("--out_dir", type=str, default="report/Figs_Unified", help="Output directory for plots")
    parser.add_argument("--env", type=str, default="ALE/Pong-v5", help="Environment to analyze")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 1. Load Data
    loader = LogLoader(args.data_dir, cache_file="all_data_cache.pkl")
    df = loader.get_dataframe()
    
    if df.empty:
        print("No data loaded. Please check data_dir.")
    else:
        # 2. Plot Learning Curves
        plot_learning_curve(df, args.env, metric='train_reward', output_dir=args.out_dir)
        plot_learning_curve(df, args.env, metric='eval_reward', output_dir=args.out_dir)
        
        # 3. Plot Sensitivity (Examples)
        plot_sensitivity(df, args.env, param='lr', output_dir=args.out_dir)
        plot_sensitivity(df, args.env, param='update_freq', output_dir=args.out_dir)
        plot_sensitivity(df, args.env, param='hidden_dim', output_dir=args.out_dir)