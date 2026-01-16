import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import argparse
from data_processor import LogLoader

sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)

def plot_ppo_curve(df, env_filter, output_dir):
    subset = df[
        (df['env'].str.contains(env_filter, case=False, na=False)) & 
        (df['metric_type'] == 'reward') &
        (df['algo'] == 'PPO')
    ]
    
    if subset.empty:
        print(f"No PPO data for {env_filter}")
        return

    # 优先选择 ep_rew_mean
    if 'ep_rew_mean' in subset['metric'].unique():
        subset = subset[subset['metric'] == 'ep_rew_mean']

    plt.figure(figsize=(10, 6))
    
    # 构造标签：如果有 clip 参数变化，显示 Clip，否则显示 Standard
    # 也可以显示 lr_actor
    if 'clip' in subset.columns and subset['clip'].nunique() > 1:
        subset['Label'] = subset['clip'].apply(lambda x: f"Clip={x}")
        hue_col = 'Label'
    elif 'lr_actor' in subset.columns and subset['lr_actor'].nunique() > 1:
        subset['Label'] = subset['lr_actor'].apply(lambda x: f"LR_A={x}")
        hue_col = 'Label'
    else:
        subset['Label'] = "PPO"
        hue_col = 'Label'

    sns.lineplot(data=subset, x='step', y='value', hue=hue_col, style='variant', linewidth=2)
    
    env_name = subset['env'].iloc[0] if not subset['env'].empty else env_filter
    plt.title(f"{env_name} (PPO): Performance")
    plt.xlabel("Timesteps")
    plt.ylabel("Average Reward")
    plt.tight_layout()
    
    filename = f"{env_filter}_ppo_curve.png"
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="results")
    parser.add_argument("--out_dir", type=str, default="report/Figs_Unified")
    parser.add_argument("--env", type=str, default="Cheetah", help="Environment name filter")
    parser.add_argument("--force_reload", action="store_true")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    loader = LogLoader(args.data_dir, cache_file="ppo_data_cache.pkl")
    df = loader.scan_and_load(force_reload=args.force_reload)
    
    if not df.empty:
        plot_ppo_curve(df, args.env, args.out_dir)
