import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import argparse
from data_processor import LogLoader

# 设置绘图风格
sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
PALETTE = {
    'DQN_Vanilla': '#1f77b4', # Blue
    'DQN_Double': '#d62728',  # Red
    'DQN_Dueling': '#2ca02c', # Green
    'DQN_Rainbow': '#9467bd'  # Purple
}

def plot_learning_curve(df, env_filter, output_dir):
    """绘制学习曲线"""
    # 过滤环境 (支持部分匹配，如 'Pong')
    # 过滤 metric_type 为 reward
    subset = df[
        (df['env'].str.contains(env_filter, case=False, na=False)) & 
        (df['metric_type'] == 'reward')
    ]
    
    # 进一步清洗：通常我们关注 eval 结果或者 train episode reward
    # 如果有 Eval/MeanReward 优先使用，否则使用 Train/EpisodeReward
    if 'meanreward' in subset['metric'].unique():
        subset = subset[subset['metric'] == 'meanreward']
    elif 'episodereward' in subset['metric'].unique():
        subset = subset[subset['metric'] == 'episodereward']
    
    if subset.empty:
        print(f"No reward data found for filter '{env_filter}'")
        return

    plt.figure(figsize=(10, 6))
    
    # 绘制曲线
    # ci='sd' 表示绘制标准差阴影，estimator='mean' 表示绘制均值线
    sns.lineplot(
        data=subset, 
        x='step', 
        y='value', 
        hue='variant', 
        palette=PALETTE, 
        linewidth=2,
        errorbar='sd' 
    )
    
    # 格式化
    env_name = subset['env'].iloc[0] if not subset['env'].empty else env_filter
    plt.title(f"{env_name}: Learning Performance")
    plt.xlabel("Timesteps")
    plt.ylabel("Average Reward")
    plt.legend(title="Algorithm")
    plt.tight_layout()
    
    # 保存
    filename = f"{env_filter}_learning_curve.png"
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved plot to {save_path}")

def plot_ablation_study(df, env_filter, output_dir):
    """(可选) 绘制消融实验对比图，例如不同 hidden_dim"""
    # 这是一个示例，展示如何针对特定超参数绘图
    subset = df[
        (df['env'].str.contains(env_filter, case=False, na=False)) & 
        (df['metric_type'] == 'reward') &
        (df['algo'] == 'DQN')
    ]
    
    if subset.empty or 'hidden_dim' not in subset.columns:
        return

    # 检查 hidden_dim 是否有多个值
    if len(subset['hidden_dim'].unique()) <= 1:
        return

    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=subset,
        x='step',
        y='value',
        hue='hidden_dim',
        style='variant',
        palette='viridis'
    )
    plt.title(f"{env_filter}: Hidden Dimension Sensitivity")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{env_filter}_hidden_dim.png"), dpi=300)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="results")
    parser.add_argument("--out_dir", type=str, default="report/Figs_Unified")
    parser.add_argument("--env", type=str, default="Pong", help="Environment name filter (e.g., Pong, Breakout)")
    parser.add_argument("--force_reload", action="store_true", help="Force reload from raw tfevents")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    loader = LogLoader(args.data_dir, cache_file="dqn_data_cache.pkl")
    df = loader.scan_and_load(force_reload=args.force_reload)
    
    if not df.empty:
        print(f"Generating plots for {args.env}...")
        plot_learning_curve(df, args.env, args.out_dir)
        # plot_ablation_study(df, args.env, args.out_dir)
    else:
        print("No data loaded.")
