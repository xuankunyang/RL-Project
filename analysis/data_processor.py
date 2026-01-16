import os
import glob
import re
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pickle

class LogLoader:
    def __init__(self, base_dir, cache_file="data_cache.pkl"):
        self.base_dir = base_dir
        self.cache_file = cache_file
        self.data = []

    def parse_dqn_folder(self, folder_name):
        """解析 DQN 文件夹名中的超参数"""
        params = {}
        # 匹配模式: lr0.0001_uf10_sd42...
        patterns = {
            'lr': r'lr([\d\.e-]+)',
            'update_freq': r'uf(\d+)',
            'seed': r'sd(\d+)',
            'hidden_dim': r'hd(\d+)',
            'batch_size': r'bs(\d+)',
            'num_envs': r'env(\d+)'
        }
        for key, pattern in patterns.items():
            match = re.search(pattern, folder_name)
            if match:
                try:
                    val = match.group(1)
                    params[key] = float(val) if ('.' in val or 'e' in val) else int(val)
                except:
                    pass
        return params

    def parse_ppo_folder(self, folder_name):
        """解析 PPO 文件夹名中的超参数"""
        params = {}
        patterns = {
            'lr_actor': r'lra([\d\.e-]+)',
            'lr_critic': r'lrc([\d\.e-]+)',
            'clip': r'clp([\d\.e-]+)',
            'seed': r'sd(\d+)',
            'hidden_dim': r'hd(\d+)',
            'num_envs': r'env(\d+)'
        }
        for key, pattern in patterns.items():
            match = re.search(pattern, folder_name)
            if match:
                try:
                    val = match.group(1)
                    params[key] = float(val) if ('.' in val or 'e' in val) else int(val)
                except:
                    pass
        return params

    def load_from_cache(self):
        if os.path.exists(self.cache_file):
            print(f"Loading data from cache: {self.cache_file}")
            try:
                with open(self.cache_file, 'rb') as f:
                    self.data = pickle.load(f)
                return True
            except Exception as e:
                print(f"Failed to load cache: {e}")
                return False
        return False

    def scan_and_load(self, force_reload=False):
        if not force_reload and self.load_from_cache():
            return pd.DataFrame(self.data)

        print(f"Scanning {self.base_dir} for tfevents...")
        # 递归查找所有 tfevents 文件
        event_files = glob.glob(os.path.join(self.base_dir, "**", "*tfevents*"), recursive=True)
        
        if not event_files:
            print("No tfevents files found! Check your data directory.")
            return pd.DataFrame()

        processed_data = []
        print(f"Found {len(event_files)} log files. Processing...")

        for ef in event_files:
            try:
                dirname = os.path.dirname(ef)
                folder_name = os.path.basename(dirname)
                
                # 路径结构分析
                # 预期: results/Atari/ALE/Breakout-v5/DQN_Vanilla/lr.../events...
                path_parts = os.path.normpath(ef).split(os.sep)
                
                # 提取环境名和算法名
                env_name = "Unknown"
                algo_type = "Unknown"
                variant = "Unknown"

                # 常见环境名关键词
                known_envs = ["Pong", "Breakout", "Cheetah", "Ant", "Hopper", "Walker"]
                for part in path_parts:
                    for ke in known_envs:
                        if ke in part:
                            env_name = part # e.g., Pong-v5
                            break
                
                # 算法判断
                for part in path_parts:
                    if "DQN" in part:
                        algo_type = "DQN"
                        variant = part # e.g., DQN_Rainbow
                    elif "PPO" in part:
                        algo_type = "PPO"
                        variant = part

                # 解析参数
                if algo_type == "DQN":
                    params = self.parse_dqn_folder(folder_name)
                elif algo_type == "PPO":
                    params = self.parse_ppo_folder(folder_name)
                else:
                    params = {}

                meta = {
                    'path': dirname,
                    'variant': variant,
                    'env': env_name,
                    'algo': algo_type
                }
                meta.update(params)

                # 读取 TensorBoard 数据
                ea = EventAccumulator(ef)
                ea.Reload()
                
                # 检查 tags
                if 'scalars' not in ea.Tags():
                    continue
                    
                tags = ea.Tags()['scalars']

                # 提取 Reward 相关指标
                # 常见 tag: 'Train/EpisodeReward', 'Eval/MeanReward', 'rollout/ep_rew_mean'
                for tag in tags:
                    is_reward = False
                    metric_name = tag.split('/')[-1].lower()
                    
                    if 'reward' in metric_name or 'rew' in metric_name:
                        is_reward = True
                    
                    # 如果不是 reward，可以根据需要添加 loss 等其他指标的过滤
                    if not is_reward and 'loss' not in metric_name:
                        continue

                    events = ea.Scalars(tag)
                    if not events:
                        continue
                        
                    # 降采样：每隔几个点取一个，防止数据量爆炸
                    # 保证至少取 1000 个点，如果总数小于 1000 则全取
                    step_interval = max(1, len(events) // 1000) 
                    
                    for i, e in enumerate(events):
                        if i % step_interval == 0:
                            row = meta.copy()
                            row['step'] = e.step
                            row['value'] = e.value
                            row['metric'] = metric_name
                            row['full_tag'] = tag
                            if is_reward:
                                row['metric_type'] = 'reward'
                            elif 'loss' in metric_name:
                                row['metric_type'] = 'loss'
                            else:
                                row['metric_type'] = 'other'
                                
                            processed_data.append(row)

            except Exception as e:
                print(f"Error processing {ef}: {e}")

        self.data = processed_data
        
        print(f"Saving {len(self.data)} records to cache...")
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.data, f)
        except Exception as e:
            print(f"Warning: Could not save cache: {e}")
        
        return pd.DataFrame(self.data)

if __name__ == "__main__":
    # 测试代码
    loader = LogLoader("results")
    # 如果没有 results 文件夹，这个测试会打印找不到文件
    if os.path.exists("results"):
        df = loader.scan_and_load()
        if not df.empty:
            print(df.head())
            print("Unique Envs:", df['env'].unique())
            print("Unique Algos:", df['algo'].unique())
    else:
        print("Note: 'results' directory not found locally. Run this after downloading data.")
