import os
import glob
import re
import pandas as pd
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pickle

class LogLoader:
    def __init__(self, base_dir, cache_file="data_cache.pkl"):
        self.base_dir = base_dir
        self.cache_file = cache_file
        self.data = [] # List of dicts

    def parse_dqn_folder(self, folder_name):
        """
        Parses DQN folder: lr{}_uf{}_sd{}_hd{}_bs{}_env{}_{timestamp}
        """
        params = {}
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
                val = match.group(1)
                try:
                    params[key] = float(val) if ('.' in val or 'e' in val) else int(val)
                except:
                    params[key] = val
        return params

    def parse_ppo_folder(self, folder_name):
        """
        Parses PPO folder: lra{}_lrc{}_clp{}_sd{}_hd{}_env{}_{timestamp}
        """
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
                val = match.group(1)
                try:
                    params[key] = float(val) if ('.' in val or 'e' in val) else int(val)
                except:
                    params[key] = val
        return params

    def load_from_cache(self):
        if os.path.exists(self.cache_file):
            print(f"Loading data from cache: {self.cache_file}")
            with open(self.cache_file, 'rb') as f:
                self.data = pickle.load(f)
            return True
        return False

    def scan_and_load(self, force_reload=False):
        if not force_reload and self.load_from_cache():
            return self.data

        print(f"Scanning {self.base_dir} for tfevents...")
        # Use recursive glob to find all tfevents
        event_files = glob.glob(os.path.join(self.base_dir, "**", "*tfevents*"), recursive=True)
        
        if not event_files:
            print("No tfevents files found! Check your directory structure.")
            return []

        processed_data = []

        for ef in event_files:
            try:
                dirname = os.path.dirname(ef)
                folder_name = os.path.basename(dirname)
                parent_dir = os.path.dirname(dirname)
                variant = os.path.basename(parent_dir) # e.g., DQN_Vanilla, PPO_Standard
                env_name = os.path.basename(os.path.dirname(parent_dir)) # e.g., Breakout-v5
                domain = os.path.basename(os.path.dirname(os.path.dirname(parent_dir))) # e.g., Atari

                # Determine Algo Type based on path or folder name
                if "DQN" in variant or "dqn" in folder_name.lower():
                    params = self.parse_dqn_folder(folder_name)
                    algo_type = "DQN"
                elif "PPO" in variant or "ppo" in folder_name.lower():
                    params = self.parse_ppo_folder(folder_name)
                    algo_type = "PPO"
                else:
                    params = {}
                    algo_type = "Unknown"

                # Meta info
                meta = {
                    'path': dirname,
                    'variant': variant,
                    'env': env_name,
                    'domain': domain,
                    'algo': algo_type
                }
                meta.update(params)

                # Read TensorBoard
                ea = EventAccumulator(ef)
                ea.Reload()
                tags = ea.Tags()['scalars']

                # Extract desired metrics (downsampled if needed)
                # 1. Train Reward
                if 'Train/EpisodeReward' in tags:
                    for e in ea.Scalars('Train/EpisodeReward'):
                        row = meta.copy()
                        row['step'] = e.step
                        row['value'] = e.value
                        row['metric'] = 'train_reward'
                        processed_data.append(row)
                
                # 2. Eval Reward
                if 'Eval/MeanReward' in tags:
                    for e in ea.Scalars('Eval/MeanReward'):
                        row = meta.copy()
                        row['step'] = e.step
                        row['value'] = e.value
                        row['metric'] = 'eval_reward'
                        processed_data.append(row)

                # 3. Loss (Sampled)
                loss_tags = ['Loss/DQN', 'Loss/Policy', 'Loss/Value', 'Loss/Entropy']
                for lt in loss_tags:
                    if lt in tags:
                        events = ea.Scalars(lt)
                        # Downsample: take every 10th point to save memory
                        for i, e in enumerate(events):
                            if i % 10 == 0:
                                row = meta.copy()
                                row['step'] = e.step
                                row['value'] = e.value
                                row['metric'] = lt.replace('/', '_').lower() # loss_dqn
                                processed_data.append(row)
            
            except Exception as e:
                print(f"Error processing {ef}: {e}")

        self.data = processed_data
        
        # Save cache
        print(f"Saving {len(self.data)} records to cache...")
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.data, f)
        
        return self.data

    def get_dataframe(self):
        if not self.data:
            self.scan_and_load()
        return pd.DataFrame(self.data)

if __name__ == "__main__":
    # Test run
    # Assume results are extracted to 'results' folder in current dir
    loader = LogLoader("results") 
    df = loader.get_dataframe()
    if not df.empty:
        print(df.head())
        print(df.info())
        print("Unique Envs:", df['env'].unique())
    else:
        print("DataFrame is empty.")