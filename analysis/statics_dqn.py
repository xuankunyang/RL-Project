import os
import re
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

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

def extract_last_10_percent_mean(scalars, total_steps=2000000):
    """Extract the last 10% of scalars and compute mean."""
    if not scalars:
        return None

    # Sort by step
    scalars.sort(key=lambda x: x.step)

    # Get last 10% of steps
    last_10_start = int(0.9 * total_steps)
    last_10_scalars = [s for s in scalars if s.step >= last_10_start]

    if not last_10_scalars:
        # If no scalars in last 10%, take the last few
        last_10_scalars = scalars[-max(1, len(scalars)//10):]

    values = [s.value for s in last_10_scalars]
    return np.mean(values) if values else None

def process_experiment(exp_path, total_steps=2000000):
    """Process a single experiment directory."""
    # Find event file
    event_files = [f for f in os.listdir(exp_path) if f.startswith('events.out.tfevents')]
    if not event_files:
        return None, None

    event_file = os.path.join(exp_path, event_files[0])

    try:
        ea = EventAccumulator(event_file)
        ea.Reload()

        # Extract train rewards
        train_scalars = ea.Scalars('Train/EpisodeReward')
        train_mean = extract_last_10_percent_mean(train_scalars, total_steps)

        # Extract eval rewards
        eval_scalars = ea.Scalars('Eval/MeanReward')
        eval_mean = extract_last_10_percent_mean(eval_scalars, total_steps)

        return train_mean, eval_mean
    except Exception as e:
        print(f"Error processing {exp_path}: {e}")
        return None, None

def main():
    results_base_dir = 'results/Atari/ALE/Breakout-v5'
    total_steps = 2000000

    # DQN variant directories to check
    dqn_variants = ['DQN_Vanilla', 'DQN_Double', 'DQN_Dueling', 'DQN_Rainbow']

    print("DQN Parameter Analysis Results")
    print("=" * 50)

    for variant in dqn_variants:
        variant_dir = os.path.join(results_base_dir, variant)
        if not os.path.exists(variant_dir):
            print(f"\nDQN Variant: {variant} (No experiments found)")
            continue

        print(f"\nDQN Variant: {variant}")
        print("-" * 30)

        # Group experiments by parameter combinations
        param_groups = {}

        for exp_dir in os.listdir(variant_dir):
            exp_path = os.path.join(variant_dir, exp_dir)
            if not os.path.isdir(exp_path):
                continue

            params = parse_dqn_exp_dir(exp_dir)
            if not params:
                continue

            key = (params['lr'], params['update_freq'], params['hidden_dim'])
            if key not in param_groups:
                param_groups[key] = []
            param_groups[key].append((params['seed'], exp_path))

        if not param_groups:
            print("No valid experiments found")
            continue

        # Process each parameter group
        for (lr, update_freq, hidden_dim), exps in param_groups.items():
            print(f"\nParameter Group: LR={lr:.0e}, Update Freq={update_freq}, Hidden Dim={hidden_dim}")

            # Since only one seed is used, just process the single experiment
            if len(exps) != 1:
                print(f"  Warning: Expected 1 experiment, found {len(exps)}")

            for seed, exp_path in exps:
                print(f"  Processing seed {seed}...")
                train_mean, eval_mean = process_experiment(exp_path, total_steps)

                if train_mean is not None:
                    print(f"    Train last 10% mean: {train_mean:.2f}")
                else:
                    print("    Train data not available")

                if eval_mean is not None:
                    print(f"    Eval last 10% mean: {eval_mean:.2f}")
                else:
                    print("    Eval data not available")

if __name__ == "__main__":
    main()
