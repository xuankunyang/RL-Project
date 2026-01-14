import os
import re
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def parse_exp_dir(exp_dir):
    """Parse experiment directory name to extract parameters."""
    # Pattern: lra{}_lrc{}_clp{}_sd{}_hd{}_env{}_{timestamp}
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
    results_dir = 'results/MuJoCo/HalfCheetah-v4/PPO_Standard'
    total_steps = 2000000

    # Group experiments by parameter combinations
    param_groups = {}

    for exp_dir in os.listdir(results_dir):
        exp_path = os.path.join(results_dir, exp_dir)
        if not os.path.isdir(exp_path):
            continue

        params = parse_exp_dir(exp_dir)
        if not params:
            continue

        key = (params['lr_actor'], params['lr_critic'], params['hidden_dim'])
        if key not in param_groups:
            param_groups[key] = []
        param_groups[key].append((params['seed'], exp_path))

    # Process each parameter group
    print("PPO Parameter Analysis Results")
    print("=" * 50)

    for (lr_actor, lr_critic, hidden_dim), exps in param_groups.items():
        print(f"\nParameter Group: Actor LR={lr_actor:.0e}, Critic LR={lr_critic:.0e}, Hidden Dim={hidden_dim}")

        train_means = []
        eval_means = []

        for seed, exp_path in exps:
            print(f"  Processing seed {seed}...")
            train_mean, eval_mean = process_experiment(exp_path, total_steps)

            if train_mean is not None:
                train_means.append(train_mean)
                print(f"    Train last 10% mean: {train_mean:.2f}")
            else:
                print(f"    Train data not available")

            if eval_mean is not None:
                eval_means.append(eval_mean)
                print(f"    Eval last 10% mean: {eval_mean:.2f}")
            else:
                print(f"    Eval data not available")

        # Compute statistics across seeds
        if train_means:
            train_avg = np.mean(train_means)
            train_std = np.std(train_means, ddof=1) if len(train_means) > 1 else 0
            print(f"  Train: Mean={train_avg:.2f}, Std={train_std:.2f} (across {len(train_means)} seeds)")
        else:
            print("  Train: No data available")

        if eval_means:
            eval_avg = np.mean(eval_means)
            eval_std = np.std(eval_means, ddof=1) if len(eval_means) > 1 else 0
            print(f"  Eval: Mean={eval_avg:.2f}, Std={eval_std:.2f} (across {len(eval_means)} seeds)")
        else:
            print("  Eval: No data available")

if __name__ == "__main__":
    main()
