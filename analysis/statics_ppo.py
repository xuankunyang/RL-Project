import os
import re
import numpy as np
import argparse
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

def get_env_config(env_name):
    """Get configuration for a specific environment."""
    configs = {
        'HalfCheetah-v4': {
            'results_dir': 'results/MuJoCo/HalfCheetah-v4/PPO_Standard',
            'default_clip': 0.2,
            'total_steps': 2000000
        },
        'Hopper-v4': {
            'results_dir': 'results/MuJoCo/Hopper-v4/PPO_Standard',
            'default_clip': 0.1,
            'total_steps': 2000000
        }
    }
    return configs.get(env_name, configs['HalfCheetah-v4'])

def analyze_environment(env_name, output_file=None):
    """Analyze PPO results for a specific environment."""
    config = get_env_config(env_name)
    results_dir = config['results_dir']
    total_steps = config['total_steps']
    default_clip = config['default_clip']

    output_lines = []
    output_lines.append(f"PPO Parameter Analysis Results for {env_name}")
    output_lines.append("=" * (40 + len(env_name)))
    output_lines.append(f"Default Clip: {default_clip}")
    output_lines.append("")

    # Group experiments by parameter combinations
    param_groups = {}

    if not os.path.exists(results_dir):
        output_lines.append(f"Results directory not found: {results_dir}")
        output_lines.append("")
    else:
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

    if not param_groups:
        output_lines.append("No valid experiments found")
        output_lines.append("")
    else:
        # Process each parameter group
        for (lr_actor, lr_critic, hidden_dim), exps in param_groups.items():
            output_lines.append(f"Parameter Group: Actor LR={lr_actor:.0e}, Critic LR={lr_critic:.0e}, Hidden Dim={hidden_dim}")

            train_means = []
            eval_means = []

            for seed, exp_path in exps:
                output_lines.append(f"  Processing seed {seed}...")
                train_mean, eval_mean = process_experiment(exp_path, total_steps)

                if train_mean is not None:
                    train_means.append(train_mean)
                    output_lines.append(f"    Train last 10% mean: {train_mean:.2f}")
                else:
                    output_lines.append("    Train data not available")

                if eval_mean is not None:
                    eval_means.append(eval_mean)
                    output_lines.append(f"    Eval last 10% mean: {eval_mean:.2f}")
                else:
                    output_lines.append("    Eval data not available")

            # Compute statistics across seeds
            if train_means:
                train_avg = np.mean(train_means)
                train_std = np.std(train_means, ddof=1) if len(train_means) > 1 else 0
                output_lines.append(f"  Train: Mean={train_avg:.2f}, Std={train_std:.2f} (across {len(train_means)} seeds)")
            else:
                output_lines.append("  Train: No data available")

            if eval_means:
                eval_avg = np.mean(eval_means)
                eval_std = np.std(eval_means, ddof=1) if len(eval_means) > 1 else 0
                output_lines.append(f"  Eval: Mean={eval_avg:.2f}, Std={eval_std:.2f} (across {len(eval_means)} seeds)")
            else:
                output_lines.append("  Eval: No data available")

            output_lines.append("")

        output_lines.append("")

    # Print to console
    for line in output_lines:
        print(line)

    # Save to file if specified
    if output_file:
        with open(output_file, 'w') as f:
            f.write('\n'.join(output_lines))
        print(f"Results saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Analyze PPO parameter search results across environments')
    parser.add_argument('--env', '--environments', type=str, default='HalfCheetah-v4',
                        help='Environment name(s), comma-separated (e.g., "HalfCheetah-v4,Hopper-v4")')
    parser.add_argument('--output_file', type=str, help='Output file to save results')

    args = parser.parse_args()

    # Parse environment list
    environments = [env.strip() for env in args.env.split(',')]

    print(f"Analyzing PPO results for environments: {', '.join(environments)}")
    print()

    # Analyze each environment
    for env_name in environments:
        analyze_environment(env_name, args.output_file)
        if len(environments) > 1:
            print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()
