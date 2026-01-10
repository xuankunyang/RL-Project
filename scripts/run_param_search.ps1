# PowerShell Script for Distributed Hyperparameter Search

# Define parameter grids
$lrs = @(1e-4, 3e-4)
$seeds = @(42, 101)
$algos = @("dqn", "ppo")
$devices = @("cuda:0", "cuda:1", "cuda:2", "cuda:3")

# Common settings
$timesteps = 1000000
$base_cmd = "python run.py"

$job_id = 0

foreach ($algo in $algos) {
    if ($algo -eq "dqn") {
        $env = "BreakoutNoFrameskip-v4"
    } else {
        $env = "HalfCheetah-v4"
    }

    foreach ($lr in $lrs) {
        foreach ($seed in $seeds) {
            
            # Round-robin device assignment
            $device = $devices[$job_id % $devices.Length]
            
            $exp_name = "search_lr${lr}"
            
            $cmd_args = "--algo $algo --env_name $env --lr $lr --seed $seed --device $device --total_timesteps $timesteps --exp_name $exp_name"
            
            Write-Host "Starting Job $job_id on $device: $algo | LR: $lr | Seed: $seed"
            
            # Start process in background
            Start-Process python -ArgumentList $cmd_args -NoNewWindow
            
            $job_id++
            
            # Optional: Add small delay to prevent startup race conditions
            Start-Sleep -Seconds 2
        }
    }
}

Write-Host "All jobs submitted!"
