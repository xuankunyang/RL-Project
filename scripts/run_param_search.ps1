# PowerShell Script for Distributed Hyperparameter Search (Enhanced)

# Define parameter grids
$lrs = @(1e-4) # Simplified for variant comparison
$seeds = @(42, 101)
$algos = @("dqn", "ppo")
$dqn_types = @("dqn", "double", "dueling", "rainbow")
$devices = @("cuda:0", "cuda:1", "cuda:2", "cuda:3")

# Common settings
$timesteps = 2000000 # Increased for better convergence comparison
$base_cmd = "python run.py"

$job_id = 0

foreach ($algo in $algos) {
    if ($algo -eq "dqn") {
        $env = "BreakoutNoFrameskip-v4"
        # Iterate over DQN variants
        foreach ($dtype in $dqn_types) {
            foreach ($lr in $lrs) {
                foreach ($seed in $seeds) {
                    $device = $devices[$job_id % $devices.Length]
                    $exp_name = "search_${dtype}_lr${lr}"
                    
                    $cmd_args = "--algo dqn --dqn_type $dtype --env_name $env --lr $lr --seed $seed --device $device --total_timesteps $timesteps --exp_name $exp_name"
                    
                    Write-Host "Starting Job $job_id on $device: DQN-$dtype | LR: $lr | Seed: $seed"
                    Start-Process python -ArgumentList $cmd_args -NoNewWindow
                    $job_id++
                    Start-Sleep -Seconds 2
                }
            }
        }
    }
    else {
        # PPO: Standard vs No Clip
        $env = "HalfCheetah-v4"
        $clips = @(0.2, 10.0)
        
        foreach ($clip in $clips) {
            foreach ($lr in $lrs) {
                # Separate LRs test: For simplicity, keeping them same for now, or you can add nested loop
                # Just testing Clip
                foreach ($seed in $seeds) {
                    $device = $devices[$job_id % $devices.Length]
                    
                    if ($clip -gt 0.5) {
                        $exp_name = "search_ppo_noclip_lr${lr}"
                    }
                    else {
                        $exp_name = "search_ppo_clip${clip}_lr${lr}"
                    }
                    
                    # Assuming lr_actor = lr_critic = lr for this search
                    $cmd_args = "--algo ppo --env_name $env --lr $lr --ppo_clip $clip --seed $seed --device $device --total_timesteps $timesteps --exp_name $exp_name"
                    
                    Write-Host "Starting Job $job_id on $device: PPO (Clip $clip) | LR: $lr | Seed: $seed"
                    Start-Process python -ArgumentList $cmd_args -NoNewWindow
                    $job_id++
                    Start-Sleep -Seconds 2
                }
            }
        }
    }
}

Write-Host "All jobs submitted!"
