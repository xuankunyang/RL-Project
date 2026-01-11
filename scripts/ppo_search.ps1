# PPO Hyperparameter Search with limited parallel jobs

# === Parameter grids ===
$lr_actors = @(1e-4, 3e-4, 5e-4)
$lr_critics = @(1e-4, 3e-4, 5e-4)
$hidden_dims = @(256, 512)
$ppo_clips = @(0.2, 0.3)
$seeds = @(42, 101)
$devices = @("cuda:0", "cuda:1", "cuda:2", "cuda:3")

# Max concurrent jobs
$max_jobs = 4
$running_jobs = @()

$job_id = 0

$env = "HalfCheetah-v4"

foreach ($clip in $ppo_clips) {
    foreach ($lr_a in $lr_actors) {
        foreach ($lr_c in $lr_critics) {
            foreach ($hd in $hidden_dims) {
                $device = $devices[$job_id % $devices.Length]
                $exp_name = "PPO_clip${clip}_lra${lr_a}_lrc${lr_c}_hd${hd}_sd42"

                $cmd_args = "--algo ppo --env_name $env --lr_actor $lr_a --lr_critic $lr_c --hidden_dim_ppo $hd --ppo_clip $clip --num_envs 1 --device $device --total_timesteps 2000000 --exp_name $exp_name"

                Write-Host "Starting Job $job_id on $device : PPO | Clip: $clip | LRA: $lr_a | LRC: $lr_c | HD: $hd"
                $proc = Start-Process python -ArgumentList $cmd_args -PassThru -NoNewWindow
                $running_jobs += $proc
                $job_id++

                # Limit parallel jobs
                while ($running_jobs.Count -ge $max_jobs) {
                    $running_jobs = $running_jobs | Where-Object { -not $_.HasExited }
                    Start-Sleep -Seconds 1
                }
            }
        }
    }
}

# Wait for all remaining jobs to finish
foreach ($p in $running_jobs) { $p.WaitForExit() }

Write-Host "All PPO hyperparameter search jobs completed!"
