# DQN Hyperparameter Search with limited parallel jobs

# === Parameter grids ===
$lrs = @(1e-4, 5e-4, 1e-3)           # Learning rates
$update_freqs = @(500, 1000, 2000)   # Target network update frequencies
$hidden_dims = @(256, 512)            # Hidden layer sizes
$dqn_types = @("dqn", "double", "dueling", "rainbow")
$devices = @("cuda:0", "cuda:1", "cuda:2", "cuda:3")

# Max concurrent jobs to limit CPU/GPU usage
$max_jobs = 4
$running_jobs = @()

$job_id = 0

$env = "ALE/Breakout-v5"

foreach ($dtype in $dqn_types) {
    foreach ($lr in $lrs) {
        foreach ($uf in $update_freqs) {
            foreach ($hd in $hidden_dims) {
                $device = $devices[$job_id % $devices.Length]
                $exp_name = "DQN_${dtype}_lr${lr}_uf${uf}_hd${hd}_sd42"

                $cmd_args = "--algo dqn --dqn_type $dtype --env_name $env --lr $lr --update_freq $uf --hidden_dim_dqn $hd --num_envs 16 --device $device --total_timesteps 2000000 --exp_name $exp_name"

                Write-Host "Starting Job $job_id on $device : DQN-$dtype | LR: $lr | UF: $uf | HD: $hd"
                $proc = Start-Process python -ArgumentList $cmd_args -PassThru -NoNewWindow
                $running_jobs += $proc
                $job_id++

                # Wait if too many concurrent jobs
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

Write-Host "All DQN hyperparameter search jobs completed!"
