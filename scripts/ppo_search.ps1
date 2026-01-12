# === PPO Best Baseline Search (CPU/GPU Optimized) ===

# 1. 核心优化：限制 PyTorch 单个进程只用 1 个 CPU 核
# 因为我们开了 16 个环境并行，如果不限制，CPU 会瞬间 100% 导致卡顿
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"

# 2. 基础环境设置
$env_name = "HalfCheetah-v4"
# 使用 16 并行环境 + 200万步 (足够收敛)
$common_args = "--algo ppo --env_name $env_name --total_timesteps 2000000 --num_envs 16 --ppo_clip 0.2"

# 3. 定义参数搜索网格 (只搜索最有希望的组合)
# 格式: @{ label="... " ; hd=... ; lra=... ; lrc=... }
$configs = @(
    # [Config 1] 经典标准配置 (Standard Baseline)
    @{ label="Std_256";   hd=256; lra=3e-4; lrc=3e-4 },

    # [Config 2] 推荐配置 (Fast Critic) - 预期效果最好
    @{ label="Dual_256";  hd=256; lra=3e-4; lrc=1e-3 },

    # [Config 3] 大网络标准配置 (Big Net Standard)
    @{ label="Std_512";   hd=512; lra=3e-4; lrc=3e-4 },

    # [Config 4] 大网络 + 推荐配置 (Big Net Dual)
    @{ label="Dual_512";  hd=512; lra=3e-4; lrc=1e-3 }
)

# 4. 随机种子 (跑两个种子看稳定性)
$seeds = @(42, 101)

# 5. 硬件分配
$devices = @("cuda:0", "cuda:1", "cuda:2", "cuda:3")
$max_jobs = 4   # 严格限制并发为 4

$running_jobs = @()
$job_id = 0

Write-Host "=== Starting PPO Baseline Search on $env_name ===" -ForegroundColor Green
Write-Host "Configs: $($configs.Count) | Seeds: $($seeds.Count) | Total Jobs: $($configs.Count * $seeds.Count)"

foreach ($conf in $configs) {
    foreach ($seed in $seeds) {
        
        # 轮询分配 GPU
        $device = $devices[$job_id % $devices.Length]
        
        # 构造实验名称: PPO_HalfCheetah_Dual_256_sd42
        $exp_name = "Search_${conf.label}_sd${seed}"
        
        # 构造参数字符串
        $args_list = "$common_args --hidden_dim_ppo $($conf.hd) --lr_actor $($conf.lra) --lr_critic $($conf.lrc) --seed $seed --device $device --exp_name $exp_name"
        
        Write-Host "[Queue] Job $job_id | Dev: $device | Config: $($conf.label) (HD=$($conf.hd), LRa=$($conf.lra), LRc=$($conf.lrc))" -ForegroundColor Cyan
        
        # 启动后台进程
        $proc = Start-Process python -ArgumentList $args_list -PassThru -NoNewWindow
        $running_jobs += $proc
        $job_id++

        # 并发控制：保持运行的任务不超过 max_jobs
        while ($running_jobs.Count -ge $max_jobs) {
            # 过滤掉已经退出的进程
            $running_jobs = $running_jobs | Where-Object { -not $_.HasExited }
            
            # 如果还满，稍微等一下
            if ($running_jobs.Count -ge $max_jobs) {
                Start-Sleep -Seconds 5
            }
        }
    }
}

# 等待剩余任务结束
Write-Host "Waiting for final batch to finish..." -ForegroundColor Yellow
foreach ($p in $running_jobs) { $p.WaitForExit() }

Write-Host "Search Completed! Check Tensorboard for results." -ForegroundColor Green