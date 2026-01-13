#!/bin/bash

# ================= 配置区域 =================
# 1. 资源控制
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
MAX_JOBS=8                 # DQN 并发数 (考虑到你还有 PPO 在跑，设为 4 很稳)
DEVICES=("cuda:0" "cuda:1") # 双卡轮询

# 2. 环境设置
# 建议选 Breakout，因为 Pong 太简单了，参数差异看不出来
ENV_NAME="ALE/Breakout-v5"
TOTAL_STEPS=5000000         # Atari 通常需要 5M - 10M 步才能收敛
NUM_ENVS=16                  # DQN 建议设为 8 (比 16 省显存，且样本相关性更低)

# 3. 参数网格 (Grid Search)
# 我们采用 "Baseline + 变体" 的策略，而不是全排列，节省时间
# 格式: "LR UpdateFreq HiddenDim"
PARAMS=(
    # [Group 1] Baseline (最经典的设置)
    "1e-4 1000 512"
    
    # [Group 2] 学习率探究
    "5e-5 1000 512"  # 更稳
    "2e-4 1000 512"  # 更激进
    "5e-4 1000 512"
    
    # [Group 3] 目标网络更新频率探究 (针对多环境优化)
    "1e-4 2000 512"     # 更新更慢，通常更稳
    "1e-4 5000 512"     # 极慢更新
    
    # [Group 4] 网络容量探究
    "1e-4 1000 256"     # 轻量版
    "5e-5 1000 256"
)

SEEDS=(42) # 先跑一个种子快速看效果，有时间再加 101

# ================= 执行循环 =================

job_count=0

echo "=== Starting DQN Hyperparameter Search on $ENV_NAME ==="
echo "Total Configurations: ${#PARAMS[@]} * ${#SEEDS[@]}"

for param in "${PARAMS[@]}"; do
    read -r lr freq hd <<< "$param"
    
    for seed in "${SEEDS[@]}"; do
        
        # 显卡分配
        gpu_idx=$((job_count % ${#DEVICES[@]}))
        device=${DEVICES[$gpu_idx]}
        
        # 实验命名: DQN_Breakout_LR1e-4_UF1000_HD512
        exp_name="Search_LR${lr}_UF${freq}_HD${hd}_sd${seed}"
        
        echo "[Queue] Job $job_count | GPU: $device | LR: $lr | Freq: $freq | HD: $hd"
        
        # 启动 Python
        # 注意：这里我们针对 DQN 传入特定参数
        python run.py \
            --algo dqn \
            --env_name $ENV_NAME \
            --total_timesteps $TOTAL_STEPS \
            --num_envs $NUM_ENVS \
            --lr $lr \
            --update_freq $freq \
            --hidden_dim_dqn $hd \
            --seed $seed \
            --device $device \
            --exp_name $exp_name \
            --epsilon_decay 2000000 \
            --train_freq 4 \
            --batch_size 32 \
            > /dev/null 2>&1 &
            
        # 并发控制
        while true; do
            running=$(jobs -r | wc -l)
            if [ "$running" -lt "$MAX_JOBS" ]; then
                break
            else
                sleep 10
            fi
        done
        
        ((job_count++))
    done
done

echo "Waiting for final DQN jobs..."
wait
echo "DQN Search Completed!"