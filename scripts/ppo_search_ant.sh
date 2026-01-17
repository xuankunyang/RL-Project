#!/bin/bash

# ================= Configuration =================
# 1. 核心资源限制
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
MAX_JOBS=8            # Ant 比较吃 CPU，建议降低并发数 (如果 CPU 核数不够多)
DEVICES=("cuda:0" "cuda:1" "cuda:2" "cuda:3") # 可用显卡

# 2. 基础参数
ENV_NAME="Ant-v4"
TOTAL_STEPS=5000000   # Ant 比较难，建议跑 5M 步 (2M 可能还没收敛)
NUM_ENVS=16            
CLIP=0.2              # Standard PPO clip

# 3. 搜索空间 (Grid Search)
# Ant 这种高维连续控制，LR 3e-4 是非常稳健的 Baseline
# 格式: "Actor_LR Critic_LR"
LR_PAIRS=(
    "3e-4 3e-4"       # Baseline (Standard PPO)
    "3e-4 1e-3"       # Strong Critic
    "1e-4 3e-4"       # Conservative Actor
    "5e-5 2e-4"       # Extra Conservative (New Recommendation)
)

HIDDEN_DIMS=(256 512) # Ant 需要大一点的网络
SEEDS=(42 101) 

# ================= Execution Loop =================

job_count=0

echo "=== Starting Grid Search on $ENV_NAME ==="
echo "Total Combinations: ${#LR_PAIRS[@]} LRs * ${#HIDDEN_DIMS[@]} Dims * ${#SEEDS[@]} Seeds"

for lr_pair in "${LR_PAIRS[@]}"; do
    # 解析 Actor 和 Critic LR
    read -r lr_a lr_c <<< "$lr_pair"

    for hd in "${HIDDEN_DIMS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            
            # 1. 简单的显卡轮询分配
            gpu_idx=$((job_count % ${#DEVICES[@]}))
            device=${DEVICES[$gpu_idx]}

            # 2. 生成实验名称
            exp_name="Search_Ant_HD${hd}_A${lr_a}_C${lr_c}_sd${seed}"

            # 3. 打印任务信息
            echo "[Queue] Job $job_count | GPU: $device | HD: $hd | LRa: $lr_a | LRc: $lr_c | Seed: $seed"

            # 4. 后台启动 Python 任务
            python train.py \
                --algo ppo \
                --env_name $ENV_NAME \
                --total_timesteps $TOTAL_STEPS \
                --num_envs $NUM_ENVS \
                --ppo_clip $CLIP \
                --hidden_dim_ppo $hd \
                --lr_actor $lr_a \
                --lr_critic $lr_c \
                --seed $seed \
                --device $device \
                --exp_name $exp_name > /dev/null 2>&1 &

            # 5. 并发控制逻辑
            while true; do
                running=$(jobs -r | wc -l)
                if [ "$running" -lt "$MAX_JOBS" ]; then
                    break
                else
                    sleep 5
                fi
            done

            ((job_count++))
        done
    done
done

# 等待所有剩余任务完成
echo "Waiting for final batch to finish..."
wait
echo "All grid search jobs completed!"