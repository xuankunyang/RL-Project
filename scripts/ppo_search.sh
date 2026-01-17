#!/bin/bash

# ================= Configuration =================
# 1. 核心资源限制
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
MAX_JOBS=8            # 最大并发数
DEVICES=("cuda:0" "cuda:1" "cuda:2" "cuda:3") # 可用显卡

# 2. 基础参数
ENV_NAME="Hopper-v4"
TOTAL_STEPS=2000000   # 2M 步足够看出好坏
NUM_ENVS=16            # 16 并行环境 (速度快)
CLIP=0.1              # 锁定 Clip 为 0.2 (Baseline不建议乱动这个)

# 3. 搜索空间 (Grid Search)
# 格式: "Actor_LR Critic_LR"
LR_PAIRS=(
    "3e-5 1e-4"
    "5e-5 2e-4"
    "1e-4 3e-4"
    "5e-5 1e-4"
)

HIDDEN_DIMS=(256 512)
SEEDS=(42 101) # 跑两个种子看稳定性

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
            # 格式: Search_HD256_A3e-4_C1e-3_sd42
            exp_name="Search_HD${hd}_A${lr_a}_C${lr_c}_sd${seed}"

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

            # 记录后台进程PID (可选，这里我们用 jobs 命令控制)
            
            # 5. 并发控制逻辑
            while true; do
                # 统计当前后台运行的 Python 任务数量
                running=$(jobs -r | wc -l)
                
                if [ "$running" -lt "$MAX_JOBS" ]; then
                    # 如果运行数 < MAX_JOBS，跳出等待，继续投递下一个任务
                    break
                else
                    # 否则等待 5 秒再检查
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