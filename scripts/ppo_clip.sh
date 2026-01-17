#!/bin/bash

# ================= Configuration =================
# 1. 核心资源限制
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
MAX_JOBS=6            # 最大并发数
DEVICES=("cuda:0" "cuda:1") # 可用显卡

# 2. 基础参数
ENV_NAME="HalfCheetah-v4"
TOTAL_STEPS=2000000   # 2M 步
NUM_ENVS=16           

# 3. 固定参数 (基于之前的最优结果)
BEST_HD=256
BEST_LR_A="5e-5"
BEST_LR_C="2e-4"

# 4. 探究变量: Clip Range
# 0.1: Tighter constraint
# 0.5: Looser constraint
# 10.0: Effectively "No Clipping" (Unconstrained / Vanilla PG behavior)
CLIPS=(0.2 0.5 10.0)

SEEDS=(42 101) # 依然跑两个种子保证结论可靠

# ================= Execution Loop =================

job_count=0

echo "=== Starting Ablation Study: PPO Clipping ==="
echo "Target Env: $ENV_NAME"
echo "Fixed Params: HD=$BEST_HD, LRa=$BEST_LR_A, LRc=$BEST_LR_C"
echo "Testing Clips: ${CLIPS[@]}"

for clip in "${CLIPS[@]}"; do
    for seed in "${SEEDS[@]}"; do
            
        # 1. 显卡分配
        gpu_idx=$((job_count % ${#DEVICES[@]}))
        device=${DEVICES[$gpu_idx]}

        # 2. 生成实验名称 (注意这里把 Clip 放在最显眼的位置)
        # 格式: Ablation_Clip0.2_HD256_...
        exp_name="Ablation_Clip${clip}_HD${BEST_HD}_A${BEST_LR_A}_C${BEST_LR_C}_sd${seed}"

        # 3. 打印任务信息
        echo "[Queue] Job $job_count | GPU: $device | Clip: $clip | Seed: $seed"

        # 4. 启动任务
        python train.py \
            --algo ppo \
            --env_name $ENV_NAME \
            --total_timesteps $TOTAL_STEPS \
            --num_envs $NUM_ENVS \
            --ppo_clip $clip \
            --hidden_dim_ppo $BEST_HD \
            --lr_actor $BEST_LR_A \
            --lr_critic $BEST_LR_C \
            --seed $seed \
            --device $device \
            --exp_name $exp_name > /dev/null 2>&1 &

        # 5. 并发控制
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

echo "Waiting for ablation jobs to finish..."
wait
echo "Ablation study completed!"