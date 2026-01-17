#!/bin/bash

# ================= Configuration =================
# Parallelization Settings
MAX_JOBS=8
# Adjust available devices here
DEVICES=("cuda:0" "cuda:1" "cuda:2" "cuda:3") 
SEEDS=(42 101)

# Common Params
STEPS_HOPPER=2000000
STEPS_ANT=5000000
NUM_ENVS=16
EVAL_FREQ=50000

# ================= Parameter Sets (FILL ME) =================
# Please replace "FILL_ME" with your specific values.

# --- Hopper Set 1 ---
H1_LR_A="FILL_ME"   # e.g. 3e-4
H1_LR_C="FILL_ME"   # e.g. 1e-3
H1_CLIP="FILL_ME"   # e.g. 0.2
H1_HIDDEN="FILL_ME" # e.g. 256

# --- Hopper Set 2 ---
H2_LR_A="FILL_ME"
H2_LR_C="FILL_ME"
H2_CLIP="FILL_ME"
H2_HIDDEN="FILL_ME"

# --- Hopper Set 3 ---
H3_LR_A="FILL_ME"
H3_LR_C="FILL_ME"
H3_CLIP="FILL_ME"
H3_HIDDEN="FILL_ME"

# --- Ant Set 1 ---
A1_LR_A="FILL_ME"   # Recommended conservative: 5e-5
A1_LR_C="FILL_ME"   # Recommended conservative: 2e-4
A1_CLIP="FILL_ME"
A1_HIDDEN="FILL_ME"

# ================= Helper Function =================
job_count=0

run_job() {
    env_name=$1
    desc=$2
    lr_a=$3
    lr_c=$4
    clip=$5
    hidden=$6
    seed=$7
    steps=$8
    
    # Check if params are filled
    if [[ "$lr_a" == "FILL_ME" ]]; then
        echo "Skipping $desc (Seed $seed): Parameters not filled."
        return
    fi

    # GPU Allocation
    gpu_idx=$((job_count % ${#DEVICES[@]}))
    device=${DEVICES[$gpu_idx]}

    exp_name="${desc}_sd${seed}"
    
    echo "[Queue] Job $job_count | GPU: $device | Exp: $exp_name"

    # Ensure log directory exists
    mkdir -p logs

    # Run in background
    # Assuming train.py is in the current directory (project root)
    nohup python train.py \
        --algo ppo \
        --env_name $env_name \
        --total_timesteps $steps \
        --num_envs $NUM_ENVS \
        --eval_freq $EVAL_FREQ \
        --device $device \
        --seed $seed \
        --exp_name $exp_name \
        --lr_actor $lr_a \
        --lr_critic $lr_c \
        --ppo_clip $clip \
        --hidden_dim_ppo $hidden \
        > "logs/${exp_name}.log" 2>&1 &
        
    # Concurrency Control
    while true; do
        running=$(jobs -r | wc -l)
        if [ "$running" -lt "$MAX_JOBS" ]; then
            break
        else
            sleep 2
        fi
    done

    ((job_count++))
}

# ================= Main Loop =================
echo "=== Starting Batch Experiments ==="
echo "Note: Logs will be saved to 'logs/' directory."

# Hopper Experiments
for seed in "${SEEDS[@]}"; do
    run_job "Hopper-v4" "Hopper_Set1" "$H1_LR_A" "$H1_LR_C" "$H1_CLIP" "$H1_HIDDEN" "$seed" "$STEPS_HOPPER"
    run_job "Hopper-v4" "Hopper_Set2" "$H2_LR_A" "$H2_LR_C" "$H2_CLIP" "$H2_HIDDEN" "$seed" "$STEPS_HOPPER"
    run_job "Hopper-v4" "Hopper_Set3" "$H3_LR_A" "$H3_LR_C" "$H3_CLIP" "$H3_HIDDEN" "$seed" "$STEPS_HOPPER"
done

# Ant Experiments
for seed in "${SEEDS[@]}"; do
    run_job "Ant-v4" "Ant_Set1" "$A1_LR_A" "$A1_LR_C" "$A1_CLIP" "$A1_HIDDEN" "$seed" "$STEPS_ANT"
done

echo "All jobs submitted. Waiting for completion..."
wait
echo "All experiments finished."