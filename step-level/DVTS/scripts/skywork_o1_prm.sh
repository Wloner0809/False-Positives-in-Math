#! local
CUDA_VISIBLE_DEVICES=0 vllm serve "/home/v-wangyu1/model/Skywork-o1-Open-PRM-Qwen-2.5-7B" \
    --host 0.0.0.0 \
    --port 8081 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --enable-prefix-caching \
    --dtype auto

#! remote
# tmux start-server
# if tmux has-session -t PRM 2>/dev/null; then
#     tmux kill-session -t PRM
# fi
# tmux new-session -s PRM -n controller -d
# tmux send-keys "cd /scratch/amlt_code/skywork_prm-guide-search" Enter
# tmux new-window -n reward_model_worker
# tmux send-keys "cd /scratch/amlt_code/skywork_prm-guide-search" Enter
# tmux send-keys "CUDA_VISIBLE_DEVICES=3 vllm serve /mnt/default/Inference-Scaling/Reward_Model/Skywork-o1-Open-PRM-Qwen-2.5-7B \
#     --host 0.0.0.0 \
#     --port 8081 \
#     --tensor-parallel-size 1 \
#     --gpu-memory-utilization 0.95 \
#     --enable-prefix-caching \
#     --dtype auto" Enter

# THRESHOLD_MEMORY=5000 # MB
# SLEEP_INTERVAL=10 # seconds

# while true; do
#     GPU_MEMORY_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)

#     ALL_GPUS_ABOVE_THRESHOLD=true

#     while read -r MEMORY_USED; do
#         if [ "$MEMORY_USED" -lt "$THRESHOLD_MEMORY" ]; then
#             ALL_GPUS_ABOVE_THRESHOLD=false
#             break
#         fi
#     done <<< "$GPU_MEMORY_USED"

#     if $ALL_GPUS_ABOVE_THRESHOLD; then
#         echo "GPU memory usage meets the requirement. Exiting."
#         break
#     else
#         echo "GPU memory usage below $THRESHOLD_MEMORY MB. Sleeping for $SLEEP_INTERVAL seconds..."
#         sleep $SLEEP_INTERVAL
#     fi
# done

# sleep 120