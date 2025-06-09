# export LD_LIBRARY_PATH="$HOME/miniconda3/envs/skywork-o1-prm/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH"
CUDA_VISIBLE_DEVICES=0 vllm serve "/home/v-wangyu1/model/Skywork-o1-Open-PRM-Qwen-2.5-7B" \
    --host 0.0.0.0 \
    --port 8081 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --enable-prefix-caching \
    --dtype auto

# tmux start-server
# if tmux has-session -t vLLM 2>/dev/null; then
#     tmux kill-session -t vLLM
# fi
# tmux new-session -s vLLM -n controller -d
# tmux send-keys "cd /scratch/amlt_code" Enter
# tmux new-window -n reward_model_worker
# tmux send-keys "cd /scratch/amlt_code" Enter
# tmux send-keys "export PATH=/home/aiscuser/.local/bin" Enter
# tmux send-keys "CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve /mnt/default/Inference-Scaling/Reward_Model/Skywork-o1-Open-PRM-Qwen-2.5-7B \
#     --host 0.0.0.0 \
#     --port 8081 \
#     --tensor-parallel-size 4 \
#     --gpu-memory-utilization 0.95 \
#     --enable-prefix-caching \
#     --dtype auto" Enter

# THRESHOLD_MEMORY=3000 # MB
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

# sleep 180