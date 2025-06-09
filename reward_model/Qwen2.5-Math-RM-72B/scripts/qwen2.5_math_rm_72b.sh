echo PYTHON_EXECUTABLE=$(which python3)
PYTHON_EXECUTABLE=$(which python3)

tmux start-server
if tmux has-session -t vLLM 2>/dev/null; then
    tmux kill-session -t vLLM
fi
tmux new-session -s vLLM -n controller -d
tmux send-keys "cd /scratch/amlt_code" Enter
tmux new-window -n reward_model_worker
tmux send-keys "cd /scratch/amlt_code" Enter
tmux send-keys "$PYTHON_EXECUTABLE -m vllm.entrypoints.openai.api_server \
    --model /mnt/default/deprecated/Inference-Scaling/Reward_Model/Qwen2.5-Math-RM-72B \
    --trust-remote-code \
    --served-model-name Qwen2.5-Math-RM-72B \
    --port 8081 \
    --tensor-parallel-size 8 \
    --enforce-eager" Enter

THRESHOLD_MEMORY=10000 # MB
SLEEP_INTERVAL=10 # seconds

while true; do
    GPU_MEMORY_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)

    ALL_GPUS_ABOVE_THRESHOLD=true

    while read -r MEMORY_USED; do
        if [ "$MEMORY_USED" -lt "$THRESHOLD_MEMORY" ]; then
            ALL_GPUS_ABOVE_THRESHOLD=false
            break
        fi
    done <<< "$GPU_MEMORY_USED"

    if $ALL_GPUS_ABOVE_THRESHOLD; then
        echo "GPU memory usage meets the requirement. Exiting."
        break
    else
        echo "GPU memory usage below $THRESHOLD_MEMORY MB. Sleeping for $SLEEP_INTERVAL seconds..."
        sleep $SLEEP_INTERVAL
    fi
done

sleep 180