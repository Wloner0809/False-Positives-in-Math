set -e

HOST_ADDR=0.0.0.0
CONTROLER_PORT=28777
WORKER_BASE_PORT=30010

echo PYTHON_EXECUTABLE=$(which python3)
PYTHON_EXECUTABLE=$(which python3)

MODEL_BASE=/home/v-wangyu1/model
CUDA_DEVICE_BASE=0
POLICY_MODEL_NAME=Qwen2.5-Math-1.5B-Instruct
MODEL_PATH=$MODEL_BASE/$POLICY_MODEL_NAME

LOGDIR=logs_fastchat

tmux start-server
if tmux has-session -t FastChat 2>/dev/null; then
    tmux kill-session -t FastChat
fi
tmux new-session -s FastChat -n controller -d
tmux send-keys "export LOGDIR=${LOGDIR}" Enter
tmux send-keys "$PYTHON_EXECUTABLE -m fastchat.serve.controller --port ${CONTROLER_PORT} --host $HOST_ADDR" Enter

NUM_LM_WORKER=1

echo "Wait 5 seconds ..."
sleep 5

echo "Starting workers"
for i in $(seq 0 $((NUM_LM_WORKER-1)))
do
    WORKER_PORT=$((WORKER_BASE_PORT+i))
    tmux new-window -n policy_worker_$i
    tmux send-keys "export LOGDIR=${LOGDIR}" Enter
    tmux send-keys "CUDA_VISIBLE_DEVICES=$((i+CUDA_DEVICE_BASE)) $PYTHON_EXECUTABLE -m reason.llm_service.workers.vllm_worker --model-path $MODEL_PATH --controller-address http://$HOST_ADDR:$CONTROLER_PORT --host $HOST_ADDR --port $WORKER_PORT --worker-address http://$HOST_ADDR:$WORKER_PORT --gpu_memory_utilization 0.6" Enter
done
