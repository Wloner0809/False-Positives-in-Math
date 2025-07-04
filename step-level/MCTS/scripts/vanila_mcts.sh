python evaluate.py \
    --LM Qwen2.5-Math-1.5B-Instruct \
    --prm_tokenizer_path /home/v-wangyu1/model/Skywork-o1-Open-PRM-Qwen-2.5-7B \
    --controller_addr http://0.0.0.0:28777 \
    --task_name MATH \
    --seed 42 \
    --method vanila_mcts \
    --num_sequence 4 \
    --temperature 0.7 \
    --top_k -1 \
    --top_p 1 \
    --max_new_tokens 2048 \
    --tree_max_depth 40 \
    --tree_max_width 4 \
    --save_dir /home/v-wangyu1/Inference-Scaling-Mirage/results/debug-qwen.jsonl \
    --num_worker 1 \