python llama_get_score.py \
    --model_name_or_path /home/v-wangyu1/model/Skywork-o1-Open-PRM-Qwen-2.5-7B \
    --test_set_path /mnt/blob/Mirage/sample/Llama-vllm0.6.3.post1/skywork-format/llama3.2_3B_Instruct-aime_256answers_per_question_skywork_format.jsonl \
    --output_path /home/v-wangyu1/Inference-Scaling-Mirage/results/llama3.2_3B_Instruct-aime_256answers_per_question-with_scores.jsonl \
    --batch_size 3 \