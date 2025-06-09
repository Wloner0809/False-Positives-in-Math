python run/test_time_compute.py recipes/Qwen2.5-Math-1.5B-Instruct/dvts.yaml \
    --model_path=/home/v-wangyu1/model/Qwen2.5-Math-1.5B-Instruct \
    --policy_model_name=Qwen2.5-Math-1.5B-Instruct \
    --reward_model_path=/home/v-wangyu1/model/Skywork-o1-Open-PRM-Qwen-2.5-7B \
    --controller_addr=http://0.0.0.0:28777 \
    --num_workers=1 \
    --output_file=/home/v-wangyu1/Inference-Scaling-Mirage/results/debug-qwen.jsonl \
    --dataset_name=qq8933/MATH500 \
    --dataset_split=test \