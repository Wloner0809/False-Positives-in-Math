python analysis/t_sne.py \
    --model_name_or_path /mnt/default/model/Llama-3.1-70B-Instruct \
    --max_memory_per_gpu 70GiB \
    --offload_folder /mnt/default/Mirage/analysis/offload \
    --batch_size 1 \
    --torch_dtype bfloat16 \
    --data_path results/analysis/correct_false_positive_incorrect_llama.jsonl \
    --device cuda \
    --max_length 6000 \
    --target_layers -1 -2 -3 -4 -5 -6 \
    --pooling_strategy last_token \
    --save_dir /mnt/default/Mirage/analysis/LLaMA70B-last_token

python analysis/t_sne.py \
    --model_name_or_path /mnt/default/model/Llama-3.1-70B-Instruct \
    --max_memory_per_gpu 70GiB \
    --offload_folder /mnt/default/Mirage/analysis/offload \
    --batch_size 1 \
    --torch_dtype bfloat16 \
    --data_path results/analysis/correct_false_positive_incorrect_llama.jsonl \
    --device cuda \
    --max_length 6000 \
    --target_layers -1 -2 -3 -4 -5 -6 \
    --pooling_strategy mean \
    --save_dir /mnt/default/Mirage/analysis/LLaMA70B-mean

python analysis/t_sne.py \
    --model_name_or_path /mnt/default/model/Qwen2.5-Math-72B-Instruct \
    --max_memory_per_gpu 70GiB \
    --offload_folder /mnt/default/Mirage/analysis/offload \
    --batch_size 1 \
    --torch_dtype bfloat16 \
    --data_path results/analysis/correct_false_positive_incorrect_qwen.jsonl \
    --device cuda \
    --max_length 6000 \
    --target_layers -1 -2 -3 -4 -5 -6 \
    --pooling_strategy last_token \
    --save_dir /mnt/default/Mirage/analysis/Qwen72B-last_token

python analysis/t_sne.py \
    --model_name_or_path /mnt/default/model/Qwen2.5-Math-72B-Instruct \
    --max_memory_per_gpu 70GiB \
    --offload_folder /mnt/default/Mirage/analysis/offload \
    --batch_size 1 \
    --torch_dtype bfloat16 \
    --data_path results/analysis/correct_false_positive_incorrect_qwen.jsonl \
    --device cuda \
    --max_length 6000 \
    --target_layers -1 -2 -3 -4 -5 -6 \
    --pooling_strategy mean \
    --save_dir /mnt/default/Mirage/analysis/Qwen72B-mean

python analysis/t_sne.py \
    --model_name_or_path /mnt/default/model/Llama-3.1-8B-Instruct \
    --max_memory_per_gpu 70GiB \
    --offload_folder /mnt/default/Mirage/analysis/offload \
    --batch_size 1 \
    --torch_dtype bfloat16 \
    --data_path results/analysis/correct_false_positive_incorrect_llama.jsonl \
    --device cuda \
    --max_length 6000 \
    --target_layers -1 -2 -3 -4 -5 -6 \
    --pooling_strategy last_token \
    --save_dir /mnt/default/Mirage/analysis/LLaMA8B-last_token

python analysis/t_sne.py \
    --model_name_or_path /mnt/default/model/Llama-3.1-8B-Instruct \
    --max_memory_per_gpu 70GiB \
    --offload_folder /mnt/default/Mirage/analysis/offload \
    --batch_size 1 \
    --torch_dtype bfloat16 \
    --data_path results/analysis/correct_false_positive_incorrect_llama.jsonl \
    --device cuda \
    --max_length 6000 \
    --target_layers -1 -2 -3 -4 -5 -6 \
    --pooling_strategy mean \
    --save_dir /mnt/default/Mirage/analysis/LLaMA8B-mean

python analysis/t_sne.py \
    --model_name_or_path /mnt/default/model/Qwen2.5-Math-7B-Instruct \
    --max_memory_per_gpu 70GiB \
    --offload_folder /mnt/default/Mirage/analysis/offload \
    --batch_size 1 \
    --torch_dtype bfloat16 \
    --data_path results/analysis/correct_false_positive_incorrect_qwen.jsonl \
    --device cuda \
    --max_length 6000 \
    --target_layers -1 -2 -3 -4 -5 -6 \
    --pooling_strategy last_token \
    --save_dir /mnt/default/Mirage/analysis/Qwen7B-last_token

python analysis/t_sne.py \
    --model_name_or_path /mnt/default/model/Qwen2.5-Math-7B-Instruct \
    --max_memory_per_gpu 70GiB \
    --offload_folder /mnt/default/Mirage/analysis/offload \
    --batch_size 1 \
    --torch_dtype bfloat16 \
    --data_path results/analysis/correct_false_positive_incorrect_qwen.jsonl \
    --device cuda \
    --max_length 6000 \
    --target_layers -1 -2 -3 -4 -5 -6 \
    --pooling_strategy mean \
    --save_dir /mnt/default/Mirage/analysis/Qwen7B-mean

python analysis/t_sne.py \
    --model_name_or_path /mnt/default/model/Llama-3.2-3B-Instruct \
    --max_memory_per_gpu 70GiB \
    --offload_folder /mnt/default/Mirage/analysis/offload \
    --batch_size 1 \
    --torch_dtype bfloat16 \
    --data_path results/analysis/correct_false_positive_incorrect_llama.jsonl \
    --device cuda \
    --max_length 6000 \
    --target_layers -1 -2 -3 -4 -5 -6 \
    --pooling_strategy last_token \
    --save_dir /mnt/default/Mirage/analysis/LLaMA3B-last_token

python analysis/t_sne.py \
    --model_name_or_path /mnt/default/model/Llama-3.2-3B-Instruct \
    --max_memory_per_gpu 70GiB \
    --offload_folder /mnt/default/Mirage/analysis/offload \
    --batch_size 1 \
    --torch_dtype bfloat16 \
    --data_path results/analysis/correct_false_positive_incorrect_llama.jsonl \
    --device cuda \
    --max_length 6000 \
    --target_layers -1 -2 -3 -4 -5 -6 \
    --pooling_strategy mean \
    --save_dir /mnt/default/Mirage/analysis/LLaMA3B-mean

python analysis/t_sne.py \
    --model_name_or_path /mnt/default/model/Qwen2.5-Math-1.5B-Instruct \
    --max_memory_per_gpu 70GiB \
    --offload_folder /mnt/default/Mirage/analysis/offload \
    --batch_size 1 \
    --torch_dtype bfloat16 \
    --data_path results/analysis/correct_false_positive_incorrect_qwen.jsonl \
    --device cuda \
    --max_length 6000 \
    --target_layers -1 -2 -3 -4 -5 -6 \
    --pooling_strategy last_token \
    --save_dir /mnt/default/Mirage/analysis/Qwen1.5B-last_token

python analysis/t_sne.py \
    --model_name_or_path /mnt/default/model/Qwen2.5-Math-1.5B-Instruct \
    --max_memory_per_gpu 70GiB \
    --offload_folder /mnt/default/Mirage/analysis/offload \
    --batch_size 1 \
    --torch_dtype bfloat16 \
    --data_path results/analysis/correct_false_positive_incorrect_qwen.jsonl \
    --device cuda \
    --max_length 6000 \
    --target_layers -1 -2 -3 -4 -5 -6 \
    --pooling_strategy mean \
    --save_dir /mnt/default/Mirage/analysis/Qwen1.5B-mean