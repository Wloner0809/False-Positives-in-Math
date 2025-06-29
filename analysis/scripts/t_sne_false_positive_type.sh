python analysis/t_sne_false_positive_type.py \
    --model_name_or_path /mnt/default/model/Qwen2.5-Math-72B-Instruct \
    --max_memory_per_gpu 70GiB \
    --offload_folder /mnt/default/Mirage/analysis/offload \
    --batch_size 1 \
    --torch_dtype bfloat16 \
    --data_path results/analysis/false_positive_type/false_positive_types.jsonl \
    --device cuda \
    --max_length 6000 \
    --target_layers -1 -2 -3 -4 -5 -6 \
    --pooling_strategy last_token \
    --save_dir /mnt/default/Mirage/analysis/Qwen72B-false_positive_type-last_token

python analysis/t_sne_false_positive_type.py \
    --model_name_or_path /mnt/default/model/Qwen2.5-Math-72B-Instruct \
    --max_memory_per_gpu 70GiB \
    --offload_folder /mnt/default/Mirage/analysis/offload \
    --batch_size 1 \
    --torch_dtype bfloat16 \
    --data_path results/analysis/false_positive_type/false_positive_types.jsonl \
    --device cuda \
    --max_length 6000 \
    --target_layers -1 -2 -3 -4 -5 -6 \
    --pooling_strategy mean \
    --save_dir /mnt/default/Mirage/analysis/Qwen72B-false_positive_type-mean

python analysis/t_sne_false_positive_type.py \
    --model_name_or_path /mnt/default/model/Llama-3.1-70B-Instruct \
    --max_memory_per_gpu 70GiB \
    --offload_folder /mnt/default/Mirage/analysis/offload \
    --batch_size 1 \
    --torch_dtype bfloat16 \
    --data_path results/analysis/false_positive_type/false_positive_types.jsonl \
    --device cuda \
    --max_length 6000 \
    --target_layers -1 -2 -3 -4 -5 -6 \
    --pooling_strategy last_token \
    --save_dir /mnt/default/Mirage/analysis/LLaMA70B-false_positive_type-last_token

python analysis/t_sne_false_positive_type.py \
    --model_name_or_path /mnt/default/model/Llama-3.1-70B-Instruct \
    --max_memory_per_gpu 70GiB \
    --offload_folder /mnt/default/Mirage/analysis/offload \
    --batch_size 1 \
    --torch_dtype bfloat16 \
    --data_path results/analysis/false_positive_type/false_positive_types.jsonl \
    --device cuda \
    --max_length 6000 \
    --target_layers -1 -2 -3 -4 -5 -6 \
    --pooling_strategy mean \
    --save_dir /mnt/default/Mirage/analysis/LLaMA70B-false_positive_type-mean

python analysis/t_sne_false_positive_type.py \
    --model_name_or_path /mnt/default/model/Qwen2.5-Math-7B-Instruct \
    --max_memory_per_gpu 70GiB \
    --offload_folder /mnt/default/Mirage/analysis/offload \
    --batch_size 1 \
    --torch_dtype bfloat16 \
    --data_path results/analysis/false_positive_type/false_positive_types.jsonl \
    --device cuda \
    --max_length 6000 \
    --target_layers -1 -2 -3 -4 -5 -6 \
    --pooling_strategy last_token \
    --save_dir /mnt/default/Mirage/analysis/Qwen7B-false_positive_type-last_token

python analysis/t_sne_false_positive_type.py \
    --model_name_or_path /mnt/default/model/Qwen2.5-Math-7B-Instruct \
    --max_memory_per_gpu 70GiB \
    --offload_folder /mnt/default/Mirage/analysis/offload \
    --batch_size 1 \
    --torch_dtype bfloat16 \
    --data_path results/analysis/false_positive_type/false_positive_types.jsonl \
    --device cuda \
    --max_length 6000 \
    --target_layers -1 -2 -3 -4 -5 -6 \
    --pooling_strategy mean \
    --save_dir /mnt/default/Mirage/analysis/Qwen7B-false_positive_type-mean

python analysis/t_sne_false_positive_type.py \
    --model_name_or_path /mnt/default/model/Llama-3.1-8B-Instruct \
    --max_memory_per_gpu 70GiB \
    --offload_folder /mnt/default/Mirage/analysis/offload \
    --batch_size 1 \
    --torch_dtype bfloat16 \
    --data_path results/analysis/false_positive_type/false_positive_types.jsonl \
    --device cuda \
    --max_length 6000 \
    --target_layers -1 -2 -3 -4 -5 -6 \
    --pooling_strategy last_token \
    --save_dir /mnt/default/Mirage/analysis/LLaMA8B-false_positive_type-last_token

python analysis/t_sne_false_positive_type.py \
    --model_name_or_path /mnt/default/model/Llama-3.1-8B-Instruct \
    --max_memory_per_gpu 70GiB \
    --offload_folder /mnt/default/Mirage/analysis/offload \
    --batch_size 1 \
    --torch_dtype bfloat16 \
    --data_path results/analysis/false_positive_type/false_positive_types.jsonl \
    --device cuda \
    --max_length 6000 \
    --target_layers -1 -2 -3 -4 -5 -6 \
    --pooling_strategy mean \
    --save_dir /mnt/default/Mirage/analysis/LLaMA8B-false_positive_type-mean

python analysis/t_sne_false_positive_type.py \
    --model_name_or_path /mnt/default/model/Qwen2.5-Math-1.5B-Instruct \
    --max_memory_per_gpu 70GiB \
    --offload_folder /mnt/default/Mirage/analysis/offload \
    --batch_size 1 \
    --torch_dtype bfloat16 \
    --data_path results/analysis/false_positive_type/false_positive_types.jsonl \
    --device cuda \
    --max_length 6000 \
    --target_layers -1 -2 -3 -4 -5 -6 \
    --pooling_strategy last_token \
    --save_dir /mnt/default/Mirage/analysis/Qwen1.5B-false_positive_type-last_token

python analysis/t_sne_false_positive_type.py \
    --model_name_or_path /mnt/default/model/Qwen2.5-Math-1.5B-Instruct \
    --max_memory_per_gpu 70GiB \
    --offload_folder /mnt/default/Mirage/analysis/offload \
    --batch_size 1 \
    --torch_dtype bfloat16 \
    --data_path results/analysis/false_positive_type/false_positive_types.jsonl \
    --device cuda \
    --max_length 6000 \
    --target_layers -1 -2 -3 -4 -5 -6 \
    --pooling_strategy mean \
    --save_dir /mnt/default/Mirage/analysis/Qwen1.5B-false_positive_type-mean

python analysis/t_sne_false_positive_type.py \
    --model_name_or_path /mnt/default/model/Llama-3.2-3B-Instruct \
    --max_memory_per_gpu 70GiB \
    --offload_folder /mnt/default/Mirage/analysis/offload \
    --batch_size 1 \
    --torch_dtype bfloat16 \
    --data_path results/analysis/false_positive_type/false_positive_types.jsonl \
    --device cuda \
    --max_length 6000 \
    --target_layers -1 -2 -3 -4 -5 -6 \
    --pooling_strategy last_token \
    --save_dir /mnt/default/Mirage/analysis/LLaMA3B-false_positive_type-last_token

python analysis/t_sne_false_positive_type.py \
    --model_name_or_path /mnt/default/model/Llama-3.2-3B-Instruct \
    --max_memory_per_gpu 70GiB \
    --offload_folder /mnt/default/Mirage/analysis/offload \
    --batch_size 1 \
    --torch_dtype bfloat16 \
    --data_path results/analysis/false_positive_type/false_positive_types.jsonl \
    --device cuda \
    --max_length 6000 \
    --target_layers -1 -2 -3 -4 -5 -6 \
    --pooling_strategy mean \
    --save_dir /mnt/default/Mirage/analysis/LLaMA3B-false_positive_type-mean