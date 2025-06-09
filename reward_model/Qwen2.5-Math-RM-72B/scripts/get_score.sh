python reward_model/Qwen2.5-Math-RM-72B/distill_r1_get_score.py \
    --model_name_or_path /mnt/default/deprecated/Inference-Scaling/Reward_Model/Qwen2.5-Math-RM-72B \
    --test_set_path /mnt/default/Mirage/sample/Distill_R1-vllm0.7.2/DeepSeek_R1_Distill_Qwen_32B-aime_64answers_per_question.jsonl \
    --output_path /mnt/default/Mirage/sample/Distill_R1-vllm0.7.2-with-scores/DeepSeek_R1_Distill_Qwen_32B-aime_64answers_per_question-with_scores.jsonl \
    --batch_size -1 \



#! llama data reward
#* llama3.2_3B_Instruct
# python reward_model/Qwen2.5-Math-RM-72B/llama_get_score.py \
#     --model_name_or_path /mnt/default/Inference-Scaling/Reward_Model/Qwen2.5-Math-RM-72B \
#     --test_set_path /mnt/default/Mirage/sample/Llama-vllm0.6.3.post1/llama3.2_3B_Instruct-aime_256answers_per_question.jsonl \
#     --output_path /mnt/default/Mirage/sample/Llama-with-scores/Qwen2.5-MATH-RM-72B/llama3.2_3B_Instruct-aime_256answers_per_question-with_scores.jsonl \
#     --batch_size -1 \

# python reward_model/Qwen2.5-Math-RM-72B/llama_get_score.py \
#     --model_name_or_path /mnt/default/Inference-Scaling/Reward_Model/Qwen2.5-Math-RM-72B \
#     --test_set_path /mnt/default/Mirage/sample/Llama-vllm0.6.3.post1/llama3.2_3B_Instruct-math500_256answers_per_question.jsonl \
#     --output_path /mnt/default/Mirage/sample/Llama-with-scores/Qwen2.5-MATH-RM-72B/llama3.2_3B_Instruct-math500_256answers_per_question-with_scores.jsonl \
#     --batch_size -1 \

# python reward_model/Qwen2.5-Math-RM-72B/llama_get_score.py \
#     --model_name_or_path /mnt/default/Inference-Scaling/Reward_Model/Qwen2.5-Math-RM-72B \
#     --test_set_path /mnt/default/Mirage/sample/Llama-vllm0.6.3.post1/llama3.2_3B_Instruct-OmniMATH500_256answers_per_question.jsonl \
#     --output_path /mnt/default/Mirage/sample/Llama-with-scores/Qwen2.5-MATH-RM-72B/llama3.2_3B_Instruct-OmniMATH500_256answers_per_question-with_scores.jsonl \
#     --batch_size -1 \

#* llama3.1_8B_Instruct
# python reward_model/Qwen2.5-Math-RM-72B/llama_get_score.py \
#     --model_name_or_path /mnt/default/Inference-Scaling/Reward_Model/Qwen2.5-Math-RM-72B \
#     --test_set_path /mnt/default/Mirage/sample/Llama-vllm0.6.3.post1/llama3.1_8B_Instruct-aime_256answers_per_question.jsonl \
#     --output_path /mnt/default/Mirage/sample/Llama-with-scores/Qwen2.5-MATH-RM-72B/llama3.1_8B_Instruct-aime_256answers_per_question-with_scores.jsonl \
#     --batch_size -1 \

# python reward_model/Qwen2.5-Math-RM-72B/llama_get_score.py \
#     --model_name_or_path /mnt/default/Inference-Scaling/Reward_Model/Qwen2.5-Math-RM-72B \
#     --test_set_path /mnt/default/Mirage/sample/Llama-vllm0.6.3.post1/llama3.1_8B_Instruct-math500_256answers_per_question.jsonl \
#     --output_path /mnt/default/Mirage/sample/Llama-with-scores/Qwen2.5-MATH-RM-72B/llama3.1_8B_Instruct-math500_256answers_per_question-with_scores.jsonl \
#     --batch_size -1 \

# python reward_model/Qwen2.5-Math-RM-72B/llama_get_score.py \
#     --model_name_or_path /mnt/default/Inference-Scaling/Reward_Model/Qwen2.5-Math-RM-72B \
#     --test_set_path /mnt/default/Mirage/sample/Llama-vllm0.6.3.post1/llama3.1_8B_Instruct-OmniMATH500_256answers_per_question.jsonl \
#     --output_path /mnt/default/Mirage/sample/Llama-with-scores/Qwen2.5-MATH-RM-72B/llama3.1_8B_Instruct-OmniMATH500_256answers_per_question-with_scores.jsonl \
#     --batch_size -1 \

#* llama3.1_70B_Instruct
# python reward_model/Qwen2.5-Math-RM-72B/llama_get_score.py \
#     --model_name_or_path /mnt/default/Inference-Scaling/Reward_Model/Qwen2.5-Math-RM-72B \
#     --test_set_path /mnt/default/Mirage/sample/Llama-vllm0.6.3.post1/llama3.1_70B_Instruct-aime_256answers_per_question.jsonl \
#     --output_path /mnt/default/Mirage/sample/Llama-with-scores/Qwen2.5-MATH-RM-72B/llama3.1_70B_Instruct-aime_256answers_per_question-with_scores.jsonl \
#     --batch_size -1 \

# python reward_model/Qwen2.5-Math-RM-72B/llama_get_score.py \
#     --model_name_or_path /mnt/default/Inference-Scaling/Reward_Model/Qwen2.5-Math-RM-72B \
#     --test_set_path /mnt/default/Mirage/sample/Llama-vllm0.6.3.post1/llama3.1_70B_Instruct-math500_256answers_per_question.jsonl \
#     --output_path /mnt/default/Mirage/sample/Llama-with-scores/Qwen2.5-MATH-RM-72B/llama3.1_70B_Instruct-math500_256answers_per_question-with_scores.jsonl \
#     --batch_size -1 \

# python reward_model/Qwen2.5-Math-RM-72B/llama_get_score.py \
#     --model_name_or_path /mnt/default/Inference-Scaling/Reward_Model/Qwen2.5-Math-RM-72B \
#     --test_set_path /mnt/default/Mirage/sample/Llama-vllm0.6.3.post1/llama3.1_70B_Instruct-OmniMATH500_256answers_per_question.jsonl \
#     --output_path /mnt/default/Mirage/sample/Llama-with-scores/Qwen2.5-MATH-RM-72B/llama3.1_70B_Instruct-OmniMATH500_256answers_per_question-with_scores.jsonl \
#     --batch_size -1 \


#! qwen data reward
#* Qwen2.5_Math_1.5B_Instruct
# python reward_model/Qwen2.5-Math-RM-72B/qwen_get_score.py \
#     --model_name_or_path /mnt/default/Inference-Scaling/Reward_Model/Qwen2.5-Math-RM-72B \
#     --test_set_path /mnt/default/Mirage/sample/Qwen-vllm0.6.3.post1/Qwen2.5_Math_1.5B_Instruct-aime_256answers_per_question.jsonl \
#     --output_path /mnt/default/Mirage/sample/Qwen-with-scores/Qwen2.5-MATH-RM-72B/Qwen2.5_Math_1.5B_Instruct-aime_256answers_per_question-with_scores.jsonl \
#     --batch_size -1 \

# python reward_model/Qwen2.5-Math-RM-72B/qwen_get_score.py \
#     --model_name_or_path /mnt/default/Inference-Scaling/Reward_Model/Qwen2.5-Math-RM-72B \
#     --test_set_path /mnt/default/Mirage/sample/Qwen-vllm0.6.3.post1/Qwen2.5_Math_1.5B_Instruct-math500_256answers_per_question.jsonl \
#     --output_path /mnt/default/Mirage/sample/Qwen-with-scores/Qwen2.5-MATH-RM-72B/Qwen2.5_Math_1.5B_Instruct-math500_256answers_per_question-with_scores.jsonl \
#     --batch_size -1 \

# python reward_model/Qwen2.5-Math-RM-72B/qwen_get_score.py \
#     --model_name_or_path /mnt/default/Inference-Scaling/Reward_Model/Qwen2.5-Math-RM-72B \
#     --test_set_path /mnt/default/Mirage/sample/Qwen-vllm0.6.3.post1/Qwen2.5_Math_1.5B_Instruct-OmniMATH500_256answers_per_question.jsonl \
#     --output_path /mnt/default/Mirage/sample/Qwen-with-scores/Qwen2.5-MATH-RM-72B/Qwen2.5_Math_1.5B_Instruct-OmniMATH500_256answers_per_question-with_scores.jsonl \
#     --batch_size -1 \

#* Qwen2.5_Math_7B_Instruct
# python reward_model/Qwen2.5-Math-RM-72B/qwen_get_score.py \
#     --model_name_or_path /mnt/default/Inference-Scaling/Reward_Model/Qwen2.5-Math-RM-72B \
#     --test_set_path /mnt/default/Mirage/sample/Qwen-vllm0.6.3.post1/Qwen2.5_Math_7B_Instruct-aime_256answers_per_question.jsonl \
#     --output_path /mnt/default/Mirage/sample/Qwen-with-scores/Qwen2.5-MATH-RM-72B/Qwen2.5_Math_7B_Instruct-aime_256answers_per_question-with_scores.jsonl \
#     --batch_size -1 \

# python reward_model/Qwen2.5-Math-RM-72B/qwen_get_score.py \
#     --model_name_or_path /mnt/default/Inference-Scaling/Reward_Model/Qwen2.5-Math-RM-72B \
#     --test_set_path /mnt/default/Mirage/sample/Qwen-vllm0.6.3.post1/Qwen2.5_Math_7B_Instruct-math500_256answers_per_question.jsonl \
#     --output_path /mnt/default/Mirage/sample/Qwen-with-scores/Qwen2.5-MATH-RM-72B/Qwen2.5_Math_7B_Instruct-math500_256answers_per_question-with_scores.jsonl \
#     --batch_size -1 \

# python reward_model/Qwen2.5-Math-RM-72B/qwen_get_score.py \
#     --model_name_or_path /mnt/default/Inference-Scaling/Reward_Model/Qwen2.5-Math-RM-72B \
#     --test_set_path /mnt/default/Mirage/sample/Qwen-vllm0.6.3.post1/Qwen2.5_Math_7B_Instruct-OmniMATH500_256answers_per_question.jsonl \
#     --output_path /mnt/default/Mirage/sample/Qwen-with-scores/Qwen2.5-MATH-RM-72B/Qwen2.5_Math_7B_Instruct-OmniMATH500_256answers_per_question-with_scores.jsonl \
#     --batch_size -1 \

#* Qwen2.5_Math_72B_Instruct
# python reward_model/Qwen2.5-Math-RM-72B/qwen_get_score.py \
#     --model_name_or_path /mnt/default/Inference-Scaling/Reward_Model/Qwen2.5-Math-RM-72B \
#     --test_set_path /mnt/default/Mirage/sample/Qwen-vllm0.6.3.post1/Qwen2.5_Math_72B_Instruct-aime_256answers_per_question.jsonl \
#     --output_path /mnt/default/Mirage/sample/Qwen-with-scores/Qwen2.5-MATH-RM-72B/Qwen2.5_Math_72B_Instruct-aime_256answers_per_question-with_scores.jsonl \
#     --batch_size -1 \

# python reward_model/Qwen2.5-Math-RM-72B/qwen_get_score.py \
#     --model_name_or_path /mnt/default/Inference-Scaling/Reward_Model/Qwen2.5-Math-RM-72B \
#     --test_set_path /mnt/default/Mirage/sample/Qwen-vllm0.6.3.post1/Qwen2.5_Math_72B_Instruct-math500_256answers_per_question.jsonl \
#     --output_path /mnt/default/Mirage/sample/Qwen-with-scores/Qwen2.5-MATH-RM-72B/Qwen2.5_Math_72B_Instruct-math500_256answers_per_question-with_scores.jsonl \
#     --batch_size -1 \

# python reward_model/Qwen2.5-Math-RM-72B/qwen_get_score.py \
#     --model_name_or_path /mnt/default/Inference-Scaling/Reward_Model/Qwen2.5-Math-RM-72B \
#     --test_set_path /mnt/default/Mirage/sample/Qwen-vllm0.6.3.post1/Qwen2.5_Math_72B_Instruct-OmniMATH500_256answers_per_question.jsonl \
#     --output_path /mnt/default/Mirage/sample/Qwen-with-scores/Qwen2.5-MATH-RM-72B/Qwen2.5_Math_72B_Instruct-OmniMATH500_256answers_per_question-with_scores.jsonl \
#     --batch_size -1 \

# python reward_model/Qwen2.5-Math-RM-72B/qwen_get_score.py \
#     --model_name_or_path /mnt/default/Inference-Scaling/Reward_Model/Qwen2.5-Math-RM-72B \
#     --test_set_path /mnt/default/Mirage/sample/Qwen-vllm0.6.3.post1/100test/OmniMATH100-rule/Qwen2.5_Math_72B_Instruct-OmniMATH100-rule_256answers_per_question.jsonl \
#     --output_path /mnt/default/Mirage/sample/Qwen-with-scores/Qwen2.5-MATH-RM-72B/100test/OmniMATH100-rule/Qwen2.5_Math_72B_Instruct-OmniMATH100-rule_256answers_per_question-with_scores.jsonl \
#     --batch_size -1 \

#* llama dvts
# python reward_model/Qwen2.5-Math-RM-72B/dvts_mcts_get_score.py \
#     --model_name_or_path /mnt/default/Inference-Scaling/Reward_Model/Qwen2.5-Math-RM-72B \
#     --test_set_path /mnt/default/Mirage/search/Llama-bug-fixed/Skywork-PRM-vllm0.6.4.post1/post-process/llama3.2_3B_Instruct-aime-dvts_256answers_per_question-sorted.jsonl \
#     --output_path /mnt/default/Mirage/search/Llama-bug-fixed-with-scores/Qwen2.5-MATH-RM-72B/llama3.2_3B_Instruct-aime-dvts_256answers_per_question-with_scores.jsonl \
#     --batch_size -1 \

# python reward_model/Qwen2.5-Math-RM-72B/dvts_mcts_get_score.py \
#     --model_name_or_path /mnt/default/Inference-Scaling/Reward_Model/Qwen2.5-Math-RM-72B \
#     --test_set_path /mnt/default/Mirage/search/Llama-bug-fixed/Skywork-PRM-vllm0.6.4.post1/post-process/llama3.2_3B_Instruct-math500-dvts_256answers_per_question-sorted.jsonl \
#     --output_path /mnt/default/Mirage/search/Llama-bug-fixed-with-scores/Qwen2.5-MATH-RM-72B/llama3.2_3B_Instruct-math500-dvts_256answers_per_question-with_scores.jsonl \
#     --batch_size -1 \

# python reward_model/Qwen2.5-Math-RM-72B/dvts_mcts_get_score.py \
#     --model_name_or_path /mnt/default/Inference-Scaling/Reward_Model/Qwen2.5-Math-RM-72B \
#     --test_set_path /mnt/default/Mirage/search/Llama-bug-fixed/Skywork-PRM-vllm0.6.4.post1/post-process/llama3.2_3B_Instruct-OmniMATH500-dvts_256answers_per_question-sorted.jsonl \
#     --output_path /mnt/default/Mirage/search/Llama-bug-fixed-with-scores/Qwen2.5-MATH-RM-72B/llama3.2_3B_Instruct-OmniMATH500-dvts_256answers_per_question-with_scores.jsonl \
#     --batch_size -1 \

# python reward_model/Qwen2.5-Math-RM-72B/dvts_mcts_get_score.py \
#     --model_name_or_path /mnt/default/Inference-Scaling/Reward_Model/Qwen2.5-Math-RM-72B \
#     --test_set_path /mnt/default/Mirage/search/Llama-bug-fixed/Skywork-PRM-vllm0.6.4.post1/post-process/llama3.1_8B_Instruct-aime-dvts_256answers_per_question-sorted.jsonl \
#     --output_path /mnt/default/Mirage/search/Llama-bug-fixed-with-scores/Qwen2.5-MATH-RM-72B/llama3.1_8B_Instruct-aime-dvts_256answers_per_question-with_scores.jsonl \
#     --batch_size -1 \

# python reward_model/Qwen2.5-Math-RM-72B/dvts_mcts_get_score.py \
#     --model_name_or_path /mnt/default/Inference-Scaling/Reward_Model/Qwen2.5-Math-RM-72B \
#     --test_set_path /mnt/default/Mirage/search/Llama-bug-fixed/Skywork-PRM-vllm0.6.4.post1/post-process/llama3.1_8B_Instruct-math500-dvts_256answers_per_question-sorted.jsonl \
#     --output_path /mnt/default/Mirage/search/Llama-bug-fixed-with-scores/Qwen2.5-MATH-RM-72B/llama3.1_8B_Instruct-math500-dvts_256answers_per_question-with_scores.jsonl \
#     --batch_size -1 \

# python reward_model/Qwen2.5-Math-RM-72B/dvts_mcts_get_score.py \
#     --model_name_or_path /mnt/default/Inference-Scaling/Reward_Model/Qwen2.5-Math-RM-72B \
#     --test_set_path /mnt/default/Mirage/search/Llama-bug-fixed/Skywork-PRM-vllm0.6.4.post1/post-process/llama3.1_8B_Instruct-OmniMATH500-dvts_256answers_per_question-sorted.jsonl \
#     --output_path /mnt/default/Mirage/search/Llama-bug-fixed-with-scores/Qwen2.5-MATH-RM-72B/llama3.1_8B_Instruct-OmniMATH500-dvts_256answers_per_question-with_scores.jsonl \
#     --batch_size -1 \

#* qwen dvts
# python reward_model/Qwen2.5-Math-RM-72B/dvts_mcts_get_score.py \
#     --model_name_or_path /mnt/default/Inference-Scaling/Reward_Model/Qwen2.5-Math-RM-72B \
#     --test_set_path /mnt/default/Mirage/search/Qwen-bug-fixed/Skywork-PRM-vllm0.6.4.post1/post-process/Qwen2.5_Math_1.5B_Instruct-aime-dvts_256answers_per_question-sorted.jsonl \
#     --output_path /mnt/default/Mirage/search/Qwen-bug-fixed-with-scores/Qwen2.5-MATH-RM-72B/Qwen2.5_Math_1.5B_Instruct-aime-dvts_256answers_per_question-with_scores.jsonl \
#     --batch_size -1 \

# python reward_model/Qwen2.5-Math-RM-72B/dvts_mcts_get_score.py \
#     --model_name_or_path /mnt/default/Inference-Scaling/Reward_Model/Qwen2.5-Math-RM-72B \
#     --test_set_path /mnt/default/Mirage/search/Qwen-bug-fixed/Skywork-PRM-vllm0.6.4.post1/post-process/Qwen2.5_Math_1.5B_Instruct-math500-dvts_256answers_per_question-sorted.jsonl \
#     --output_path /mnt/default/Mirage/search/Qwen-bug-fixed-with-scores/Qwen2.5-MATH-RM-72B/Qwen2.5_Math_1.5B_Instruct-math500-dvts_256answers_per_question-with_scores.jsonl \
#     --batch_size -1 \

# python reward_model/Qwen2.5-Math-RM-72B/dvts_mcts_get_score.py \
#     --model_name_or_path /mnt/default/Inference-Scaling/Reward_Model/Qwen2.5-Math-RM-72B \
#     --test_set_path /mnt/default/Mirage/search/Qwen-bug-fixed/Skywork-PRM-vllm0.6.4.post1/post-process/Qwen2.5_Math_1.5B_Instruct-OmniMATH500-dvts_256answers_per_question-sorted.jsonl \
#     --output_path /mnt/default/Mirage/search/Qwen-bug-fixed-with-scores/Qwen2.5-MATH-RM-72B/Qwen2.5_Math_1.5B_Instruct-OmniMATH500-dvts_256answers_per_question-with_scores.jsonl \
#     --batch_size -1 \

# python reward_model/Qwen2.5-Math-RM-72B/dvts_mcts_get_score.py \
#     --model_name_or_path /mnt/default/Inference-Scaling/Reward_Model/Qwen2.5-Math-RM-72B \
#     --test_set_path /mnt/default/search/Qwen-bug-fixed/Skywork-PRM-vllm0.6.4.post1/post-process/Qwen2.5_Math_7B_Instruct-aime-dvts_256answers_per_question-sorted.jsonl \
#     --output_path /mnt/default/Mirage/search/Qwen-bug-fixed-with-scores/Qwen2.5-MATH-RM-72B/Qwen2.5_Math_7B_Instruct-aime-dvts_256answers_per_question-with_scores.jsonl \
#     --batch_size -1 \

# python reward_model/Qwen2.5-Math-RM-72B/dvts_mcts_get_score.py \
#     --model_name_or_path /mnt/default/Inference-Scaling/Reward_Model/Qwen2.5-Math-RM-72B \
#     --test_set_path /mnt/default/search/Qwen-bug-fixed/Skywork-PRM-vllm0.6.4.post1/post-process/Qwen2.5_Math_7B_Instruct-math500-dvts_256answers_per_question-sorted.jsonl \
#     --output_path /mnt/default/Mirage/search/Qwen-bug-fixed-with-scores/Qwen2.5-MATH-RM-72B/Qwen2.5_Math_7B_Instruct-math500-dvts_256answers_per_question-with_scores.jsonl \
#     --batch_size -1 \

# python reward_model/Qwen2.5-Math-RM-72B/dvts_mcts_get_score.py \
#     --model_name_or_path /mnt/default/Inference-Scaling/Reward_Model/Qwen2.5-Math-RM-72B \
#     --test_set_path /mnt/default/search/Qwen-bug-fixed/Skywork-PRM-vllm0.6.4.post1/post-process/Qwen2.5_Math_7B_Instruct-OmniMATH500-dvts_256answers_per_question-sorted.jsonl \
#     --output_path /mnt/default/Mirage/search/Qwen-bug-fixed-with-scores/Qwen2.5-MATH-RM-72B/Qwen2.5_Math_7B_Instruct-OmniMATH500-dvts_256answers_per_question-with_scores.jsonl \
#     --batch_size -1 \

#* llama mcts
# python reward_model/Qwen2.5-Math-RM-72B/dvts_mcts_get_score.py \
#     --model_name_or_path /mnt/default/Inference-Scaling/Reward_Model/Qwen2.5-Math-RM-72B \
#     --test_set_path /mnt/default/Mirage/mcts/Llama-bug-fixed/Skywork-PRM-vllm0.6.4.post1/post-process/llama3.2_3B_Instruct-aime-mcts_256answers_per_question-sorted.jsonl \
#     --output_path /mnt/default/Mirage/mcts/Llama-bug-fixed-with-scores/Qwen2.5-MATH-RM-72B/llama3.2_3B_Instruct-aime-mcts_256answers_per_question-with_scores.jsonl \
#     --batch_size -1 \

# python reward_model/Qwen2.5-Math-RM-72B/dvts_mcts_get_score.py \
#     --model_name_or_path /mnt/default/Inference-Scaling/Reward_Model/Qwen2.5-Math-RM-72B \
#     --test_set_path /mnt/default/Mirage/mcts/Llama-bug-fixed/Skywork-PRM-vllm0.6.4.post1/post-process/llama3.2_3B_Instruct-math500-mcts_256answers_per_question-sorted.jsonl \
#     --output_path /mnt/default/Mirage/mcts/Llama-bug-fixed-with-scores/Qwen2.5-MATH-RM-72B/llama3.2_3B_Instruct-math500-mcts_256answers_per_question-with_scores.jsonl \
#     --batch_size -1 \

# python reward_model/Qwen2.5-Math-RM-72B/dvts_mcts_get_score.py \
#     --model_name_or_path /mnt/default/Inference-Scaling/Reward_Model/Qwen2.5-Math-RM-72B \
#     --test_set_path /mnt/default/Mirage/mcts/Llama-bug-fixed/Skywork-PRM-vllm0.6.4.post1/post-process/llama3.2_3B_Instruct-OmniMATH500-mcts_256answers_per_question-sorted.jsonl \
#     --output_path /mnt/default/Mirage/mcts/Llama-bug-fixed-with-scores/Qwen2.5-MATH-RM-72B/llama3.2_3B_Instruct-OmniMATH500-mcts_256answers_per_question-with_scores.jsonl \
#     --batch_size -1 \

# python reward_model/Qwen2.5-Math-RM-72B/dvts_mcts_get_score.py \
#     --model_name_or_path /mnt/default/Inference-Scaling/Reward_Model/Qwen2.5-Math-RM-72B \
#     --test_set_path /mnt/default/Mirage/mcts/Llama-bug-fixed/Skywork-PRM-vllm0.6.4.post1/post-process/llama3.1_8B_Instruct-aime-mcts_256answers_per_question-sorted.jsonl \
#     --output_path /mnt/default/Mirage/mcts/Llama-bug-fixed-with-scores/Qwen2.5-MATH-RM-72B/llama3.1_8B_Instruct-aime-mcts_256answers_per_question-with_scores.jsonl \
#     --batch_size -1 \

# python reward_model/Qwen2.5-Math-RM-72B/dvts_mcts_get_score.py \
#     --model_name_or_path /mnt/default/Inference-Scaling/Reward_Model/Qwen2.5-Math-RM-72B \
#     --test_set_path /mnt/default/Mirage/mcts/Llama-bug-fixed/Skywork-PRM-vllm0.6.4.post1/post-process/llama3.1_8B_Instruct-math500-mcts_256answers_per_question-sorted.jsonl \
#     --output_path /mnt/default/Mirage/mcts/Llama-bug-fixed-with-scores/Qwen2.5-MATH-RM-72B/llama3.1_8B_Instruct-math500-mcts_256answers_per_question-with_scores.jsonl \
#     --batch_size -1 \

# python reward_model/Qwen2.5-Math-RM-72B/dvts_mcts_get_score.py \
#     --model_name_or_path /mnt/default/Inference-Scaling/Reward_Model/Qwen2.5-Math-RM-72B \
#     --test_set_path /mnt/default/Mirage/mcts/Llama-bug-fixed/Skywork-PRM-vllm0.6.4.post1/post-process/llama3.1_8B_Instruct-OmniMATH500-mcts_256answers_per_question-sorted.jsonl \
#     --output_path /mnt/default/Mirage/mcts/Llama-bug-fixed-with-scores/Qwen2.5-MATH-RM-72B/llama3.1_8B_Instruct-OmniMATH500-mcts_256answers_per_question-with_scores.jsonl \
#     --batch_size -1 \

#* qwen mcts
# python reward_model/Qwen2.5-Math-RM-72B/dvts_mcts_get_score.py \
#     --model_name_or_path /mnt/default/Inference-Scaling/Reward_Model/Qwen2.5-Math-RM-72B \
#     --test_set_path /mnt/default/Mirage/mcts/Qwen-bug-fixed/Skywork-PRM-vllm0.6.4.post1/post-process/Qwen2.5_Math_1.5B_Instruct-aime-mcts_256answers_per_question-sorted.jsonl \
#     --output_path /mnt/default/Mirage/mcts/Qwen-bug-fixed-with-scores/Qwen2.5-MATH-RM-72B/Qwen2.5_Math_1.5B_Instruct-aime-mcts_256answers_per_question-with_scores.jsonl \
#     --batch_size -1 \

# python reward_model/Qwen2.5-Math-RM-72B/dvts_mcts_get_score.py \
#     --model_name_or_path /mnt/default/Inference-Scaling/Reward_Model/Qwen2.5-Math-RM-72B \
#     --test_set_path /mnt/default/Mirage/mcts/Qwen-bug-fixed/Skywork-PRM-vllm0.6.4.post1/post-process/Qwen2.5_Math_1.5B_Instruct-math500-mcts_256answers_per_question-sorted.jsonl \
#     --output_path /mnt/default/Mirage/mcts/Qwen-bug-fixed-with-scores/Qwen2.5-MATH-RM-72B/Qwen2.5_Math_1.5B_Instruct-math500-mcts_256answers_per_question-with_scores.jsonl \
#     --batch_size -1 \

# python reward_model/Qwen2.5-Math-RM-72B/dvts_mcts_get_score.py \
#     --model_name_or_path /mnt/default/Inference-Scaling/Reward_Model/Qwen2.5-Math-RM-72B \
#     --test_set_path /mnt/default/Mirage/mcts/Qwen-bug-fixed/Skywork-PRM-vllm0.6.4.post1/post-process/Qwen2.5_Math_1.5B_Instruct-OmniMATH500-mcts_256answers_per_question-sorted.jsonl \
#     --output_path /mnt/default/Mirage/mcts/Qwen-bug-fixed-with-scores/Qwen2.5-MATH-RM-72B/Qwen2.5_Math_1.5B_Instruct-OmniMATH500-mcts_256answers_per_question-with_scores.jsonl \
#     --batch_size -1 \

# python reward_model/Qwen2.5-Math-RM-72B/dvts_mcts_get_score.py \
#     --model_name_or_path /mnt/default/Inference-Scaling/Reward_Model/Qwen2.5-Math-RM-72B \
#     --test_set_path /mnt/default/Mirage/mcts/Qwen-bug-fixed/Skywork-PRM-vllm0.6.4.post1/post-process/Qwen2.5_Math_7B_Instruct-aime-mcts_256answers_per_question-sorted.jsonl \
#     --output_path /mnt/default/Mirage/mcts/Qwen-bug-fixed-with-scores/Qwen2.5-MATH-RM-72B/Qwen2.5_Math_7B_Instruct-aime-mcts_256answers_per_question-with_scores.jsonl \
#     --batch_size -1 \

# python reward_model/Qwen2.5-Math-RM-72B/dvts_mcts_get_score.py \
#     --model_name_or_path /mnt/default/Inference-Scaling/Reward_Model/Qwen2.5-Math-RM-72B \
#     --test_set_path /mnt/default/Mirage/mcts/Qwen-bug-fixed/Skywork-PRM-vllm0.6.4.post1/post-process/Qwen2.5_Math_7B_Instruct-math500-mcts_256answers_per_question-sorted.jsonl \
#     --output_path /mnt/default/Mirage/mcts/Qwen-bug-fixed-with-scores/Qwen2.5-MATH-RM-72B/Qwen2.5_Math_7B_Instruct-math500-mcts_256answers_per_question-with_scores.jsonl \
#     --batch_size -1 \

# python reward_model/Qwen2.5-Math-RM-72B/dvts_mcts_get_score.py \
#     --model_name_or_path /mnt/default/Inference-Scaling/Reward_Model/Qwen2.5-Math-RM-72B \
#     --test_set_path /mnt/default/Mirage/mcts/Qwen-bug-fixed/Skywork-PRM-vllm0.6.4.post1/post-process/Qwen2.5_Math_7B_Instruct-OmniMATH500-mcts_256answers_per_question-sorted.jsonl \
#     --output_path /mnt/default/Mirage/mcts/Qwen-bug-fixed-with-scores/Qwen2.5-MATH-RM-72B/Qwen2.5_Math_7B_Instruct-OmniMATH500-mcts_256answers_per_question-with_scores.jsonl \
#     --batch_size -1 \