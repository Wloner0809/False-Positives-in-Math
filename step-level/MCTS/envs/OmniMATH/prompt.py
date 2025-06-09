# #! qwen prompt
# COT_EXAMPLES = None
# COT_TASK_DESC = "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>"
# PROBLEM_FORMAT_STR = (
#     """<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"""
# )

# SEP = "\n\n"

#! llama prompt
Llama = "\n"
COT_EXAMPLES = None
COT_TASK_DESC = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nSolve the following math problem efficiently and clearly:\n\n- For simple problems (2 steps or fewer):\nProvide a concise solution with minimal explanation.\n\n- For complex problems (3 steps or more):\nUse this step-by-step format:\n\n## Step 1: [Concise description]\n[Brief explanation and calculations]\n\n## Step 2: [Concise description]\n[Brief explanation and calculations]\n\n...\n\nRegardless of the approach, always conclude with:\n\nTherefore, the final answer is: $\\boxed{answer}$. I hope it is correct.\n\nWhere [answer] is just the final number or expression that solves the problem.\n"
PROBLEM_FORMAT_STR = (
    """Problem: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"""
)

SEP = "\n\n"
