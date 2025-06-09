from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
)
from vllm import LLM, SamplingParams

prompt = (
    "Problem:\n{problem}\n\n"
    + "Solution:\n{solution}\n\n"
    + "Based on the problem and solution provided above:\n"
    + "1. Output True if the solution is considered correct.\n"
    + "2. Output False if the solution is considered incorrect and contains some errors.\n\n"
    + "Please comprehensively evaluate all the steps in the solution and provide only True or False as your final output."
)


@dataclass
class ScriptArguments:
    # --------------------- Data Arguments -------------------------
    model_name_or_path: Optional[str] = field(
        default="meta-llama/Llama-3.1-8B-Instruct",
        metadata={"help": "the location of the instruct model name or path"},
    )
    check_path: Optional[str] = field(
        default="/mnt/blob/Mirage/search/Llama-results/100test/math100/llama3.1_8B/BoN_Reward_correct_256samples-Skywork-o1-Open-PRM-Qwen-2.5-7B.jsonl",
        metadata={"help": "the location of the test set"},
    )
    output_dir: Optional[str] = field(
        default="output.jsonl",
        metadata={"help": "the location of the output file"},
    )
    batch_size: Optional[int] = field(
        default=-1,
        metadata={"help": "the number of problems to save"},
    )
    # --------------------- Sampling Arguments ---------------------
    max_new_tokens: Optional[int] = field(
        default=4096,
        metadata={"help": "the maximum length of the new tokens"},
    )
    seed: Optional[int] = field(
        default=42,
        metadata={"help": "the random seed"},
    )
    eos_ids: List[int] = field(
        default_factory=lambda: [128001, 128008],
        metadata={"help": "the ids of the end of sentence tokens"},
    )
    # ----------------------- LLM Arguments ------------------------
    max_model_len: Optional[int] = field(
        default=8192,
        metadata={"help": "the maximum length of the input tokens"},
    )
    tensor_parallel_size: Optional[int] = field(
        default=1,
        metadata={"help": "the tensor parallel size"},
    )
    swap_space: Optional[float] = field(
        default=8.0,
        metadata={"help": "the swap space"},
    )
    gpu_memory_utilization: Optional[float] = field(
        default=-1,
        metadata={"help": "the gpu memory utilization, default is 0.9 in vLLM"},
    )
    # ----------------------- Merely Use ---------------------------
    max_num_seqs: Optional[int] = field(
        default=-1,
        metadata={"help": "the maximum number of sequences, default is 256 in vLLM"},
    )
    cpu_offload_gb: Optional[float] = field(
        default=-1,
        metadata={"help": "the cpu offload gb"},
    )
    preemption_mode: Optional[str] = field(
        default=None,
        metadata={"help": "the preemption mode"},
    )
    use_v2_block_manager: Optional[bool] = field(
        default=False,
        metadata={"help": "the use v2 block manager"},
    )
    max_num_batched_tokens: Optional[int] = field(
        default=-1,
        metadata={
            "help": "the maximum number of batched tokens, default is 512 in vLLM"
        },
    )
    enable_chunked_prefill: Optional[bool] = field(
        default=False,
        metadata={"help": "the enable chunked prefill"},
    )
    enforce_eager: Optional[bool] = field(
        default=False,
        metadata={"help": "the enforce eager"},
    )
    # ----------------------- Merely Use ---------------------------


def check_answer_qwen(
    check_path,
    output_path,
    llm: LLM,
    sampling_params: SamplingParams,
    batch_size: int,
    tokenizer: AutoTokenizer,
):
    df = pd.read_json(check_path, lines=True)
    problem_id = [df.iloc[i]["id"] for i in range(len(df))]
    final_prompt = [
        prompt.format(problem=df.iloc[i]["problem"], solution=df.iloc[i]["solution"])
        for i in range(len(df))
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [
                {
                    "role": "system",
                    "content": "You are an expert mathematician and your task is to verify the correctness of a step-by-step solution to a math problem. Carefully analyze each step for logical consistency, mathematical accuracy, and adherence to any given formulas or rules. Disregard minor errors that do not affect the validity of the final answer or are irrelevant to it.",
                },
                {"role": "user", "content": final_prompt[i]},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        for i in range(len(df))
    ]
    print(prompts[0])
    # generate the outputs
    if batch_size == -1:
        outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=True)
        gathered_data = []
        for i, output in enumerate(outputs):
            tmp_data = {
                "id": str(problem_id[i]),
                "response": output.outputs[0].text,
            }
            gathered_data.append(tmp_data)

        df = pd.DataFrame(gathered_data)
        df.to_json(output_path, orient="records", lines=True)


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    seed = script_args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    kwargs = {}
    if script_args.gpu_memory_utilization >= 0:
        kwargs["gpu_memory_utilization"] = script_args.gpu_memory_utilization
    if script_args.max_num_seqs >= 0:
        kwargs["max_num_seqs"] = script_args.max_num_seqs
    if script_args.cpu_offload_gb >= 0:
        kwargs["cpu_offload_gb"] = script_args.cpu_offload_gb
    if script_args.preemption_mode is not None:
        kwargs["preemption_mode"] = script_args.preemption_mode
    if script_args.use_v2_block_manager:
        kwargs["use_v2_block_manager"] = script_args.use_v2_block_manager
    if script_args.max_num_batched_tokens >= 0:
        kwargs["max_num_batched_tokens"] = script_args.max_num_batched_tokens
    if script_args.enable_chunked_prefill:
        kwargs["enable_chunked_prefill"] = script_args.enable_chunked_prefill
    if script_args.enforce_eager:
        kwargs["enforce_eager"] = script_args.enforce_eager
    llm = LLM(
        model=script_args.model_name_or_path,
        tokenizer=script_args.model_name_or_path,
        dtype="float32",
        max_model_len=script_args.max_model_len,
        load_format="auto",
        seed=script_args.seed,
        tensor_parallel_size=script_args.tensor_parallel_size,
        swap_space=script_args.swap_space,
        **kwargs,
    )

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    sampling_params = SamplingParams(
        n=1,
        temperature=0.0,
        seed=seed,
        stop=[
            "</s>",
            "<|im_end|>",
            "<|endoftext|>",
        ],  # * copied from https://github.com/QwenLM/Qwen2.5-Math/blob/25a9bc84bc4c43bcc499c4bec9ab14a0983d3852/evaluation/math_eval.py#L234C18-L234C57
        stop_token_ids=tokenizer.encode("<|endoftext|>", add_special_tokens=False)
        + tokenizer.encode(
            "<|im_end|>", add_special_tokens=False
        ),  # * copied from https://github.com/QwenLM/Qwen2.5-Math/blob/25a9bc84bc4c43bcc499c4bec9ab14a0983d3852/evaluation/math_eval.py#L269
        max_tokens=script_args.max_new_tokens,
    )

    check_answer_qwen(
        script_args.check_path,
        script_args.output_dir,
        llm,
        sampling_params,
        script_args.batch_size,
        tokenizer,
    )
