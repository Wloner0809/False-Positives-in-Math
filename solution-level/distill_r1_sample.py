import json
import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
)
from vllm import LLM, SamplingParams


@dataclass
class ScriptArguments:
    # --------------------- Data Arguments -------------------------
    model_name_or_path: Optional[str] = field(
        default="/home/v-wangyu1/model/DeepSeek-R1-Distill-Qwen-7B",
        metadata={"help": "the location of the model"},
    )
    dataset_name_or_path: Optional[str] = field(
        default="AI-MO/aimo-validation-aime",
        metadata={"help": "the location of the dataset name or path"},
    )
    output_dir: Optional[str] = field(
        default="output.jsonl",
        metadata={"help": "the location of the output file"},
    )
    batch_size: Optional[int] = field(
        default=50,
        metadata={"help": "the number of problems to save"},
    )
    # --------------------- Sampling Arguments ---------------------
    max_new_tokens: Optional[int] = field(
        default=32768,
        metadata={"help": "the maximum length of the new tokens"},
    )
    seed: Optional[int] = field(
        default=42,
        metadata={"help": "the random seed"},
    )
    temperature: Optional[float] = field(
        default=0.6,
        metadata={"help": "the temperature"},
    )
    eos_ids: List[int] = field(
        default_factory=lambda: [],
        metadata={"help": "the ids of the end of sentence tokens"},
    )
    n: Optional[int] = field(
        default=64,
        metadata={"help": "the number of output sequences"},
    )
    # ----------------------- LLM Arguments ------------------------
    max_model_len: Optional[int] = field(
        default=32768,
        metadata={"help": "the maximum length of the input tokens"},
    )
    tensor_parallel_size: Optional[int] = field(
        default=1,
        metadata={"help": "the tensor parallel size"},
    )
    swap_space: Optional[float] = field(
        default=32.0,
        metadata={"help": "the swap space"},
    )
    gpu_memory_utilization: Optional[float] = field(
        default=0.95,
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


def generate_benchmark_answer_distill_r1(
    benchmark_name_or_path,
    output_path,
    llm: LLM,
    sampling_params: SamplingParams,
    batch_size: int,
    tokenizer: AutoTokenizer,
):
    if benchmark_name_or_path == "qq8933/MATH500":
        dataset = load_dataset(benchmark_name_or_path, split="test")
        dataset = dataset.map(
            lambda x: {
                "prompt": tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": x["problem"]},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                ).strip(" ")
            },
            num_proc=16,
        )
    elif benchmark_name_or_path == "openai/gsm8k":
        dataset = load_dataset(path=benchmark_name_or_path, name="main", split="test")
        dataset = dataset.map(
            lambda x: {
                "prompt": tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": x["question"]},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                ).strip(" ")
            },
            num_proc=16,
        )
    elif benchmark_name_or_path == "KbsdJames/Omni-MATH":
        dataset = load_dataset(path=benchmark_name_or_path, split="test")
        dataset = dataset.map(
            lambda x: {
                "prompt": tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": x["problem"]},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                ).strip(" ")
            },
            num_proc=16,
        )
    elif benchmark_name_or_path == "AI-MO/aimo-validation-aime":
        dataset = load_dataset(path=benchmark_name_or_path, split="train")
        dataset = dataset.map(
            lambda x: {
                "prompt": tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": x["problem"]},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                ).strip(" ")
            },
            num_proc=16,
        )
    elif "Omni-MATH500" in benchmark_name_or_path:
        dataset = load_dataset("json", data_files=benchmark_name_or_path, split="train")
        dataset = dataset.map(
            lambda x: {
                "prompt": tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": x["problem"]},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                ).strip(" ")
            },
            num_proc=16,
        )
    elif "OmniMATH100-rule" in benchmark_name_or_path:
        dataset = load_dataset("json", data_files=benchmark_name_or_path, split="train")
        dataset = dataset.map(
            lambda x: {
                "prompt": tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": x["problem"]},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                ).strip(" ")
            },
            num_proc=16,
        )

    print(f"Total number of problems: {len(dataset)}")
    print("Example prompt:\n", dataset["prompt"][0])

    # generate the outputs
    if batch_size != -1:
        prompts = dataset["prompt"]
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i : i + batch_size]
            if os.path.exists(output_path):
                df = pd.read_json(output_path, lines=True)
                if (
                    batch_prompts[0] in df["prompt"].tolist()
                ):  #! assume all the question occurs only once
                    continue
            outputs = llm.generate(
                batch_prompts, sampling_params=sampling_params, use_tqdm=True
            )
            for j, output in enumerate(outputs):
                tmp_data = {
                    "prompt": batch_prompts[j],
                    "responses": [out.text for out in output.outputs],
                }
                with open(output_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(tmp_data, ensure_ascii=False) + "\n")
    else:
        prompts = dataset["prompt"]
        outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=True)
        gathered_data = []
        for i, output in enumerate(outputs):
            tmp_data = {
                "prompt": prompts[i],
                "responses": [out.text for out in output.outputs],
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
        dtype="auto",
        max_model_len=script_args.max_model_len,
        load_format="auto",
        seed=script_args.seed,
        tensor_parallel_size=script_args.tensor_parallel_size,
        swap_space=script_args.swap_space,
        **kwargs,
    )

    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(
        n=script_args.n,
        temperature=script_args.temperature,
        top_p=0.95,
        top_k=-1,
        seed=seed,
        max_tokens=script_args.max_new_tokens,
    )

    generate_benchmark_answer_distill_r1(
        script_args.dataset_name_or_path,
        script_args.output_dir,
        llm,
        sampling_params,
        script_args.batch_size,
        tokenizer,
    )
