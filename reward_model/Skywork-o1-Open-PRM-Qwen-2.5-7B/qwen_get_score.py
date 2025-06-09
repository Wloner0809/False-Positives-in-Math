import json
import os
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
from datasets import load_dataset
from model_utils.io_utils import derive_step_rewards_vllm, prepare_input
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser


@dataclass
class ScriptArguments:
    model_name_or_path: Optional[str] = field(
        default="/home/v-wangyu1/model/Skywork-o1-Open-PRM-Qwen-2.5-7B"
    )
    test_set_path: Optional[str] = field(
        default="/mnt/blob/Inference-Scaling/sample/Llama-prm-format/llama3.2_3B_Instruct-math500_256answers_per_question_skywork_format.jsonl"
    )
    output_path: Optional[str] = field(
        default="/mnt/blob/Inference-Scaling/sample/Llama-with-scores/Skywork-o1-Open-PRM-Qwen-2.5-7B/llama3.1_70B_Instruct-math500_256answers_per_question-with_scores.jsonl",
    )
    batch_size: Optional[int] = field(default=-1)


def get_qwen_test_set(test_set_path, tokenizer, max_tokens=4096):
    test_set = load_dataset("json", data_files=test_set_path)["train"]
    test_list = []
    for test_data in tqdm(test_set, desc="Process Data", total=len(test_set)):
        single_test_list = []
        problem = (
            test_data["prompt"].split("<|im_start|>user\n")[1].split("<|im_end|>")[0]
        )
        for res in test_data["responses"]:
            single_test_list.append(
                prepare_input(
                    problem=problem,
                    response=res,
                    tokenizer=tokenizer,
                    step_token="\n\n",
                    max_tokens=max_tokens,
                )
            )
        test_list.append(single_test_list)
    return test_list, test_set["prompt"], test_set["responses"]


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    tokenizer = AutoTokenizer.from_pretrained(
        script_args.model_name_or_path, trust_remote_code=True
    )
    # ----------------- call vllm server -----------------
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8081/v1"
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    models = client.models.list()
    model = models.data[0].id
    # -----------------------------------------------------
    test_list, prompt_list, response_list = get_qwen_test_set(
        script_args.test_set_path, tokenizer
    )
    if script_args.batch_size != -1:
        for i in range(0, len(test_list), script_args.batch_size):
            batch_test_list = test_list[i : i + script_args.batch_size]
            batch_prompt = prompt_list[i : i + script_args.batch_size]
            batch_responses = response_list[i : i + script_args.batch_size]
            batch_scores = []
            if os.path.exists(script_args.output_path):
                df = pd.read_json(script_args.output_path, lines=True)
                if batch_prompt[0] in df["prompt"].tolist():
                    continue
            for single_test_list in tqdm(
                batch_test_list, desc="Scoring", total=len(batch_test_list)
            ):
                input_ids, steps, reward_flags = zip(*single_test_list)
                rewards = client.embeddings.create(
                    input=input_ids,
                    model=model,
                )
                step_rewards = derive_step_rewards_vllm(rewards, reward_flags)
                batch_scores.append(step_rewards)
            with open(script_args.output_path, "a", encoding="utf-8") as f:
                for i, single_scores in enumerate(batch_scores):
                    f.write(
                        json.dumps(
                            {
                                "prompt": batch_prompt[i],
                                "responses": batch_responses[i],
                                "step_scores": single_scores,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
    else:
        if os.path.exists(script_args.output_path):
            print("Output file already exists.\nExiting.")
        else:
            scores = []
            for single_test_list in tqdm(
                test_list, desc="Scoring", total=len(test_list)
            ):
                input_ids, steps, reward_flags = zip(*single_test_list)
                rewards = client.embeddings.create(
                    input=input_ids,
                    model=model,
                )
                step_rewards = derive_step_rewards_vllm(rewards, reward_flags)
                scores.append(step_rewards)
            gathered_data = []
            for i, single_scores in enumerate(scores):
                gathered_data.append(
                    {
                        "prompt": prompt_list[i],
                        "responses": response_list[i],
                        "step_scores": single_scores,
                    }
                )
            df = pd.DataFrame(gathered_data)
            df.to_json(script_args.output_path, orient="records", lines=True)
