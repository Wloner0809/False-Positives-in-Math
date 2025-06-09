import os
import json
import pandas as pd
from transformers import AutoTokenizer, HfArgumentParser
from openai import OpenAI
from dataclasses import dataclass, field
from typing import Optional
from datasets import load_dataset
from tqdm import tqdm


@dataclass
class ScriptArguments:
    model_name_or_path: Optional[str] = field(
        default="/mnt/default/Inference-Scaling/Reward_Model/Qwen2.5-Math-RM-72B"
    )
    test_set_path: Optional[str] = field(
        default="/mnt/default/Inference-Scaling/sample/Llama/llama3.1_70B_Instruct-math500_256answers_per_question.jsonl"
    )
    output_path: Optional[str] = field(
        default="/mnt/default/Inference-Scaling/sample/Llama-with-scores/Qwen2.5-MATH-RM-72B/llama3.1_70B_Instruct-math500_256answers_per_question-with_scores.jsonl",
    )
    batch_size: Optional[int] = field(default=-1)


def get_test_set(test_set_path, tokenizer, max_tokens=4096):
    test_set = load_dataset("json", data_files=test_set_path)["train"]
    test_list = []
    for test_data in tqdm(test_set, desc="Process Data", total=len(test_set)):
        single_test_list = []
        problem = test_data["prompt"]
        for res in test_data["responses"]:
            message = [
                {
                    "role": "system",
                    "content": "Please reason step by step, and put your final answer within \\boxed{}.",
                },
                {
                    "role": "user",
                    "content": problem,
                },
                {
                    "role": "assistant",
                    "content": res,
                },
            ]
            message_str = tokenizer.apply_chat_template(
                message, tokenize=False, add_generation_prompt=False
            )
            tokens = tokenizer.encode(message_str, add_special_tokens=False)
            if len(tokens) > max_tokens:
                message_str = tokenizer.decode(tokens[-max_tokens:])
            single_test_list.append(message_str)
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
    openai_api_base = "http://localhost:8000/v1"
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    models = client.models.list()
    model = models.data[0].id
    # -----------------------------------------------------
    test_list, original_prompt, original_responses = get_test_set(
        script_args.test_set_path, tokenizer
    )
    if script_args.batch_size != -1:
        for i in range(0, len(test_list), script_args.batch_size):
            batch_test_list = test_list[i : i + script_args.batch_size]
            batch_prompt = original_prompt[i : i + script_args.batch_size]
            batch_responses = original_responses[i : i + script_args.batch_size]
            batch_scores = []
            if os.path.exists(script_args.output_path):
                df = pd.read_json(script_args.output_path, lines=True)
                if batch_test_list[0] in df["qwen_rm_prompt"].tolist():
                    continue
            for single_test_list in tqdm(
                batch_test_list, desc="Scoring", total=len(batch_test_list)
            ):
                single_scores = []
                responses = client.embeddings.create(
                    input=single_test_list,
                    model=model,
                )
                for data in responses.data:
                    single_scores.append(data.embedding[-1])
                batch_scores.append(single_scores)
            with open(script_args.output_path, "a", encoding="utf-8") as f:
                for i, single_scores in enumerate(batch_scores):
                    f.write(
                        json.dumps(
                            {
                                "prompt": batch_prompt[i],
                                "responses": batch_responses[i],
                                "scores": single_scores,
                                "qwen_rm_prompt": batch_test_list[i],
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
    else:
        if os.path.exists(script_args.output_path):
            print("Output file already exists.")
        else:
            scores = []
            for single_test_list in tqdm(
                test_list, desc="Scoring", total=len(test_list)
            ):
                single_scores = []
                responses = client.embeddings.create(
                    input=single_test_list,
                    model=model,
                )
                for data in responses.data:
                    single_scores.append(data.embedding[-1])
                scores.append(single_scores)
            gathered_data = []
            for i, single_scores in enumerate(scores):
                gathered_data.append(
                    {
                        "prompt": original_prompt[i],
                        "responses": original_responses[i],
                        "scores": single_scores,
                        "qwen_rm_prompt": test_list[i],
                    }
                )
            df = pd.DataFrame(gathered_data)
            df.to_json(script_args.output_path, orient="records", lines=True)
