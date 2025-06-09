import json
import os
import time
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
from openai import OpenAI, RateLimitError
from tqdm import tqdm
from transformers import HfArgumentParser

prompt = (
    "You are an expert mathematician and your task is to verify the correctness of a step-by-step solution to a math problem. Carefully analyze each step for logical consistency, mathematical accuracy, and adherence to any given formulas or rules. Disregard minor errors that do not affect the validity of the final answer or are irrelevant to it.\n\n"
    + "Problem:\n{problem}\n\n"
    + "Solution:\n{solution}\n\n"
    + "Based on the problem and solution provided above:\n"
    + "1. Output True if the solution is considered correct.\n"
    + "2. Output False if the solution is considered incorrect and contains some errors.\n\n"
    + "Please comprehensively evaluate all the steps in the solution and provide only True or False as your final output."
)

os.environ["OPENAI_API_KEY"] = "your_api_key_here"


class BaseModel:
    def __init__(self):
        pass

    @abstractmethod
    def get_model_response(self, prompt: str):  # type: ignore
        pass


class OpenAIModel(BaseModel):
    def __init__(
        self,
        model: str,
        temperature: float,
        max_tokens: int,
    ):
        super().__init__()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = OpenAI()

    def get_model_response(self, prompt: str):  # type: ignore
        max_num_try = 3
        for i in range(max_num_try):
            try:
                content = [{"type": "text", "text": prompt}]
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": content}],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    n=1,
                )
                return response.choices[0].message.content
            except RateLimitError as e:
                if i == max_num_try - 1:
                    raise e
                print(
                    "Hit OpenAI rate limit with {} tries, sleep for a while and try again".format(
                        i + 1
                    )
                )
                time.sleep(30)
            except Exception as e:
                if i == max_num_try - 1:
                    raise e
                print("Failed to call OpenAI API: {}".format(e))
                time.sleep(30)


@dataclass
class ScriptArguments:
    check_path: Optional[str] = field(
        default="/mnt/blob/Mirage/search/Llama-results/100test/math100/llama3.1_8B/BoN_Reward_correct_256samples-Skywork-o1-Open-PRM-Qwen-2.5-7B.jsonl",
        metadata={"help": "the location of the test set"},
    )
    output_path: Optional[str] = field(
        default="/mnt/blob/Inference-Scaling/sample/Llama-results/math500/llama3.2_3B",
        metadata={"help": "the location of the checked result"},
    )
    temperature: Optional[float] = field(
        default=0.8,
        metadata={"help": "temperature for the model"},
    )
    max_tokens: Optional[int] = field(
        default=4096,
        metadata={"help": "max tokens for the model"},
    )
    save_batch_size: Optional[int] = field(
        default=20,
        metadata={"help": "batch size for saving the results"},
    )


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    model = OpenAIModel(
        "gpt-4o",
        temperature=script_args.temperature,
        max_tokens=script_args.max_tokens,
    )
    df = pd.read_json(script_args.check_path, lines=True)
    final_prompt = [
        prompt.format(problem=df.iloc[i]["problem"], solution=df.iloc[i]["solution"])
        for i in range(len(df))
    ]
    if script_args.save_batch_size != -1:
        for i in range(0, len(final_prompt), script_args.save_batch_size):
            print(
                "Processing batch {} to {}".format(i, i + script_args.save_batch_size)
            )
            response = []
            for p in tqdm(final_prompt[i : i + script_args.save_batch_size]):
                res = model.get_model_response(p)
                response.append(res)
            problem_id = [
                df.iloc[i]["id"]
                for i in range(i, min(i + script_args.save_batch_size, len(df)))
            ]
            for p_id, res in zip(problem_id, response):
                tmp_data = {"id": str(p_id), "response": res}
                with open(script_args.output_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(tmp_data, ensure_ascii=False) + "\n")
