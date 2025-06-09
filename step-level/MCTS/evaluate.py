import json
import os
import random
from argparse import ArgumentParser
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import ray
import torch
from ray.util.actor_pool import ActorPool
from reason.evaluation.evaluator import RemoteMathEvaluator, Task
from reason.evaluation.methods import (
    VanilaMCTSConfig,
    vanila_mcts,
)
from reason.inference.lm_call import LMCallingConfig, VLLMRemoteCaller
from tqdm import tqdm
from transformers import AutoTokenizer


def str2bool(x: str):
    if x == "False":
        return False
    elif x == "True":
        return True
    else:
        raise ValueError(
            'you should either input "True" or "False" but not {}'.format(x)
        )


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--LM", type=str, required=True)
    parser.add_argument("--prm_tokenizer_path", type=str, required=True)
    parser.add_argument("--controller_addr", type=str, default="http://0.0.0.0:28778")
    # task config
    parser.add_argument("--task_name", type=str, default="gsm8k")
    parser.add_argument("--test", type=str2bool, default=True)
    parser.add_argument("--is_few_shot", type=str2bool, default=False)
    parser.add_argument("--seed", type=int, default=42)
    # method config
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--num_sequence", type=int, default=1)
    # LM gen config
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    # Tree construction config
    parser.add_argument("--tree_max_depth", type=int, default=None)
    parser.add_argument("--tree_max_width", type=int, default=None)
    # save config
    parser.add_argument("--save_dir", type=str, default=None)
    # parallel config
    parser.add_argument("--local", action="store_true", default=False)
    parser.add_argument("--num_worker", type=int, default=32)
    config = parser.parse_args()

    setup_seed(config.seed)
    if config.local:
        print("run in pure local mode for debug only")
        config.num_worker = 1
        ray.init(local_mode=True)

    llm_gen_fn = VLLMRemoteCaller(config.LM, config.controller_addr, lm_step_tag="\n\n")

    #! skywork reward model
    prm_tokenizer = AutoTokenizer.from_pretrained(config.prm_tokenizer_path)

    task = Task(task_name=config.task_name, is_few_shot=config.is_few_shot)

    def parallel_evaluate_test_dataset(
        solver_fn: Callable, save_dir: Optional[Path] = None
    ) -> List[Dict[str, Any]]:
        test_ds = task.test_ds
        if os.path.exists(save_dir):
            answered_questions = set()
            df = pd.read_json(save_dir, lines=True)
            for i in range(len(df)):
                answered_questions.add(df.iloc[i]["question"])
            test_ds = [
                problem_inst
                for problem_inst in test_ds
                if problem_inst["question"] not in answered_questions
            ]

        actor_pool = ActorPool(
            [
                RemoteMathEvaluator.remote(config.task_name, llm_gen_fn, prm_tokenizer)
                for _ in range(config.num_worker)
            ]
        )
        res_q = actor_pool.map_unordered(
            lambda p, x: p.generate_solutions.remote(x, solver_fn), test_ds
        )
        for i, (problem_inst, solution) in enumerate(tqdm(res_q, total=len(test_ds))):
            obj = {
                "question": problem_inst["question"],
                "ground_truth": problem_inst["answer"],
                "responses": solution,
            }
            with open(save_dir, "a", encoding="utf-8") as f:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    # qwen-2.5 requires add more stop words but not do it now.
    # stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]
    gen_config = LMCallingConfig(
        n=config.num_sequence,
        temperature=config.temperature,
        top_k=config.top_k,
        top_p=config.top_p,
        max_new_tokens=config.max_new_tokens,
    )
    if config.method == "vanila_mcts":
        method_config = VanilaMCTSConfig(
            task_name=config.task_name,
            tree_max_depth=config.tree_max_depth,
            tree_max_width=config.tree_max_width,
            select_by_prior=False,
            num_path=config.num_sequence,
        )
        solver_fn = partial(vanila_mcts, method_config, gen_config)
    else:
        raise ValueError(f"Unknown method: {config.method}")

    parallel_evaluate_test_dataset(solver_fn, config.save_dir)
