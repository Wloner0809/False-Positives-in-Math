import re
import signal
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from parser import extract_answer, extract_prediction, parse_ground_truth
from typing import Optional

import numpy as np
import pandas as pd
from datasets import load_dataset
from grader import math_equal_process
from transformers import HfArgumentParser

# import debugpy

# debugpy.listen(("localhost", 9501))
# debugpy.wait_for_client()


@dataclass
class ScriptArguments:
    test_set_path: Optional[str] = field(
        default="/home/v-wangyu1/Verifier/results/data/dataset_prm/llama3.1_8B_Instruct-math500_4answers_per_question_prm.jsonl",
        metadata={"help": "the location of the test set"},
    )
    benchmark_name: Optional[str] = field(
        default="qq8933/MATH500",
        metadata={"help": "the name of the benchmark"},
    )
    sample_size: Optional[int] = field(
        default=None,
        metadata={"help": "the number of samples"},
    )
    prm_reward_path: Optional[str] = field(
        default=None,
        metadata={"help": "the location of the prm reward file"},
    )


def get_ground_truth(dataset_name, test_size=None):
    if dataset_name == "openai/gsm8k":
        data_name = "gsm8k"
        dataset = load_dataset(dataset_name, "main", split="test")
    elif dataset_name == "qq8933/MATH500":
        data_name = "math"
        dataset = load_dataset(dataset_name, split="test")
    elif dataset_name == "AI-MO/aimo-validation-aime":
        data_name = (
            "aime24"  #! the dataset actually contains aime22, aime23, and aime24
        )
        dataset = load_dataset(dataset_name, split="train")
    elif "Omni-MATH500" in dataset_name:
        # TODO: need to fix ground truth for Omni-MATH500
        dataset = load_dataset("json", data_files=dataset_name, split="train")
        ground_truth = []
        for data in dataset:
            ground_truth.append(data["answer"])
        return ground_truth
    if test_size is not None:
        dataset = dataset.select(range(test_size))
    ground_truth = []
    for data in dataset:
        gt_cot, gt_ans = parse_ground_truth(data, data_name)
        if data_name == "math":
            ground_truth.append(gt_ans)
        elif data_name == "gsm8k":
            ground_truth.append(gt_ans)
        elif data_name == "aime24":

            def extract_number(s):
                s = re.sub(r"\\[a-zA-Z]+\{", "", s)
                s = re.sub(r"[^0-9\-]", "", s)
                return s

            extracted = extract_answer(
                pred_str=gt_ans, data_name=data_name, use_last_number=True
            )
            extracted = extract_number(extracted)
            if gt_ans == extracted:
                ground_truth.append(gt_ans)
            else:
                print("gt:", gt_ans)
                print("extracted:", extracted)
                print(f"Problem {data['id']} solution can not be extracted correctly")
    return ground_truth


def extract_answer_from_generated_text(data_path: str, dataset_name=None):
    dataset = load_dataset("json", data_files=data_path)["train"]
    solutions = []
    if dataset_name == "openai/gsm8k":
        data_name = "gsm8k"
    elif dataset_name == "qq8933/MATH500":
        data_name = "math"
    elif dataset_name == "AI-MO/aimo-validation-aime":
        data_name = "aime24"
    elif "Omni-MATH500" in dataset_name:
        data_name = "Omni-MATH500"
    for example in dataset:
        solution = []
        responses = example["responses"]
        for response in responses:
            pred_str = response
            pred = extract_prediction(result=pred_str, data_name=data_name)
            if data_name == "aime24":

                def extract_number(s):
                    s = re.sub(r"\\[a-zA-Z]+\{", "", s)
                    s = re.sub(r"[^0-9\-]", "", s)
                    return s

                pred = extract_number(pred)
            solution.append(pred)
        solutions.append(solution)
    return solutions


def get_true_positive_weight(rewards):
    true_positive_weight = []
    max_reward = max(rewards)
    min_reward = min(rewards)
    if max_reward == min_reward:
        return [max_reward] * len(rewards)
    else:
        for i in range(len(rewards)):
            true_positive_weight.append(
                (rewards[i] - min_reward) / (max_reward - min_reward)
            )
        return true_positive_weight


class TimeoutException(Exception):
    pass


def timeout(seconds):
    def decorator(func):
        def wrapper(*args, **kwargs):
            def handler(signum, frame):
                raise TimeoutException()

            signal.signal(signal.SIGALRM, handler)
            signal.alarm(seconds)
            try:
                return func(*args, **kwargs)
            except TimeoutException:
                return False
            finally:
                signal.alarm(0)

        return wrapper

    return decorator


@timeout(2)
def math_equal_process_timeout(solution, ground_truth):
    return math_equal_process(solution, ground_truth)


def check_pass_at_n(args):
    i, ground_truth_i, solutions_i = args
    for solution in solutions_i:
        if math_equal_process_timeout(solution, ground_truth_i):
            return 1
    return 0


def check_self_consistency(args):
    i, ground_truth_i, solutions_i = args
    counts = Counter(solutions_i)
    top_one = counts.most_common(1)[0][0]
    if math_equal_process_timeout(top_one, ground_truth_i):
        return 1
    return 0


def check_best_of_n_prm_reward(args):
    i, ground_truth_i, solutions_i, post_process_rewards_i = args
    max_index = post_process_rewards_i.index(max(post_process_rewards_i))
    if math_equal_process_timeout(solutions_i[max_index], ground_truth_i):
        return 1
    return 0


def check_self_consistency_prm_reward(args):
    i, ground_truth_i, solutions_i, post_process_rewards_i = args
    counts = Counter(solutions_i)
    all = counts.most_common()
    all_prm_reward = []
    for j in range(len(all)):
        index = [k for k in range(len(solutions_i)) if solutions_i[k] == all[j][0]]
        prm_reward = 0
        for idx in index:
            prm_reward += post_process_rewards_i[idx]
        all_prm_reward.append((all[j][0], prm_reward))
    all_prm_reward = sorted(all_prm_reward, key=lambda x: x[1], reverse=True)
    top_one = all_prm_reward[0][0]
    if math_equal_process_timeout(top_one, ground_truth_i):
        return 1
    return 0


def get_prm_results(solutions, ground_truth, prm_reward_path=None, sample_size=None):
    #! mainly used for Math-Shepherd & PQM & Qwen2.5-Math-RM-72B & Skywork-o1-Open-PRM-Qwen-2.5-7B
    if prm_reward_path is not None:
        if "prm" in prm_reward_path:
            prm_rewards = load_dataset("json", data_files=prm_reward_path)["train"]
            post_process_rewards = []
            for i in range(len(prm_rewards["scores"])):
                single_example_rewards = []
                for score in prm_rewards["scores"][i]:
                    if len(score) == 0:
                        single_example_rewards.append(-float("inf"))
                    else:
                        single_example_rewards.append(
                            min(score)
                        )  #! use the minimum reward across all steps
                post_process_rewards.append(single_example_rewards)
        elif "pqm" in prm_reward_path:
            prm_rewards = load_dataset("json", data_files=prm_reward_path)["train"]
            post_process_rewards = []
            for i in range(len(prm_rewards["rewards"])):
                post_process_rewards.append(prm_rewards["rewards"][i])
        elif "Qwen2.5-MATH-RM-72B" in prm_reward_path:
            prm_rewards = pd.read_json(prm_reward_path, lines=True)
            post_process_rewards = prm_rewards["scores"].tolist()
        elif "Skywork-o1-Open-PRM-Qwen-2.5-7B" in prm_reward_path:
            prm_rewards = pd.read_json(prm_reward_path, lines=True)
            post_process_rewards = []
            for i in range(len(prm_rewards["step_scores"])):
                single_example_rewards = []
                for score in prm_rewards["step_scores"][i]:
                    if len(score) == 0:
                        single_example_rewards.append(-float("inf"))
                    else:
                        single_example_rewards.append(
                            np.mean(score)
                        )  #! use the average reward across all steps
                post_process_rewards.append(single_example_rewards)
        elif "Skywork-PRM-vllm0.6.4.post1" in prm_reward_path:
            prm_rewards = pd.read_json(prm_reward_path, lines=True)
            post_process_rewards = []
            for i in range(len(prm_rewards["scores"])):
                single_example_rewards = []
                for score in prm_rewards["scores"][i]:
                    if len(score) == 0:
                        single_example_rewards.append(-float("inf"))
                    else:
                        single_example_rewards.append(
                            np.mean(score)
                        )  #! use the average reward across all steps
                post_process_rewards.append(single_example_rewards)

    if sample_size is not None:
        for i in range(len(solutions)):
            solutions[i] = solutions[i][:sample_size]
        if prm_reward_path is not None:
            for i in range(len(post_process_rewards)):
                post_process_rewards[i] = post_process_rewards[i][:sample_size]

    if prm_reward_path is None:
        # Pass@sample_size
        score_pass_at_sample_size = 0
        # for i in range(len(ground_truth)):
        #     for j in range(len(solutions[i])):
        #         if math_equal_process(solutions[i][j], ground_truth[i]):
        #             score_pass_at_sample_size += 1
        #             break
        with ProcessPoolExecutor() as executor:
            results = list(
                executor.map(
                    check_pass_at_n,
                    [
                        (i, ground_truth[i], solutions[i])
                        for i in range(len(ground_truth))
                    ],
                )
            )
        score_pass_at_sample_size = sum(results)
        print(
            f"Pass@{len(solutions[0])}: {score_pass_at_sample_size / len(ground_truth)}"
        )
        print("-" * 50)

        # Self-Consistency
        score_self_consistency = 0
        # for i in range(len(ground_truth)):
        #     counts = Counter(solutions[i])
        #     top_one = counts.most_common(1)[0][0]
        #     if math_equal_process(top_one, ground_truth[i]):
        #         score_self_consistency += 1
        with ProcessPoolExecutor() as executor:
            results = list(
                executor.map(
                    check_self_consistency,
                    [
                        (i, ground_truth[i], solutions[i])
                        for i in range(len(ground_truth))
                    ],
                )
            )
        score_self_consistency = sum(results)
        print(f"Self-Consistency: {score_self_consistency / len(ground_truth)}")
        print("-" * 50)

    if prm_reward_path is not None:
        if "Qwen2.5-MATH-RM-72B" in prm_reward_path:
            for i in range(len(post_process_rewards)):
                post_process_rewards[i] = get_true_positive_weight(
                    post_process_rewards[i]
                )
        elif "pqm" in prm_reward_path:
            for i in range(len(post_process_rewards)):
                post_process_rewards[i] = get_true_positive_weight(
                    post_process_rewards[i]
                )
        # Best-of-N with prm reward
        score_best_of_n_prm_reward = 0
        # for i in range(len(ground_truth)):
        #     max_index = post_process_rewards[i].index(max(post_process_rewards[i]))
        #     if math_equal_process(solutions[i][max_index], ground_truth[i]):
        #         score_best_of_n_prm_reward += 1
        with ProcessPoolExecutor() as executor:
            results = list(
                executor.map(
                    check_best_of_n_prm_reward,
                    [
                        (i, ground_truth[i], solutions[i], post_process_rewards[i])
                        for i in range(len(ground_truth))
                    ],
                )
            )
        score_best_of_n_prm_reward = sum(results)
        print(
            f"Best-of-N with prm reward: {score_best_of_n_prm_reward / len(ground_truth)}"
        )
        print("-" * 50)

        # Self-Consistency with prm reward weight
        score_self_consistency_prm_reward = 0
        # for i in range(len(ground_truth)):
        #     counts = Counter(solutions[i])
        #     all = counts.most_common()
        #     all_prm_reward = []
        #     for j in range(len(all)):
        #         index = [
        #             k for k in range(len(solutions[i])) if solutions[i][k] == all[j][0]
        #         ]
        #         prm_reward = 0
        #         for idx in index:
        #             prm_reward += post_process_rewards[i][idx]
        #         all_prm_reward.append((all[j][0], prm_reward))
        #     all_prm_reward = sorted(all_prm_reward, key=lambda x: x[1], reverse=True)
        #     top_one = all_prm_reward[0][0]
        #     if math_equal_process(top_one, ground_truth[i]):
        #         score_self_consistency_prm_reward += 1
        with ProcessPoolExecutor() as executor:
            results = list(
                executor.map(
                    check_self_consistency_prm_reward,
                    [
                        (i, ground_truth[i], solutions[i], post_process_rewards[i])
                        for i in range(len(ground_truth))
                    ],
                )
            )
        score_self_consistency_prm_reward = sum(results)
        print(
            f"Self-Consistency with prm reward weight: {score_self_consistency_prm_reward / len(ground_truth)}"
        )


def get_answer_from_generated_text(data_path: str):
    dataset = load_dataset("json", data_files=data_path)["train"]
    solutions = []
    for example in dataset:
        solution = []
        responses = example["responses"]
        for response in responses:
            pred_str = response
            solution.append(pred_str)
        solutions.append(solution)
    return solutions


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    test_set_path = script_args.test_set_path
    benchmark_name = script_args.benchmark_name
    prm_reward_path = script_args.prm_reward_path

    ground_truth = get_ground_truth(benchmark_name)
    solutions = extract_answer_from_generated_text(test_set_path, benchmark_name)

    get_prm_results(
        solutions=solutions,
        ground_truth=ground_truth,
        prm_reward_path=prm_reward_path,
        sample_size=script_args.sample_size,
    )
