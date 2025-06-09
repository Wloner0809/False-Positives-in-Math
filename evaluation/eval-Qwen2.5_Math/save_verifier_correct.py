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
        default="/mnt/blob/Inference-Scaling/sample/Llama/llama3.2_3B_Instruct-math500_256answers_per_question.jsonl",
        metadata={"help": "the location of the test set"},
    )
    benchmark_name: Optional[str] = field(
        default="qq8933/MATH500",
        metadata={"help": "the name of the benchmark"},
    )
    prm_reward_path: Optional[str] = field(
        default="/mnt/blob/Inference-Scaling/sample/Llama-with-scores/Skywork-o1-Open-PRM-Qwen-2.5-7B/llama3.2_3B_Instruct-math500_256answers_per_question-with_scores.jsonl",
        metadata={"help": "the location of the prm reward file"},
    )
    output_path: Optional[str] = field(
        default="/mnt/blob/Inference-Scaling/sample/Llama-results/math500/llama3.2_3B",
        metadata={"help": "the location of the checked result"},
    )
    greedy_decode_path: Optional[str] = field(
        default=None,
        metadata={"help": "the location of the greedy decode file"},
    )


def get_ground_truth(dataset_name, test_size=None):
    if dataset_name == "openai/gsm8k":
        data_name = "gsm8k"
        dataset = load_dataset(dataset_name, "main", split="test")
        problem_set = dataset["question"]
    elif dataset_name == "qq8933/MATH500":
        data_name = "math"
        dataset = load_dataset(dataset_name, split="test")
        problem_set = dataset["problem"]
    elif dataset_name == "AI-MO/aimo-validation-aime":
        data_name = (
            "aime24"  #! the dataset actually contains aime22, aime23, and aime24
        )
        dataset = load_dataset(dataset_name, split="train")
        problem_set = dataset["problem"]
    elif "Omni-MATH500" in dataset_name:
        # TODO: need to fix ground truth for Omni-MATH500
        dataset = load_dataset("json", data_files=dataset_name, split="train")
        problem_set = dataset["problem"]
        ground_truth = []
        for data in dataset:
            ground_truth.append(data["answer"])
        return ground_truth, problem_set
    elif "OmniMATH100-rule" in dataset_name:
        dataset = load_dataset("json", data_files=dataset_name, split="train")
        problem_set = dataset["problem"]
        ground_truth = []
        for data in dataset:
            ground_truth.append(data["answer"])
        return ground_truth, problem_set
    elif "MATH100" in dataset_name:
        data_name = "math"
        dataset = load_dataset("json", data_files=dataset_name, split="train")
        problem_set = dataset["problem"]
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
    return ground_truth, problem_set


def extract_answer_from_generated_text(data_path: str, dataset_name=None):
    dataset = load_dataset("json", data_files=data_path)["train"]
    solutions = []
    unextracted_solutions = []
    if dataset_name == "openai/gsm8k":
        data_name = "gsm8k"
    elif dataset_name == "qq8933/MATH500":
        data_name = "math"
    elif dataset_name == "AI-MO/aimo-validation-aime":
        data_name = "aime24"
    elif "Omni-MATH500" in dataset_name:
        data_name = "Omni-MATH500"
    elif "OmniMATH100-rule" in dataset_name:
        data_name = "OmniMATH100-rule"
    elif "MATH100" in dataset_name:
        data_name = "math"
    for example in dataset:
        solution = []
        unextracted_solution = []
        responses = example["responses"]
        for response in responses:
            pred_str = response
            unextracted_solution.append(pred_str)
            pred = extract_prediction(result=pred_str, data_name=data_name)
            if data_name == "aime24":

                def extract_number(s):
                    s = re.sub(r"\\[a-zA-Z]+\{", "", s)
                    s = re.sub(r"[^0-9\-]", "", s)
                    return s

                pred = extract_number(pred)
            solution.append(pred)
        solutions.append(solution)
        unextracted_solutions.append(unextracted_solution)
    return solutions, unextracted_solutions


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


# def check_pass_at_n(args):
#     i, ground_truth_i, solutions_i = args
#     for solution in solutions_i:
#         if math_equal_process_timeout(solution, ground_truth_i):
#             return 1, i
#     return 0, i


# def check_self_consistency(args):
#     i, ground_truth_i, solutions_i = args
#     counts = Counter(solutions_i)
#     top_one = counts.most_common(1)[0][0]
#     if math_equal_process_timeout(top_one, ground_truth_i):
#         return 1, i
#     return 0, i


def check_best_of_n_prm_reward(args):
    i, ground_truth_i, solutions_i, post_process_rewards_i = args
    max_index = post_process_rewards_i.index(max(post_process_rewards_i))
    if math_equal_process_timeout(solutions_i[max_index], ground_truth_i):
        return 1, i, max_index
    return 0, i, max_index


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
    prm_reward_max_index = [
        k for k in range(len(solutions_i)) if solutions_i[k] == top_one
    ]
    prm_reward_max_index = post_process_rewards_i.index(
        max([post_process_rewards_i[k] for k in prm_reward_max_index])
    )
    if math_equal_process_timeout(top_one, ground_truth_i):
        return 1, i, prm_reward_max_index
    return 0, i, prm_reward_max_index


def save_verifier_correct(
    problem_set,
    unextracted_solutions,
    solutions,
    ground_truth,
    prm_reward_path=None,
    result_path=None,
    sample_size=None,
):
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

    if sample_size is not None and type(sample_size) is list:
        solutions_sample_size = []
        unextracted_solutions_sample_size = []
        post_process_rewards_sample_size = []
        for size in sample_size:
            single_solutions_sample_size = []
            single_unextracted_solutions_sample_size = []
            single_post_process_rewards_sample_size = []
            for i in range(len(solutions)):
                single_solutions_sample_size.append(solutions[i][:size])
            for i in range(len(unextracted_solutions)):
                single_unextracted_solutions_sample_size.append(
                    unextracted_solutions[i][:size]
                )
            for i in range(len(post_process_rewards)):
                single_post_process_rewards_sample_size.append(
                    post_process_rewards[i][:size]
                )
            solutions_sample_size.append(single_solutions_sample_size)
            unextracted_solutions_sample_size.append(
                single_unextracted_solutions_sample_size
            )
            post_process_rewards_sample_size.append(
                single_post_process_rewards_sample_size
            )

    # # Self-Consistency
    # self_consistency_sample_size = []
    # for sample_size_idx in range(len(solutions_sample_size)):
    #     with ProcessPoolExecutor() as executor:
    #         results = list(
    #             executor.map(
    #                 check_self_consistency,
    #                 [
    #                     (i, ground_truth[i], solutions_sample_size[sample_size_idx][i])
    #                     for i in range(len(ground_truth))
    #                 ],
    #             )
    #         )
    #     single_self_consistency_sample_size = []
    #     for result in results:
    #         if result[0] == 1:
    #             single_self_consistency_sample_size.append(result[1])
    #     single_self_consistency_sample_size = sorted(
    #         single_self_consistency_sample_size
    #     )
    #     self_consistency_sample_size.append(single_self_consistency_sample_size)

    if "Qwen2.5-MATH-RM-72B" in prm_reward_path:
        for sample_size_idx in range(len(post_process_rewards_sample_size)):
            for i in range(len(post_process_rewards_sample_size[sample_size_idx])):
                post_process_rewards_sample_size[sample_size_idx][i] = (
                    get_true_positive_weight(
                        post_process_rewards_sample_size[sample_size_idx][i]
                    )
                )
    # Best-of-N with prm reward
    best_of_n_prm_reward_sample_size = []
    for sample_size_idx in range(len(solutions_sample_size)):
        with ProcessPoolExecutor() as executor:
            results = list(
                executor.map(
                    check_best_of_n_prm_reward,
                    [
                        (
                            i,
                            ground_truth[i],
                            solutions_sample_size[sample_size_idx][i],
                            post_process_rewards_sample_size[sample_size_idx][i],
                        )
                        for i in range(len(ground_truth))
                    ],
                )
            )
        single_best_of_n_prm_reward_sample_size = []
        for result in results:
            if result[0] == 1:
                single_best_of_n_prm_reward_sample_size.append(
                    (result[1], result[2])
                )  # * (result[1], result[2]) is the problem id and the index of the best solution
        single_best_of_n_prm_reward_sample_size = sorted(
            single_best_of_n_prm_reward_sample_size, key=lambda x: x[0]
        )
        best_of_n_prm_reward_sample_size.append(single_best_of_n_prm_reward_sample_size)

    # Self-Consistency with prm reward weight
    self_consistency_prm_reward_sample_size = []
    for sample_size_idx in range(len(solutions_sample_size)):
        with ProcessPoolExecutor() as executor:
            results = list(
                executor.map(
                    check_self_consistency_prm_reward,
                    [
                        (
                            i,
                            ground_truth[i],
                            solutions_sample_size[sample_size_idx][i],
                            post_process_rewards_sample_size[sample_size_idx][i],
                        )
                        for i in range(len(ground_truth))
                    ],
                )
            )
        single_self_consistency_prm_reward_sample_size = []
        for result in results:
            if result[0] == 1:
                single_self_consistency_prm_reward_sample_size.append(
                    (result[1], result[2])
                )  # * (result[1], result[2]) is the problem id and the index of the best solution
        single_self_consistency_prm_reward_sample_size = sorted(
            single_self_consistency_prm_reward_sample_size, key=lambda x: x[0]
        )
        self_consistency_prm_reward_sample_size.append(
            single_self_consistency_prm_reward_sample_size
        )

    #! BoN+Reward and SC+Reward
    for sample_size_idx in range(len(sample_size)):
        BoN_Reward_correct_problem_id = [
            idx for idx, _ in best_of_n_prm_reward_sample_size[sample_size_idx]
        ]
        df_BoN_Reward_correct = pd.DataFrame(
            {
                "id": BoN_Reward_correct_problem_id,
                "problem": [problem_set[i] for i in BoN_Reward_correct_problem_id],
                "ground_truth": [
                    ground_truth[i] for i in BoN_Reward_correct_problem_id
                ],
                "solution": [
                    unextracted_solutions_sample_size[sample_size_idx][problem_id][idx]
                    for problem_id, idx in best_of_n_prm_reward_sample_size[
                        sample_size_idx
                    ]
                ],
            }
        )
        SC_Reward_correct_problem_id = [
            idx for idx, _ in self_consistency_prm_reward_sample_size[sample_size_idx]
        ]
        df_SC_Reward_correct = pd.DataFrame(
            {
                "id": SC_Reward_correct_problem_id,
                "problem": [problem_set[i] for i in SC_Reward_correct_problem_id],
                "ground_truth": [ground_truth[i] for i in SC_Reward_correct_problem_id],
                "solution": [
                    unextracted_solutions_sample_size[sample_size_idx][problem_id][idx]
                    for problem_id, idx in self_consistency_prm_reward_sample_size[
                        sample_size_idx
                    ]
                ],
            }
        )
        if result_path is not None:
            if "Skywork-o1-Open-PRM-Qwen-2.5-7B" in prm_reward_path:
                df_BoN_Reward_correct.to_json(
                    f"{result_path}/BoN_Reward_correct_{sample_size[sample_size_idx]}samples-Skywork-o1-Open-PRM-Qwen-2.5-7B.jsonl",
                    orient="records",
                    lines=True,
                )
                df_SC_Reward_correct.to_json(
                    f"{result_path}/SC_Reward_correct_{sample_size[sample_size_idx]}samples-Skywork-o1-Open-PRM-Qwen-2.5-7B.jsonl",
                    orient="records",
                    lines=True,
                )
            elif "Qwen2.5-MATH-RM-72B" in prm_reward_path:
                df_BoN_Reward_correct.to_json(
                    f"{result_path}/BoN_Reward_correct_{sample_size[sample_size_idx]}samples-Qwen2.5-MATH-RM-72B.jsonl",
                    orient="records",
                    lines=True,
                )
                df_SC_Reward_correct.to_json(
                    f"{result_path}/SC_Reward_correct_{sample_size[sample_size_idx]}samples-Qwen2.5-MATH-RM-72B.jsonl",
                    orient="records",
                    lines=True,
                )
            elif "Skywork-PRM-vllm0.6.4.post1" in prm_reward_path:
                df_BoN_Reward_correct.to_json(
                    f"{result_path}/BoN_Reward_correct_{sample_size[sample_size_idx]}samples-Skywork-o1-Open-PRM-Qwen-2.5-7B.jsonl",
                    orient="records",
                    lines=True,
                )
                df_SC_Reward_correct.to_json(
                    f"{result_path}/SC_Reward_correct_{sample_size[sample_size_idx]}samples-Skywork-o1-Open-PRM-Qwen-2.5-7B.jsonl",
                    orient="records",
                    lines=True,
                )


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    test_set_path = script_args.test_set_path
    benchmark_name = script_args.benchmark_name
    prm_reward_path = script_args.prm_reward_path
    result_path = script_args.output_path

    ground_truth, problem_set = get_ground_truth(benchmark_name)
    solutions, unextracted_solutions = extract_answer_from_generated_text(
        test_set_path, benchmark_name
    )

    save_verifier_correct(
        problem_set=problem_set,
        unextracted_solutions=unextracted_solutions,
        solutions=solutions,
        ground_truth=ground_truth,
        prm_reward_path=prm_reward_path,
        result_path=result_path,
        sample_size=[4, 16, 64, 256],
    )
