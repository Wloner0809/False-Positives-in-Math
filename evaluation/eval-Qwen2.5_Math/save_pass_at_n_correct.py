import re
import signal
from dataclasses import dataclass, field
from parser import extract_answer, extract_prediction, parse_ground_truth
from typing import Optional

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
    output_path: Optional[str] = field(
        default="/mnt/blob/Inference-Scaling/sample/Llama-results/math500/llama3.2_3B",
        metadata={"help": "the location of the checked result"},
    )
    N: Optional[int] = field(
        default=256,
        metadata={"help": "the number of answers generated for each question"},
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


def check_pass_at_n_results(
    problem_set,
    unextracted_solutions,
    solutions,
    ground_truth,
    N,
    pass_at_n_result_path=None,
):
    score_pass_at_sample_size = 0
    correct = []
    correct_problem = []
    correct_ground_truth = []
    correct_solutions = []
    for i in range(len(ground_truth)):
        for j in range(N):
            if math_equal_process_timeout(solutions[i][j], ground_truth[i]):
                score_pass_at_sample_size += 1
                correct.append(i)
                break
    print(f"Pass@{N}: {score_pass_at_sample_size / len(ground_truth)}")
    print("-" * 50)

    correct = sorted(correct)
    for index in correct:
        single_solutions = []
        for j in range(N):
            if math_equal_process_timeout(solutions[index][j], ground_truth[index]):
                single_solutions.append(unextracted_solutions[index][j])
        correct_problem.append(problem_set[index])
        correct_ground_truth.append(ground_truth[index])
        correct_solutions.append(single_solutions)
    df = pd.DataFrame(
        {
            "id": correct,
            "problem": correct_problem,
            "ground_truth": correct_ground_truth,
            "solution": correct_solutions,
        }
    )
    if pass_at_n_result_path is not None:
        df.to_json(pass_at_n_result_path, orient="records", lines=True)


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    test_set_path = script_args.test_set_path
    benchmark_name = script_args.benchmark_name
    result_path = script_args.output_path

    ground_truth, problem_set = get_ground_truth(benchmark_name)
    solutions, unextracted_solutions = extract_answer_from_generated_text(
        test_set_path, benchmark_name
    )

    check_pass_at_n_results(
        problem_set=problem_set,
        unextracted_solutions=unextracted_solutions,
        solutions=solutions,
        ground_truth=ground_truth,
        N=script_args.N,
        pass_at_n_result_path=result_path,
    )
