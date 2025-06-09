import importlib
from dataclasses import dataclass
from typing import Callable, Dict, List, Union

import ray
from envs import get_default_query_str_builder, get_env_datasets
from reason.inference.lm_call import LanguageModelCallingFunction
from transformers import AutoTokenizer


class Task:
    def __init__(self, task_name: str, is_few_shot: bool = False):
        self.task_name = task_name
        task_module = importlib.import_module(f"envs.{task_name}")
        if task_name == "MATH" or task_name == "AIME" or task_name == "OmniMATH":
            self.extract_answer = task_module.extract_answer
            self.extract_groundtruth = task_module.extract_groundtruth
            self.judge_correct = task_module.judge_correct
        else:
            raise NotImplementedError(f"Task {task_name} is not supported")

        self._is_few_shot = is_few_shot
        self.env_fn = task_module.Env

    def prompt_fn(self, problem_input: str):
        return get_default_query_str_builder(self.task_name)(
            problem_input, is_few_shot=self._is_few_shot
        )

    @property
    def test_ds(self):
        return get_env_datasets(self.task_name)


@dataclass
class SolutionOutput:
    solutions: List[str]
    # Define the completion tokens for each solution
    # For best_of_n, it's a list of int, indicate how many tokens in each generation
    # For beam search, it's a list of zeros, except the last element indicates total tokens
    # For mcts, it's a list of int, indicate how many tokens comsumed between two paths
    completion_tokens: List[int]


@dataclass
class TreeSearchSolutionOutput(SolutionOutput):
    tree_completion_tokens: List[int]


class MathEvaluator:
    def __init__(
        self,
        task: Union[str, Task],
        lm_call: LanguageModelCallingFunction,
        prm_tokenizer: AutoTokenizer,
    ):
        if isinstance(task, str):
            self._task = Task(task_name=task)
        else:
            assert isinstance(task, Task)
            self._task = task
        self.lm_call = lm_call
        self.prm_tokenizer = prm_tokenizer

    def generate_solutions(
        self, problem_inst: Dict[str, str], solver_fn: Callable
    ) -> List[str]:
        solution: SolutionOutput = solver_fn(
            problem_inst, self.lm_call, self.prm_tokenizer
        )
        return problem_inst, solution.solutions


@ray.remote
class RemoteMathEvaluator(MathEvaluator):
    def __init__(
        self,
        task: str,
        lm_call: LanguageModelCallingFunction,
        prm_tokenizer: AutoTokenizer,
    ):
        super().__init__(task, lm_call, prm_tokenizer)
