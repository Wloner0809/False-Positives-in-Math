from dataclasses import dataclass
from typing import Dict

from reason.evaluation.evaluator import Task, TreeSearchSolutionOutput
from reason.guided_search.tree import SearchTree
from reason.inference.lm_call import LanguageModelCallingFunction, LMCallingConfig
from transformers import AutoTokenizer


@dataclass
class BasicConfig:
    task_name: str


@dataclass
class TreeSearchConfig(BasicConfig):
    # construction config
    tree_max_width: int = 10
    tree_max_depth: int = 10
    # node config
    init_critic_value: bool = True

    def __post_init__(self):
        assert self.tree_max_width > 0, "Tree width must be greater than 0"
        assert self.tree_max_depth > 0, "Tree depth must be greater than 0"


@dataclass
class MCTSBaseConfig(TreeSearchConfig):
    # PUCT hparams
    pb_c_base: float = 19652
    pb_c_init: float = 1.25


@dataclass
class VanilaMCTSConfig(MCTSBaseConfig):
    # rollout step strategy, if `select_by_prior` is False,
    #  then select by the initial critic value
    # otherwise, random choice by the prior probability
    select_by_prior: bool = False
    num_path: int = 1

    def __post_init__(self):
        super().__post_init__()
        if not self.select_by_prior:
            assert self.init_critic_value, "VanilaMCTS with greedy as rollout method should set init_critic_value to True"
        assert self.num_path > 0


def vanila_mcts(
    config: VanilaMCTSConfig,
    gen_config: LMCallingConfig,
    problem_inst: Dict[str, str],
    lm_call: LanguageModelCallingFunction,
    prm_tokenizer: AutoTokenizer,
):
    task = Task(task_name=config.task_name)
    env = task.env_fn(
        config={
            "max_actions": config.tree_max_width,
            "max_length": config.tree_max_depth,
            "stop_str": "The answer is ",
            "generation_config": {
                "max_new_tokens": gen_config.max_new_tokens,
                "temperature": gen_config.temperature,
                "top_p": gen_config.top_p,
                "top_k": gen_config.top_k,
            },
        },
        math_problems=[
            {
                "question": problem_inst["question"],
                "answer": task.extract_groundtruth(problem_inst["answer"]),
            }
        ],
        llm_gen_fn=lm_call,
    )

    search_tree = SearchTree(
        cfg={
            "pb_c_base": config.pb_c_base,
            "pb_c_init": config.pb_c_init,
            "init_critic_value": config.init_critic_value,
        }
    )
    traj_list = search_tree.vanila_mcts(
        simulate_env=env,
        num_path=config.num_path,
        prm_tokenizer=prm_tokenizer,
        select_by_prior=config.select_by_prior,
    )
    return TreeSearchSolutionOutput(
        solutions=[t["text"] for t in traj_list],
        completion_tokens=[t["api_completion_tokens"] for t in traj_list],
        tree_completion_tokens=[t["tree_completion_tokens"] for t in traj_list],
    )
