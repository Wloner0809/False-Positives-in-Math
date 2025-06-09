import signal
from typing import Optional

from envs.base_env import CoTEnv

from .grader import math_equal

# from .verify_utils import extract_answer as extract_fn, grade_answer
from .parse_utils_qwen import extract_answer as extract_fn
from .parse_utils_qwen import parse_ground_truth
from .prompt import COT_EXAMPLES, COT_TASK_DESC, PROBLEM_FORMAT_STR, SEP

ANS_RE = None
STOP_STR = None


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
def math_equal_timeout(prediction, reference):
    return math_equal(prediction, reference)


def extract_answer(answer_str: str) -> str:
    return extract_fn(answer_str, data_name="math")


def extract_groundtruth(groundtruth_str: str) -> str:
    return parse_ground_truth(groundtruth_str, data_name="math")


def judge_correct(
    problem_str: str, extracted_groundtruth: Optional[str], answer: str
) -> bool:
    # return grade_answer(given_answer=answer, ground_truth=extracted_groundtruth)
    result = math_equal_timeout(answer, extracted_groundtruth)
    return result


class Env(CoTEnv):
    sep = SEP

    def __init__(
        self,
        config,
        math_problems,
        llm_gen_fn,
        task_desc_str: str = COT_TASK_DESC,
        cot_example_str: str = COT_EXAMPLES,
        problem_format_str: str = PROBLEM_FORMAT_STR,
        reset=True,
    ):
        super().__init__(
            config,
            math_problems,
            llm_gen_fn,
            task_desc_str,
            cot_example_str,
            problem_format_str,
            reset,
        )

    @property
    def stop_str(self):
        return STOP_STR

    def post_process_act(self, action: str):
        if not action.endswith(self.sep):
            action = action.strip() + self.sep

        return action

    def _is_correct(self, completion):
        extracted_answer = extract_answer(completion)
        # print("Compare: {} -- {}".format(extrated_answer, self.math_problem['answer']))
        # return extrated_answer == self.math_problem['answer']
        return judge_correct(
            self.math_problem["question"], self.math_problem["answer"], extracted_answer
        )

    def get_reward(self):
        """To implement based on learned reward model"""
        return 0
