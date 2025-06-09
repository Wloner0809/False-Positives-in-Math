from dataclasses import dataclass
from typing import List

import requests


@dataclass
class ConcatedLMGenResult:
    gen_text: str | List[str]
    stop_reason: str | List[str]
    finish_reason: str | List[str]
    prompt_tokens: List[int]
    num_tokens: List[int]
    cumulative_logprob: List[float]
    logp_avg_by_len: List[float]

    # post init compute number of completion_tokens
    def __post_init__(self):
        self.completion_tokens = sum(self.num_tokens)


def _generate_fastchat(
    gen_prompts,
    model_name,
    n,
    temperature,
    top_p,
    top_k,
    max_new_tokens,
    stop_token_ids,
    stop_str,
    include_stop_str_in_output,
    controller_addr,
) -> ConcatedLMGenResult:
    ret = requests.post(
        controller_addr + "/get_worker_address", json={"model": model_name}
    )
    worker_addr = ret.json()["address"]
    if not worker_addr:
        raise ValueError("Language Model name {} does not exist.".format(model_name))

    headers = {"User-Agent": "FastChat Client"}
    gen_params = {
        "model": model_name,
        "prompt": gen_prompts,
        "n": n,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_new_tokens": max_new_tokens,
        "stop": stop_str,
        "stop_token_ids": stop_token_ids,
        "include_stop_str_in_output": include_stop_str_in_output,
    }
    response = requests.post(
        worker_addr + "/worker_generate",
        headers=headers,
        json=gen_params,
        stream=True,
    )
    results = response.json()
    output_token_lens = results["output_token_len"]
    cum_logps = results["cumulative_logprob"]
    avg_len_logps = [
        clp / max(1, otl) for clp, otl in zip(cum_logps, output_token_lens)
    ]
    return ConcatedLMGenResult(
        gen_text=results["gen_text"],
        stop_reason=results["stop_reason"],
        finish_reason=results["finish_reason"],
        prompt_tokens=results["usage"]["prompt_tokens"],
        num_tokens=results["output_token_len"],
        cumulative_logprob=cum_logps,
        logp_avg_by_len=avg_len_logps,
    )
