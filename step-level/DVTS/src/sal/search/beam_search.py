#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy
import logging
from collections import defaultdict

import numpy as np
from openai import OpenAI
from skywork_prm.model_utils.io_utils import derive_step_rewards_vllm, prepare_input
from tqdm import tqdm
from transformers import AutoTokenizer

from sal.config import Config
from sal.llm_caller.policy_model_caller import LMCallingConfig, VLLMRemoteCaller
from sal.utils.score import aggregate_scores

from .utils import Beam, build_conv, generate_k_steps

logger = logging.getLogger()


def skywork_score(questions, outputs, model, client, prm_tokenizer, max_tokens=4096):
    all_scores = []
    test_list = []
    for question, answers in zip(questions, outputs, strict=True):
        single_test_list = []
        for answer in answers:
            single_test_list.append(
                prepare_input(
                    problem=question,
                    response=answer,
                    tokenizer=prm_tokenizer,
                    step_token="\n\n",
                    max_tokens=max_tokens,
                )
            )
        test_list.append(single_test_list)
    for test_data in tqdm(test_list, desc="PRM scoring", total=len(test_list)):
        input_ids, steps, reward_flags = zip(*test_data)
        rewards = client.embeddings.create(
            input=input_ids,
            model=model,
        )
        step_rewards = derive_step_rewards_vllm(rewards, reward_flags)
        all_scores.append(step_rewards)
    return all_scores


def _beam_search(
    batch_of_prompts,
    config: Config,
    tokenizer: AutoTokenizer,
    prm_tokenizer: AutoTokenizer,
    policy_model_caller: VLLMRemoteCaller,
) -> list[Beam]:
    sampling_params = LMCallingConfig(
        n=1,
        temperature=config.temperature,
        top_p=config.top_p,
        max_new_tokens=config.max_tokens,
        stop_str=[
            "\n\n"
        ],  # we consider that a step in the problem is indicated by a double newline
        include_stop_str_in_output=True,
    )

    beams: list[Beam] = []
    for prompt in batch_of_prompts:
        for i in range(config.n):
            beams.append(
                Beam(
                    prompt=prompt,
                    index=i,
                    current_text="",
                    next_texts=None,
                    lookahead_texts=None,
                    pruned=False,
                    completed=False,  # New flag to track completion
                    stop_reasons=None,
                    history=[],
                    best_scores=[],
                    all_scores=[],
                    previous_text=None,
                    completion_tokens=0,
                )
            )

    # --------------- call vllm(prm) server -------------
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8081/v1"
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    models = client.models.list()
    skywork_prm = models.data[0].id
    # ---------------------------------------------------

    completed_beams: list[Beam] = []

    for i in tqdm(range(config.num_iterations), desc="Beam search iterations"):
        if i == 0:
            active_beams = [b for b in beams if not b.pruned]
        else:
            active_beams = [b for b in active_beams if not b.pruned]

        # Duplicate active beams to ensure that we have config.n beams per iteration
        if len(active_beams) != config.n:
            repeats = (config.n // len(active_beams)) + 1
            logger.debug(
                f"Extending active_beams with {repeats} repetitions to reach size {config.n}"
            )
            extended_active_beams = [
                copy.deepcopy(b) for b in (active_beams * repeats)[: config.n]
            ]
            active_beams = extended_active_beams
            if len(active_beams) != config.n:
                raise ValueError(
                    f"Expected {config.n} active beams, but got {len(active_beams)}"
                )

        if i == config.num_iterations - 1:
            # Last iteration, generate to EOS
            sampling_params = LMCallingConfig(
                n=1,
                temperature=config.temperature,
                top_p=config.top_p,
                max_new_tokens=config.max_tokens,
            )

        convs = [
            build_conv(b.prompt, b.current_text, config.system_prompt)
            for b in active_beams
        ]
        continue_final_message = i > 0
        add_generation_prompt = i == 0

        if config.custom_chat_template is not None:
            tokenizer.chat_template = config.custom_chat_template
        templated_convs = tokenizer.apply_chat_template(
            convs,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            tokenize=False,
        )
        lookahead = 0 if i == config.num_iterations - 1 else config.lookahead
        gen_results = generate_k_steps(
            templated_convs=templated_convs,
            lookahead_steps=lookahead,
            policy_model_caller=policy_model_caller,
            lm_calling_config=sampling_params,
            beam_width=1,
        )

        prompts, completions = [], []
        for beam, gen_result in zip(active_beams, gen_results, strict=True):
            beam.next_texts = gen_result.next_texts
            beam.stop_reasons = gen_result.stop_reasons
            beam.lookahead_texts = gen_result.lookahead_texts
            beam.completion_tokens += gen_result.completion_tokens
            beam.current_text += beam.next_texts[0]
            beam.history.append(beam.next_texts[0])

            if (
                beam.stop_reasons[0] == "EOS"
                or beam.stop_reasons[0] == "length"
                or beam.next_texts[0] == ""
            ):
                beam.completed = True
                completed_beams.append(beam)
            prompts.append(beam.prompt)
            completions.append([beam.current_text])

        scores = skywork_score(
            questions=prompts,
            outputs=completions,
            model=skywork_prm,
            client=client,
            prm_tokenizer=prm_tokenizer,
        )

        agg_scores = [
            [aggregate_scores(s, config.agg_strategy) for s in score]
            for score in scores
        ]

        for beam, score in zip(active_beams, scores, strict=True):
            beam.all_scores = score[0]

        # Now filter active_beams and agg_scores for beams that are completed
        agg_scores = [
            agg_scores[i] for i, b in enumerate(active_beams) if not b.completed
        ]
        active_beams = [b for b in active_beams if not b.completed]

        # Early stopping if all beams are completed
        if len(active_beams) == 0:
            break

        # Filter duplicate active beams
        if config.filter_duplicates:
            # Create a dictionary to filter duplicates and retain order
            unique_beam_dict = {}
            for i, b in enumerate(active_beams):
                if b.current_text not in unique_beam_dict:
                    unique_beam_dict[b.current_text] = (
                        i  # Map the unique text to its index
                    )
            active_beams = [active_beams[i] for i in unique_beam_dict.values()]
            agg_scores = [agg_scores[i] for i in unique_beam_dict.values()]

        # Get indices for top (config.n / config.beam_width) completions
        top_indices = np.argsort(np.array(agg_scores).flatten())[
            -(config.n // config.beam_width) :
        ]

        for idx, beam in enumerate(active_beams):
            if idx not in top_indices:
                beam.pruned = True

    # Filter completed beams for those with top config.n scores
    if config.sort_completed:
        completed_beams = sorted(
            completed_beams,
            key=lambda b: aggregate_scores(b.all_scores, config.agg_strategy),
            reverse=True,
        )[: config.n]
    else:
        completed_beams = completed_beams[: config.n]

    if len(completed_beams) != config.n:
        # If we don't have enough completed_beams, duplicate until we reach config.n
        repeats = (config.n // len(completed_beams)) + 1
        logger.debug(
            f"Extending completed_beams with {repeats} repetitions to reach size {config.n}"
        )
        extended_completed_beams = [
            copy.deepcopy(b) for b in (completed_beams * repeats)[: config.n]
        ]
        completed_beams = extended_completed_beams

    return completed_beams


def beam_search(
    examples,
    config: Config,
    tokenizer: AutoTokenizer,
    prm_tokenizer: AutoTokenizer,
    policy_model_caller: VLLMRemoteCaller,
):
    problems = examples["problem"]
    if not isinstance(problems, list):
        problems = [problems]
    solutions = None
    if "solution" in examples:
        solutions = examples["solution"]
        if not isinstance(solutions, list):
            solutions = [solutions]
    beam_results = _beam_search(
        batch_of_prompts=problems,
        config=config,
        tokenizer=tokenizer,
        prm_tokenizer=prm_tokenizer,
        policy_model_caller=policy_model_caller,
    )

    # Group together alike beams and store in the dataset
    grouped_results = defaultdict(list)
    for results in beam_results:
        grouped_results[results.prompt].append(results)

    results = {
        "problem": [],
        "solution": [],
        "completions": [],
        "pred": [],
        "completion_tokens": [],
        "scores": [],
    }

    for index, p in enumerate(problems):
        beams = grouped_results[p]
        completions = [b.current_text for b in beams]
        agg_scores = [
            aggregate_scores(b.all_scores, config.agg_strategy) for b in beams
        ]
        pred = completions[np.argmax(agg_scores)]
        if solutions is not None:
            results["solution"].append(solutions[index])
        results["problem"].append(p)
        results["completions"].append(completions)
        results["scores"].append([b.all_scores for b in beams])
        results["pred"].append(pred)
        results["completion_tokens"].append([b.completion_tokens for b in beams])

    return results
