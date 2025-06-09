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

import json
import logging
import os

import pandas as pd
import ray
from datasets import Dataset
from ray.util.actor_pool import ActorPool
from sal.config import Config
from sal.llm_caller.policy_model_caller import VLLMRemoteCaller
from sal.search import beam_search, dvts
from sal.utils.data import get_dataset
from sal.utils.parser import H4ArgumentParser
from tqdm import tqdm
from transformers import AutoTokenizer

# import debugpy

# debugpy.listen(("localhost", 9501))
# debugpy.wait_for_client()


logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


APPROACHES = {
    "beam_search": beam_search,
    "dvts": dvts,
}


@ray.remote
class Actor:
    def __init__(self, policy_model_caller, tokenizer, prm_tokenizer):
        self.policy_model_caller = policy_model_caller
        self.tokenizer = tokenizer
        self.prm_tokenizer = prm_tokenizer

    def approach_fn(
        self,
        examples,
        config: Config,
    ):
        return APPROACHES[config.approach](
            examples,
            config,
            tokenizer=self.tokenizer,
            prm_tokenizer=self.prm_tokenizer,
            policy_model_caller=self.policy_model_caller,
        )


def main():
    parser = H4ArgumentParser(Config)
    config = parser.parse()

    policy_model_caller = VLLMRemoteCaller(
        model_name=config.policy_model_name,
        controller_addr=config.controller_addr,
    )
    dataset = get_dataset(config)
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    prm_tokenizer = AutoTokenizer.from_pretrained(config.reward_model_path)

    # #! debug
    # for example in tqdm(dataset, desc="Running search"):
    #     results = dvts(
    #         example,
    #         config,
    #         tokenizer,
    #         policy_model_caller,
    #         reward_model_caller,
    #     )

    if os.path.exists(config.output_file):
        answered_problem = set()
        df = pd.read_json(config.output_file, lines=True)
        for i in range(len(df)):
            answered_problem.add(df.iloc[i]["problem"][0])
        dataset = [
            example for example in dataset if example["problem"] not in answered_problem
        ]
        dataset = Dataset.from_list(dataset)

    actor_pool = ActorPool(
        [
            Actor.remote(policy_model_caller, tokenizer, prm_tokenizer)
            for _ in range(config.num_workers)
        ]
    )
    responses = actor_pool.map_unordered(
        lambda actor, example: actor.approach_fn.remote(
            example,
            config,
        ),
        dataset,
    )
    for response in tqdm(responses, total=len(dataset), desc="Running search"):
        with open(config.output_file, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    response,
                    ensure_ascii=False,
                )
                + "\n"
            )

    logger.info("Done 🔥!")


if __name__ == "__main__":
    main()
