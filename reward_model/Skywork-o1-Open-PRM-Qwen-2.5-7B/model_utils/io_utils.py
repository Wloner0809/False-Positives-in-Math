import torch
import numpy as np


def prepare_input(problem, response, tokenizer, step_token, max_tokens):
    prompt_ids = tokenizer.encode(tokenizer.bos_token + problem + "\n")
    response_ids = []
    steps = []
    reward_flags = [0] * len(prompt_ids)
    step_token_id = tokenizer.encode(step_token)[-1]
    for idx, step in enumerate(response.split(step_token)):
        if step != "":
            step_ids = tokenizer.encode(step)
        else:
            continue
        step_ids += [step_token_id]
        step = step + step_token
        flag = [0] * len(step_ids)
        flag[-1] = 1
        response_ids.extend(step_ids)
        reward_flags.extend(flag)
        steps.append(step)
    input_ids = prompt_ids + response_ids
    if len(input_ids) > max_tokens:
        input_ids = input_ids[-max_tokens:]
        reward_flags = reward_flags[-max_tokens:]
    return input_ids, steps, reward_flags


def prepare_batch_input_for_model(input_ids, reward_flags, pad_token_id):
    padded_input_ids = torch.nn.utils.rnn.pad_sequence(
        [torch.LongTensor(ids) for ids in input_ids],
        batch_first=True,
        padding_value=pad_token_id,
    )
    padded_attention_mask = torch.nn.utils.rnn.pad_sequence(
        [torch.LongTensor([1] * len(ids)) for ids in input_ids],
        batch_first=True,
        padding_value=0,
    )
    padded_reward_flags = torch.nn.utils.rnn.pad_sequence(
        [torch.LongTensor(reward_flag) for reward_flag in reward_flags],
        batch_first=True,
        padding_value=0,
    )
    return padded_input_ids, padded_attention_mask, padded_reward_flags


def derive_step_rewards(rewards, reward_flags):
    batch_size = rewards.shape[0]
    batch_step_rewards = []
    for i in range(batch_size):
        rewards_indices = torch.nonzero(reward_flags[i] == 1).view(-1)
        step_rewards = [
            rewards[i][rewards_indices[j]].item() for j in range(len(rewards_indices))
        ]
        batch_step_rewards.append(step_rewards)
    return batch_step_rewards


def sigmoid(x):
    return 1 / (np.exp(-x) + 1)


def derive_step_rewards_vllm(raw_rewards, batch_reward_flags):
    batch_step_rewards = []
    for idx, data in enumerate(raw_rewards.data):
        rewards = data.embedding
        reward_flags = batch_reward_flags[idx]

        step_rewards = [
            sigmoid(reward) for reward, flag in zip(rewards, reward_flags) if flag == 1
        ]
        batch_step_rewards.append(step_rewards)
    return batch_step_rewards
