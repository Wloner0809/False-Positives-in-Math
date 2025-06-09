from skywork_prm.model_utils.io_utils import derive_step_rewards_vllm, prepare_input


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
    for test_data in test_list:
        input_ids, steps, reward_flags = zip(*test_data)
        rewards = client.embeddings.create(
            input=input_ids,
            model=model,
        )
        step_rewards = derive_step_rewards_vllm(rewards, reward_flags)
        all_scores.append(step_rewards)
    return all_scores
