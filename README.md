<div align="center">
  <h1>Examining False Positives under Inference Scaling for Mathematical Reasoning</h1>
 </p>
</div>
<br>

# Setup

```bash
conda create -n false_positive python==3.10.9
conda activate false_positive
pip install -r requirements.txt
# skyword-prm
cd reward_model/Skywork-o1-Open-PRM-Qwen-2.5-7B && pip install -e . && cd ../..
# qwen-rm (vllm 0.6.4.post2.dev339+g9f3974a3)
pip install https://vllm-wheels.s3.us-west-2.amazonaws.com/9f3974a31911b551d416bb4d435273409d23f021/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl
# DVTS
cd step-level/DVTS && pip install -e '.[dev]' && cd ../..
# MCTS
pip install loguru==0.7.3 jsonlines==4.0.0 pylatexenc==2.10
```

# File Structure

```bash
├── README.md
├── benchmarks
│   ├── AIME.jsonl
│   ├── MATH100.jsonl
│   └── OmniMATH100-rule.jsonl
├── evaluation
│   ├── eval-Qwen2.5_Math
│   │   ├── get_results.py
│   │   ├── grader.py
│   │   ├── math_utils.py
│   │   ├── parser.py
│   │   ├── save_pass_at_n_correct.py
│   │   ├── save_verifier_correct.py
│   │   └── scripts
│   │       ├── get_results.sh
│   │       ├── save_pass_at_n_correct.sh
│   │       └── save_verifier_correct.sh
│   └── false-positives-model-detection
│       ├── gpt4o_check.py
│       ├── llama_check.py
│       └── qwen_check.py
├── requirements.txt
├── results
│   ├── exp5.2-model_detection_false_positives
│   │   ├── gpt4o_check
│   │   │   ├── ...
│   │   ├── llama_check
│   │   │   ├── ...
│   │   └── qwen_check
│   │       ├── ...
│   ├── exp5.3.1-false_positives_inference_scaling
│   │   ├── diff_benchmarks
│   │   │   ├── ...
│   │   ├── diff_methods
│   │   │   ├── solution-level
│   │   │   │   ├── ...
│   │   │   └── step-level
│   │   │       ├── ...
│   │   └── diff_model_types
│   │       ├── ...
│   ├── exp5.3.2-pass_at_n
│   │   ├── ...
│   ├── exp5.3.3-rule_based_grpo
│   │   └── ...
│   ├── exp5.3.4-long_cot
│   │   └── ...
│   └── human_detection_records
│       ├── ...
├── reward_model
│   ├── Qwen2.5-Math-RM-72B
│   │   ├── dvts_mcts_get_score.py
│   │   ├── llama_get_score.py
│   │   ├── qwen_get_score.py
│   │   └── scripts
│   │       ├── get_score.sh
│   │       └── qwen2.5_math_rm_72b.sh
│   └── Skywork-o1-Open-PRM-Qwen-2.5-7B
│       ├── llama_get_score.py
│       ├── mcts_get_score.py
│       ├── model_utils
│       │   ├── io_utils.py
│       │   ├── modeling_base.py
│       │   └── prm_model.py
│       ├── qwen_get_score.py
│       ├── scripts
│       │   ├── get_score.sh
│       │   └── skywork_o1_prm.sh
│       ├── setup.py
│       └── vllm_add_dummy_model
│           ├── __init__.py
│           └── prm_model.py
├── solution-level
│   ├── distill_r1_sample.py
│   ├── llama_sample.py
│   ├── qwen_sample.py
│   └── scripts
│       ├── distill_r1_sample.sh
│       ├── llama_sample.sh
│       ├── oat_zero_sample.sh
│       └── qwen_sample.sh
└── step-level
    ├── DVTS
    │   ├── Makefile
    │   ├── pyproject.toml
    │   ├── recipes
    │   │   ├── Llama-3.1-8B-Instruct
    │   │   │   └── dvts.yaml
    │   │   ├── Llama-3.2-3B-Instruct
    │   │   │   └── dvts.yaml
    │   │   ├── Qwen2.5-Math-1.5B-Instruct
    │   │   │   └── dvts.yaml
    │   │   └── Qwen2.5-Math-7B-Instruct
    │   │       └── dvts.yaml
    │   ├── run
    │   │   ├── merge_chunks.py
    │   │   └── test_time_compute.py
    │   ├── scripts
    │   │   ├── create_service_vllm.sh
    │   │   ├── dvts.sh
    │   │   └── skywork_o1_prm.sh
    │   ├── setup.py
    │   └── src
    │       ├── sal
    │       │   ├── __init__.py
    │       │   ├── config.py
    │       │   ├── llm_caller
    │       │   │   ├── __init__.py
    │       │   │   ├── policy_model_caller.py
    │       │   │   └── policy_model_inference.py
    │       │   ├── llm_service
    │       │   │   ├── __init__.py
    │       │   │   ├── base_model_worker.py
    │       │   │   └── policy_model_worker.py
    │       │   ├── search
    │       │   │   ├── __init__.py
    │       │   │   ├── beam_search.py
    │       │   │   ├── diverse_verifier_tree_search.py
    │       │   │   └── utils.py
    │       │   └── utils
    │       │       ├── __init__.py
    │       │       ├── data.py
    │       │       ├── hub.py
    │       │       ├── math.py
    │       │       ├── parser.py
    │       │       ├── qwen_math_parser.py
    │       │       └── score.py
    │       └── skywork_prm
    │           ├── get_score.py
    │           ├── model_utils
    │           │   ├── io_utils.py
    │           │   ├── modeling_base.py
    │           │   └── prm_model.py
    │           ├── setup.py
    │           └── vllm_add_dummy_model
    │               ├── __init__.py
    │               └── prm_model.py
    └── MCTS
        ├── distributed
        │   └── utils.py
        ├── envs
        │   ├── AIME
        │   │   ├── __init__.py
        │   │   ├── data.py
        │   │   ├── dataset
        │   │   │   └── aime22_23_24.jsonl
        │   │   ├── env.py
        │   │   ├── grader.py
        │   │   ├── parse_utils_qwen.py
        │   │   ├── prompt.py
        │   │   └── verify_utils.py
        │   ├── MATH
        │   │   ├── __init__.py
        │   │   ├── data.py
        │   │   ├── dataset
        │   │   │   └── math500.jsonl
        │   │   ├── env.py
        │   │   ├── grader.py
        │   │   ├── parse_utils_qwen.py
        │   │   ├── prompt.py
        │   │   └── verify_utils.py
        │   ├── OmniMATH
        │   │   ├── __init__.py
        │   │   ├── data.py
        │   │   ├── dataset
        │   │   │   └── OmniMATH500.jsonl
        │   │   ├── env.py
        │   │   ├── grader.py
        │   │   ├── parse_utils_qwen.py
        │   │   ├── prompt.py
        │   │   └── verify_utils.py
        │   ├── __init__.py
        │   └── base_env.py
        ├── evaluate.py
        ├── reason
        │   ├── evaluation
        │   │   ├── evaluator.py
        │   │   ├── methods.py
        │   │   └── utils.py
        │   ├── guided_search
        │   │   └── tree.py
        │   ├── inference
        │   │   ├── lm_call.py
        │   │   ├── skywork_score.py
        │   │   └── text_generation.py
        │   ├── llm_service
        │   │   └── workers
        │   │       ├── base_model_worker.py
        │   │       └── vllm_worker.py
        │   └── reranking
        │       └── vote_utils.py
        ├── scripts
        │   ├── create_service_vllm.sh
        │   ├── skywork_o1_prm.sh
        │   └── vanila_mcts.sh
        └── skywork_prm
            ├── get_score.py
            ├── model_utils
            │   ├── io_utils.py
            │   ├── modeling_base.py
            │   └── prm_model.py
            ├── setup.py
            └── vllm_add_dummy_model
                ├── __init__.py
                └── prm_model.py
```