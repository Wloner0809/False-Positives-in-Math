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
# Analysis & Visualizer
pip install matplotlib==3.10.3 scikit-learn==1.7.0 plotly==6.1.2
```

# File Structure

```bash
├── README.md
├── analysis
│   ├── scripts
│   │   ├── t_sne.sh
│   │   └── t_sne_false_positive_type.sh
│   ├── t_sne.py
│   └── t_sne_false_positive_type.py
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
│   │   ├── save_verifier_incorrect.py
│   │   └── scripts
│   │       ├── get_results.sh
│   │       ├── save_pass_at_n_correct.sh
│   │       ├── save_verifier_correct.sh
│   │       └── save_verifier_incorrect.sh
│   └── false-positives-model-detection
│       ├── gpt4o_check.py
│       ├── llama_check.py
│       └── qwen_check.py
├── requirements.txt
├── results
│   ├── analysis
│   │   ├── all_model
│   │   │   ├── ...
│   │   ├── correct_false_positive_incorrect_all.jsonl
│   │   ├── correct_false_positive_incorrect_llama.jsonl
│   │   ├── correct_false_positive_incorrect_qwen.jsonl
│   │   └── false_positive_type
│   │       ├── ...
│   │       └── false_positive_types.jsonl
│   ├── exp5.2-model_detection_false_positives
│   │   ├── false_positive_detection_benchmark.jsonl
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
    │   ├── ...
    └── MCTS
        ├── ...
```