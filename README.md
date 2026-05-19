# $\tau^2$-Bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains (AISG Internal Fork)

> **This is an internal AISG fork of [sierra-research/tau2-bench](https://github.com/sierra-research/tau2-bench) for evaluating locally-served models. It is not connected to the official tau-bench leaderboard and results are not submitted externally.**
>
> Default model under evaluation: [`Qwen/Qwen3.6-27B`](https://huggingface.co/Qwen/Qwen3.6-27B), served locally via vLLM. Judge: [`openai/gpt-oss-120b`](https://huggingface.co/openai/gpt-oss-120b), also served locally.

$\tau^2$-bench simulates customer service agent evaluation across multiple text-based domains. An LLM agent follows a policy, uses tools, and completes tasks — graded against expected outcomes.

**Domains used** (text half-duplex only):

| Domain | Tasks (base split) |
|--------|--------------------|
| `airline` | 50 |
| `retail` | 114 |
| `telecom` | 114 |
| `mock` | 10 (smoke test only) |

## Quick Start

### 1. Install (one-time, GPU node required)

```bash
sbatch setup_env.slurm
```

Check `logs/setup_<jobid>.out` for completion. Creates `.venv/` with tau2-bench + vLLM. A GPU node is required — vLLM compiles CUDA kernels on first install.

### 2. Add models to evaluate

Edit `config_vllm.yaml`:

```yaml
eval:
  default_model: Qwen/Qwen3.6-27B   # ← change this to switch default

models:
  Qwen/Qwen3.6-27B:
    tp: 1
    enable_thinking: true
    reasoning_parser: qwen3
    tool_call_parser: qwen3_coder

  # Example: add a new model
  Your/Model-Name:
    tp: 1
    enable_thinking: true
    tool_call_parser: qwen3_coder   # check vLLM docs for your model
    reasoning_parser: qwen3
```

`tp` = tensor parallel size (number of GPUs). The submit script reads this to request the right GPU count automatically.

### 3. Submit

```bash
# Default model from config_vllm.yaml
./submit_tau2bench.sh

# Specific model
./submit_tau2bench.sh Qwen/Qwen3.6-27B

# Single domain
DOMAIN=retail ./submit_tau2bench.sh Qwen/Qwen3.6-27B

# Smoke test (10 tasks)
NUM_TASKS=5 DOMAIN=airline ./submit_tau2bench.sh
```

Logs: `logs/tau2bench-<jobid>.out`

Resume an interrupted run by passing `OUTPUT=<existing-dir-basename>`:

```bash
OUTPUT=Qwen3.6-27B ./submit_tau2bench.sh Qwen/Qwen3.6-27B
```

| Variable          | Default                             | Source                                  | Description                                             |
| ----------------- | ----------------------------------- | --------------------------------------- | ------------------------------------------------------- |
| `MODEL`           | `Qwen/Qwen3.6-27B`                  | `config_vllm.yaml` → submit arg         | HuggingFace model ID under evaluation                   |
| `MODEL_TP`        | `1`                                 | `config_vllm.yaml models[MODEL].tp`     | Tensor parallel size — set per model in config          |
| `JUDGE_MODEL`     | `openai/gpt-oss-120b`               | `config_vllm.yaml eval.judge_model`     | Judge model for NL assertions                           |
| `JUDGE_TP`        | `1`                                 | `config_vllm.yaml eval.judge_tp`        | Tensor parallel size for judge model                    |
| `DOMAIN`          | `all`                               | `config_vllm.yaml eval.domain`          | `airline` (50), `retail` (114), `telecom` (114), `all` |
| `NUM_TRIALS`      | `1`                                 | `config_vllm.yaml eval.num_trials`      | Trials per task for pass@k                              |
| `NUM_TASKS`       | *(all)*                             | `config_vllm.yaml eval.num_tasks`       | Cap tasks per domain                                    |
| `MAX_CONCURRENCY` | `4`                                 | `config_vllm.yaml eval.max_concurrency` | Concurrent simulations                                  |
| `TASK_TIMEOUT`    | `300`                               | `config_vllm.yaml eval.task_timeout`    | Max wallclock seconds per task before abandoned         |
| `OUTPUT`          | `<model basename>`                  | SLURM script                            | Output name under `data/simulations/`                   |

## Results

Results for each domain are saved to:

```
data/simulations/<OUTPUT>_<domain>/results.json    # full simulation data
data/simulations/<OUTPUT>_<domain>/summary.txt     # printed metrics
```

Use `tau2 view` (after `source .venv/bin/activate`) to browse simulations interactively.

### AISG evaluation results

All models run with 3 trials. \* marks incomplete task counts (infrastructure errors — Gemma E-series only).

| Model | Domain | Tasks | Avg | Pass@1 | Pass^1 | Pass^2 | Pass^3 |
|-------|--------|------:|----:|-------:|-------:|-------:|-------:|
| aisingapore/gemma4_e2b_cand1 | airline | 50 | 0.203 | 0.240 | 0.200 | 0.220 | 0.140 |
| | retail | 114 | 0.117 | 0.175 | 0.117 | 0.096 | 0.079 |
| | telecom | 112\* | 0.114 | 0.140 | 0.111 | 0.123 | 0.070 |
| | **TOTAL** | **276\*** | **0.131** | **0.173** | **0.130** | **0.130** | **0.086** |
| aisingapore/gemma4_e2b_cand2 | airline | 49\* | 0.204 | 0.260 | 0.200 | 0.200 | 0.140 |
| | retail | 114 | 0.091 | 0.123 | 0.091 | 0.088 | 0.061 |
| | telecom | 113\* | 0.121 | 0.132 | 0.120 | 0.123 | 0.105 |
| | **TOTAL** | **276\*** | **0.124** | **0.151** | **0.122** | **0.122** | **0.093** |
| aisingapore/qwen36_27b_arcee | airline | 50 | 0.553 | 0.720 | 0.553 | 0.560 | 0.380 |
| | retail | 114 | 0.361 | 0.544 | 0.360 | 0.351 | 0.184 |
| | telecom | 114 | 0.573 | 0.842 | 0.573 | 0.588 | 0.289 |
| | **TOTAL** | **278** | **0.482** | **0.698** | **0.482** | **0.486** | **0.263** |
| aisingapore/qwen36_27b_cand1 | airline | 50 | 0.480 | 0.660 | 0.480 | 0.500 | 0.280 |
| | retail | 114 | 0.310 | 0.526 | 0.310 | 0.272 | 0.132 |
| | telecom | 114 | 0.534 | 0.772 | 0.532 | 0.535 | 0.289 |
| | **TOTAL** | **278** | **0.432** | **0.651** | **0.432** | **0.421** | **0.223** |
| aisingapore/qwen36_27b_cand2 | airline | 50 | 0.430 | 0.700 | 0.427 | 0.400 | 0.180 |
| | retail | 114 | 0.249 | 0.465 | 0.249 | 0.211 | 0.070 |
| | telecom | 114 | 0.941 | 1.000 | 0.939 | 0.965 | 0.851 |
| | **TOTAL** | **278** | **0.565** | **0.727** | **0.564** | **0.554** | **0.410** |
| aisingapore/qwen36_27b_cand3 | airline | 50 | 0.413 | 0.660 | 0.413 | 0.400 | 0.180 |
| | retail | 114 | 0.374 | 0.693 | 0.374 | 0.316 | 0.114 |
| | telecom | 114 | 0.915 | 1.000 | 0.915 | 0.956 | 0.789 |
| | **TOTAL** | **278** | **0.603** | **0.813** | **0.603** | **0.594** | **0.403** |
| aisingapore/qwen36_27b_cand5 | airline | 50 | 0.473 | 0.700 | 0.473 | 0.500 | 0.220 |
| | retail | 114 | 0.316 | 0.553 | 0.316 | 0.307 | 0.088 |
| | telecom | 114 | 0.886 | 0.982 | 0.886 | 0.947 | 0.728 |
| | **TOTAL** | **278** | **0.578** | **0.755** | **0.578** | **0.604** | **0.374** |
| google/gemma-4-31B-it | airline | 50 | 0.627 | 0.720 | 0.627 | 0.640 | 0.520 |
| | retail | 114 | 0.690 | 0.798 | 0.690 | 0.684 | 0.588 |
| | telecom | 114 | 0.327 | 0.421 | 0.327 | 0.316 | 0.246 |
| | **TOTAL** | **278** | **0.530** | **0.629** | **0.530** | **0.525** | **0.435** |
| google/gemma-4-E2B-it | airline | 49\* | 0.271 | 0.320 | 0.260 | 0.260 | 0.200 |
| | retail | 114 | 0.105 | 0.123 | 0.105 | 0.105 | 0.088 |
| | telecom | 96\* | 0.151 | 0.149 | 0.111 | 0.105 | 0.079 |
| | **TOTAL** | **259\*** | **0.154** | **0.170** | **0.137** | **0.135** | **0.106** |
| google/gemma-4-E4B-it | airline | 49\* | 0.404 | 0.480 | 0.393 | 0.360 | 0.340 |
| | retail | 113\* | 0.107 | 0.175 | 0.105 | 0.079 | 0.061 |
| | telecom | 109\* | 0.171 | 0.184 | 0.161 | 0.167 | 0.132 |
| | **TOTAL** | **271\*** | **0.186** | **0.234** | **0.180** | **0.165** | **0.140** |
| Qwen/Qwen3.5-27B | airline | 50 | 0.593 | 0.780 | 0.593 | 0.580 | 0.420 |
| | retail | 114 | 0.333 | 0.614 | 0.333 | 0.289 | 0.096 |
| | telecom | 114 | 0.962 | 1.000 | 0.962 | 0.991 | 0.895 |
| | **TOTAL** | **278** | **0.638** | **0.802** | **0.638** | **0.629** | **0.482** |
| Qwen/Qwen3.6-27B | airline | 50 | 0.584 | 0.760 | 0.580 | 0.620 | 0.360 |
| | retail | 114 | 0.364 | 0.596 | 0.363 | 0.333 | 0.158 |
| | telecom | 114 | 0.895 | 0.965 | 0.895 | 0.912 | 0.807 |
| | **TOTAL** | **278** | **0.621** | **0.777** | **0.620** | **0.622** | **0.460** |

\* = incomplete (fewer tasks than domain total — infrastructure errors, not retriable).

`aisingapore/qwen36_27b_arcee` is a short alias for `qwen36_27b_tlmsmytathvi_sparse_reversekl_otr_response_sys_397b_teacher_w_sys_4000_arcee`.

### Thinking-off (nothink) variants

Runs with `enable_thinking: false` (`--default-chat-template-kwargs '{"enable_thinking": false}'`). 3 trials each. \* marks incomplete task counts.

| Model | Domain | Tasks | Avg | Pass@1 | Pass^1 | Pass^2 | Pass^3 |
|-------|--------|------:|----:|-------:|-------:|-------:|-------:|
| aisingapore/gemma4_e2b_cand1_nothink | airline | 50 | 0.195 | 0.280 | 0.193 | 0.160 | 0.140 |
| | retail | 114 | 0.094 | 0.114 | 0.094 | 0.096 | 0.070 |
| | telecom | 108\* | 0.118 | 0.123 | 0.108 | 0.114 | 0.088 |
| | **TOTAL** | **272\*** | **0.122** | **0.148** | **0.118** | **0.115** | **0.090** |
| aisingapore/gemma4_e2b_cand2_nothink | airline | 49\* | 0.240 | 0.280 | 0.233 | 0.240 | 0.180 |
| | retail | 114 | 0.094 | 0.140 | 0.094 | 0.070 | 0.070 |
| | telecom | 114 | 0.136 | 0.158 | 0.135 | 0.132 | 0.114 |
| | **TOTAL** | **277\*** | **0.137** | **0.172** | **0.135** | **0.126** | **0.108** |
| aisingapore/qwen36_27b_arcee_nothink | airline | 50 | 0.720 | 0.800 | 0.720 | 0.740 | 0.620 |
| | retail | 114 | 0.781 | 0.868 | 0.781 | 0.807 | 0.667 |
| | telecom | 114 | 0.795 | 0.886 | 0.795 | 0.789 | 0.711 |
| | **TOTAL** | **278** | **0.776** | **0.863** | **0.776** | **0.788** | **0.676** |
| aisingapore/qwen36_27b_cand1_nothink | airline | 50 | 0.780 | 0.840 | 0.780 | 0.780 | 0.720 |
| | retail | 114 | 0.807 | 0.877 | 0.807 | 0.833 | 0.711 |
| | telecom | 114 | 0.813 | 0.895 | 0.813 | 0.807 | 0.737 |
| | **TOTAL** | **278** | **0.805** | **0.878** | **0.805** | **0.813** | **0.723** |
| aisingapore/qwen36_27b_cand2_nothink | airline | 50 | 0.820 | 0.900 | 0.820 | 0.820 | 0.740 |
| | retail | 114 | 0.830 | 0.912 | 0.830 | 0.851 | 0.728 |
| | telecom | 114 | 0.839 | 0.912 | 0.839 | 0.851 | 0.754 |
| | **TOTAL** | **278** | **0.832** | **0.910** | **0.832** | **0.845** | **0.741** |
| aisingapore/qwen36_27b_cand3_nothink | airline | 50 | 0.760 | 0.880 | 0.760 | 0.760 | 0.640 |
| | retail | 114 | 0.818 | 0.895 | 0.816 | 0.825 | 0.728 |
| | telecom | 114 | 0.860 | 0.956 | 0.860 | 0.886 | 0.737 |
| | **TOTAL** | **278** | **0.825** | **0.917** | **0.824** | **0.838** | **0.716** |
| aisingapore/qwen36_27b_cand5_nothink | airline | 50 | 0.720 | 0.840 | 0.720 | 0.720 | 0.600 |
| | retail | 114 | 0.743 | 0.860 | 0.743 | 0.763 | 0.605 |
| | telecom | 114 | 0.930 | 0.991 | 0.930 | 0.930 | 0.868 |
| | **TOTAL** | **278** | **0.815** | **0.910** | **0.815** | **0.824** | **0.712** |
| google/gemma-4-31B-it_nothink | airline | 50 | 0.667 | 0.760 | 0.667 | 0.660 | 0.580 |
| | retail | 114 | 0.643 | 0.754 | 0.643 | 0.667 | 0.509 |
| | telecom | 114 | 0.350 | 0.482 | 0.348 | 0.307 | 0.254 |
| | **TOTAL** | **278** | **0.527** | **0.644** | **0.526** | **0.518** | **0.417** |
| google/gemma-4-E2B-it_nothink | airline | 48\* | 0.224 | 0.280 | 0.213 | 0.240 | 0.120 |
| | retail | 114 | 0.111 | 0.149 | 0.111 | 0.105 | 0.079 |
| | telecom | 49\* | 0.179 | 0.079 | 0.064 | 0.061 | 0.053 |
| | **TOTAL** | **211\*** | **0.152** | **0.163** | **0.124** | **0.126** | **0.082** |
| google/gemma-4-E4B-it_nothink | airline | 48\* | 0.382 | 0.440 | 0.367 | 0.340 | 0.320 |
| | retail | 114 | 0.100 | 0.158 | 0.099 | 0.079 | 0.061 |
| | telecom | 109\* | 0.183 | 0.184 | 0.164 | 0.175 | 0.132 |
| | **TOTAL** | **271\*** | **0.183** | **0.218** | **0.173** | **0.164** | **0.135** |
| Qwen/Qwen3.5-27B_nothink | airline | 50 | 0.767 | 0.860 | 0.767 | 0.800 | 0.640 |
| | retail | 114 | 0.690 | 0.816 | 0.690 | 0.719 | 0.535 |
| | telecom | 114 | 0.749 | 0.789 | 0.749 | 0.746 | 0.711 |
| | **TOTAL** | **278** | **0.728** | **0.813** | **0.728** | **0.745** | **0.626** |
| Qwen/Qwen3.6-27B_nothink | airline | 50 | 0.727 | 0.840 | 0.727 | 0.760 | 0.580 |
| | retail | 114 | 0.789 | 0.851 | 0.789 | 0.789 | 0.728 |
| | telecom | 114 | 0.816 | 0.930 | 0.816 | 0.833 | 0.684 |
| | **TOTAL** | **278** | **0.789** | **0.881** | **0.789** | **0.802** | **0.683** |

Metric definitions:
- **Avg** — mean reward across all simulations (0–1)
- **Pass@1** — fraction of tasks where ≥1 trial passed (reward ≥ 0.5); equals Pass^1 for 1-trial runs
- **Pass^1** — per-trial pass rate across all task-trial pairs
- **Pass^2** — fraction of tasks where ≥2 trials passed
- **Pass^3** — fraction of tasks where all 3 trials passed (strict reliability)

To re-score manually:

```bash
srun --gres=gpu:0 --mem=8G .venv/bin/python score_summary.py data/simulations/
```

Or for a single domain dir:

```bash
srun --gres=gpu:0 --mem=8G .venv/bin/python score_summary.py data/simulations/Qwen3.6-27B_airline
```

## Reference

Upstream repo, docs, voice/knowledge features, and citation info below — not used in this fork's evaluation setup.

<details>
<summary>Upstream documentation</summary>

### Docs

| Document | Description |
|----------|-------------|
| [Getting Started](docs/getting-started.md) | Installation, API keys, first run, output structure, configuration |
| [CLI Reference](docs/cli-reference.md) | All `tau2` commands and options |
| [Agent Developer Guide](src/tau2/agent/README.md) | Build and evaluate your own agent |
| [Domains](src/tau2/domains/README.md) | Domain structure, data format, and available domains |
| [Knowledge Retrieval](src/tau2/knowledge/README.md) | `banking_knowledge` domain setup |
| [Voice (Full-Duplex)](src/tau2/voice/README.md) | Voice evaluation with realtime providers |
| [Gym Interface](src/tau2/gym/README.md) | Gymnasium-compatible RL environment |
| [Changelog](CHANGELOG.md) | Version history |

### Citation

```bibtex
@misc{barres2025tau2,
      title={$\tau^2$-Bench: Evaluating Conversational Agents in a Dual-Control Environment},
      author={Victor Barres and Honghua Dong and Soham Ray and Xujie Si and Karthik Narasimhan},
      year={2025},
      eprint={2506.07982},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2506.07982},
}

@misc{yao2024tau,
      title={$\tau$-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains},
      author={Shunyu Yao and Noah Shinn and Pedram Razavi and Karthik Narasimhan},
      year={2024},
      eprint={2406.12045},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2406.12045},
}
```

</details>
