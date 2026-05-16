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

Base models (Gemma, Qwen3.5-27B, Qwen3.6-27B) ran with 1 trial — Pass^2/^3 not shown. Candidate models used 3 trials; †-marked domains had only 1 trial completed (pass^2/^3 not available for that domain; TOTAL pass^2/^3 is conservative accordingly).

| Model | Domain | Tasks | Avg | Pass@1 | Pass^1 | Pass^2 | Pass^3 |
|-------|--------|------:|----:|-------:|-------:|-------:|-------:|
| aisingapore/qwen36_27b_arcee | airline | 50 | 0.553 | 0.720 | 0.553 | 0.560 | 0.380 |
| | retail | 114 | 0.362 | 0.544 | 0.360 | 0.351 | 0.184 |
| | telecom | 94\*† | 0.660 | 0.660 | 0.660 | — | — |
| | **TOTAL** | **258\*** | **0.507** | **0.620** | **0.507** | **0.264** | **0.155** |
| aisingapore/qwen36_27b_cand1 | airline | 50 | 0.480 | 0.660 | 0.480 | 0.500 | 0.280 |
| | retail | 114 | 0.310 | 0.526 | 0.310 | 0.272 | 0.132 |
| | telecom | 84\*† | 0.667 | 0.659 | 0.659 | — | — |
| | **TOTAL** | **248\*** | **0.465** | **0.598** | **0.462** | **0.226** | **0.117** |
| aisingapore/qwen36_27b_cand2 | airline | 50 | 0.430 | 0.700 | 0.427 | 0.400 | 0.180 |
| | retail | 114 | 0.221 | 0.342 | 0.220 | 0.088 | 0.000 |
| | telecom | — | — | — | — | — | — |
| | **TOTAL** | **164** | **0.284** | **0.451** | **0.283** | **0.183** | **0.055** |
| aisingapore/qwen36_27b_cand3 | airline | 50 | 0.413 | 0.660 | 0.413 | 0.400 | 0.180 |
| | retail | 114 | 0.392 | 0.579 | 0.392 | 0.167 | 0.000 |
| | telecom | — | — | — | — | — | — |
| | **TOTAL** | **164** | **0.398** | **0.604** | **0.398** | **0.238** | **0.055** |
| google/gemma-4-31B-it | airline | 50 | 0.627 | 0.640 | 0.640 | — | — |
| | retail | 114 | 0.690 | 0.658 | 0.658 | — | — |
| | telecom | 114 | 0.328 | 0.333 | 0.333 | — | — |
| | **TOTAL** | **278** | **0.530** | **0.522** | **0.522** | — | — |
| google/gemma-4-E2B-it | airline | 48\* | 0.261 | 0.260 | 0.260 | — | — |
| | retail | 114 | 0.105 | 0.097 | 0.097 | — | — |
| | telecom | 80\* | 0.157 | 0.053 | 0.053 | — | — |
| | **TOTAL** | **242** | **0.153** | **0.114** | **0.114** | — | — |
| google/gemma-4-E4B-it | airline | 49\* | 0.404 | 0.400 | 0.400 | — | — |
| | retail | 113\* | 0.106 | 0.114 | 0.114 | — | — |
| | telecom | 109\* | 0.170 | 0.149 | 0.149 | — | — |
| | **TOTAL** | **271** | **0.186** | **0.180** | **0.180** | — | — |
| Qwen/Qwen3.5-27B | airline | 50 | 0.593 | 0.540 | 0.540 | — | — |
| | retail | 114 | 0.333 | 0.316 | 0.316 | — | — |
| | telecom | 114 | 0.962 | 0.965 | 0.965 | — | — |
| | **TOTAL** | **278** | **0.638** | **0.622** | **0.622** | — | — |
| Qwen/Qwen3.6-27B | airline | 50 | 0.584 | 0.560 | 0.560 | — | — |
| | retail | 114 | 0.365 | 0.368 | 0.368 | — | — |
| | telecom | 114 | 0.895 | 0.895 | 0.895 | — | — |
| | **TOTAL** | **278** | **0.622** | **0.619** | **0.619** | — | — |

\* = run incomplete. "—" = domain not run. † = only 1 trial completed for this domain; pass^2/^3 not applicable.

`aisingapore/qwen36_27b_arcee` is a short alias for `qwen36_27b_tlmsmytathvi_sparse_reversekl_otr_response_sys_397b_teacher_w_sys_4000_arcee`.

Re-run incomplete domains: `./submit_tau2bench.sh <model>` (uses `--auto-resume`).

> **Known issue — gemma-4-E2B-it / E4B-it telecom infrastructure errors:**
> Both Gemma E-series models produce empty assistant messages (no `content`, no `tool_calls`) on some telecom tasks, causing simulation failures. This is a model capability limitation. Re-runs will not improve these numbers significantly.

### Thinking-off (nothink) variants

Runs with `enable_thinking: false` (`--default-chat-template-kwargs '{"enable_thinking": false}'`). 3 trials each. All models have full 3-domain coverage; \* marks incomplete task counts within a domain.

| Model | Domain | Tasks | Avg | Pass@1 | Pass^1 | Pass^2 | Pass^3 |
|-------|--------|------:|----:|-------:|-------:|-------:|-------:|
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
| google/gemma-4-31B-it_nothink | airline | 50 | 0.667 | 0.760 | 0.667 | 0.660 | 0.580 |
| | retail | 114 | 0.643 | 0.754 | 0.643 | 0.667 | 0.509 |
| | telecom | 114 | 0.350 | 0.483 | 0.348 | 0.307 | 0.254 |
| | **TOTAL** | **278** | **0.527** | **0.644** | **0.526** | **0.518** | **0.417** |
| google/gemma-4-E2B-it_nothink | airline | 48\* | 0.218 | 0.280 | 0.207 | 0.240 | 0.100 |
| | retail | 114 | 0.111 | 0.149 | 0.111 | 0.105 | 0.079 |
| | telecom | 45\*† | 0.216 | 0.079 | 0.064 | 0.061 | 0.053 |
| | **TOTAL** | **207\*** | **0.159** | **0.164** | **0.123** | **0.127** | **0.078** |
| google/gemma-4-E4B-it_nothink | airline | 48\* | 0.387 | 0.440 | 0.367 | 0.340 | 0.320 |
| | retail | 112\* | 0.099 | 0.149 | 0.097 | 0.079 | 0.061 |
| | telecom | 109\* | 0.167 | 0.184 | 0.135 | 0.140 | 0.079 |
| | **TOTAL** | **269\*** | **0.178** | **0.215** | **0.160** | **0.150** | **0.115** |
| Qwen/Qwen3.5-27B_nothink | airline | 50 | 0.767 | 0.860 | 0.767 | 0.800 | 0.640 |
| | retail | 114 | 0.690 | 0.816 | 0.690 | 0.719 | 0.535 |
| | telecom | 114 | 0.749 | 0.789 | 0.749 | 0.746 | 0.711 |
| | **TOTAL** | **278** | **0.728** | **0.813** | **0.728** | **0.745** | **0.626** |
| Qwen/Qwen3.6-27B_nothink | airline | 50 | 0.727 | 0.840 | 0.727 | 0.760 | 0.580 |
| | retail | 114 | 0.789 | 0.851 | 0.789 | 0.789 | 0.728 |
| | telecom | 114 | 0.816 | 0.930 | 0.816 | 0.833 | 0.684 |
| | **TOTAL** | **278** | **0.789** | **0.881** | **0.789** | **0.802** | **0.684** |

† = high infrastructure failure rate (empty-message issue, same as thinking variant).

Metric definitions:
- **Avg** — mean reward across all simulations (0–1)
- **Pass@1** — fraction of tasks where ≥1 trial passed (reward ≥ 0.5); equals Pass^1 for 1-trial runs
- **Pass^1** — per-trial pass rate across all task-trial pairs
- **Pass^2** — fraction of tasks where ≥2 trials passed
- **Pass^3** — fraction of tasks where all 3 trials passed (strict reliability)

To re-score manually:

```bash
python score_summary.py data/simulations/
```

Or for a single domain dir:

```bash
python score_summary.py data/simulations/Qwen3.6-27B_airline
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
