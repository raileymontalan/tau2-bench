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

Models ordered alphabetically. Current data is 1-trial runs — Pass^2/^3 will populate after 3-trial re-runs.

| Model | Domain | Tasks | Avg | Pass@1 | Pass^1 | Pass^2 | Pass^3 |
|-------|--------|------:|----:|-------:|-------:|-------:|-------:|
| google/gemma-4-31B-it | airline | 50 | 0.627 | 0.640 | 0.640 | — | — |
| | retail | 114 | 0.702 | 0.702 | 0.702 | — | — |
| | telecom | 114 | 0.281 | 0.281 | 0.281 | — | — |
| | **TOTAL** | **278** | **0.516** | **0.518** | **0.518** | — | — |
| google/gemma-4-E2B-it | airline | 47\* | 0.250 | 0.240 | 0.240 | — | — |
| | retail | 113\* | 0.124 | 0.123 | 0.123 | — | — |
| | telecom | 47\* | 0.170 | 0.070 | 0.070 | — | — |
| | **TOTAL** | **207** | **0.163** | **0.137** | **0.137** | — | — |
| google/gemma-4-E4B-it | airline | 49\* | 0.417 | 0.440 | 0.440 | — | — |
| | retail | 110\* | 0.100 | 0.096 | 0.096 | — | — |
| | telecom | 94\* | 0.138 | 0.114 | 0.114 | — | — |
| | **TOTAL** | **253** | **0.176** | **0.170** | **0.170** | — | — |
| Qwen/Qwen3.5-27B | airline | 50 | 0.726 | 0.700 | 0.700 | — | — |
| | retail | 41\* | 0.439 | 0.158 | 0.158 | — | — |
| | telecom | — | — | — | — | — | — |
| | **TOTAL** | **91** | **0.597** | **0.456** | **0.456** | — | — |
| Qwen/Qwen3.6-27B | airline | 50 | 0.638 | 0.620 | 0.620 | — | — |
| | retail | 94\* | 0.479 | 0.395 | 0.395 | — | — |
| | telecom | — | — | — | — | — | — |
| | **TOTAL** | **144** | **0.534** | **0.473** | **0.473** | — | — |

\* = run incomplete (fewer tasks than domain total). "—" = run did not complete.

Re-run incomplete domains: `./submit_tau2bench.sh <model>` (uses `--auto-resume`).

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
