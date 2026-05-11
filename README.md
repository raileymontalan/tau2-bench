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

### 1. Install

```bash
git clone https://github.com/raileymontalan/tau2-bench.git
cd tau2-bench
uv sync
```

### 2. Install vLLM and serve the model

Works with CUDA 13 (loaded automatically in the SLURM script):

```bash
uv sync --extra vllm
```

```bash
.venv/bin/vllm serve Qwen/Qwen3.6-27B \
  --port 8000 \
  --tensor-parallel-size ${NUM_GPUS:-1}
```

Set `HF_HOME` if model weights are cached in a non-default location:

```bash
export HF_HOME="${SHARED_FS_DIR}/hf_cache"
```

### 3. Set up environment

```bash
cp .env.example .env
```

Add to `.env`:

```bash
OPENAI_API_BASE=http://localhost:8000/v1
OPENAI_API_KEY=dummy
```

### 4. Run an evaluation

```bash
tau2 run --domain airline \
  --agent-llm openai/Qwen/Qwen3-32B \
  --user-llm openai/Qwen/Qwen3-32B \
  --num-trials 1 --num-tasks 5
```

Results are saved to `data/simulations/`. Use `tau2 view` to browse them.

## Results

After the job finishes, results for each domain are saved to:

```
data/simulations/<OUTPUT>_<domain>/results.json    # full simulation data
data/simulations/<OUTPUT>_<domain>/summary.txt     # printed metrics
```

where `OUTPUT` defaults to the model basename (e.g. `Qwen3.6-27B`).

**Per-domain summary output:**

```
Tasks:       50
Simulations: 50
Pass^1:      0.8000
Avg Reward:  0.8000
Write Acts:  40/50 (80.0%)
DB Match:    45/50 (90.0%)
```

- **Pass^1** — fraction of tasks where at least 1 trial fully succeeded
- **Avg Reward** — mean reward across all simulations (0–1)
- **Write Acts** — correct write-action calls / total write-action calls
- **DB Match** — simulations whose final DB state matched expected

To re-score manually:

```bash
uv run python3 - data/simulations/Qwen3.6-27B_airline/results.json <<'PYEOF'
import sys
from pathlib import Path
from tau2.data_model.simulation import Results
from tau2.metrics.agent_metrics import compute_metrics
results = Results.load(Path(sys.argv[1]))
m = compute_metrics(results)
pass1 = m.pass_hat_ks.get(1, m.avg_reward)
print(f"Tasks:       {m.total_tasks}")
print(f"Simulations: {m.total_simulations}")
print(f"Pass^1:      {pass1:.4f}")
print(f"Avg Reward:  {m.avg_reward:.4f}")
PYEOF
```

### Running on SLURM

Use `submit_tau2bench.sh` to submit jobs. It reads model settings from `config_vllm.yaml` and automatically sets the correct GPU count:

```bash
# Default model (from config_vllm.yaml eval.default_model)
./submit_tau2bench.sh

# Specific model
./submit_tau2bench.sh Qwen/Qwen3-32B

# Benchmark overrides via env vars
NUM_TASKS=5 ./submit_tau2bench.sh

# Single domain
DOMAIN=retail ./submit_tau2bench.sh Qwen/Qwen3.6-27B
```

To add a new model, add an entry to `config_vllm.yaml` under `models:`:

```yaml
models:
  Your/Model-Name:
    tp: 1                        # number of GPUs (tensor parallel size)
    enable_thinking: true        # false to disable thinking tokens
    tool_call_parser: qwen3_coder  # vLLM tool call parser; check vLLM docs for your model
    reasoning_parser: qwen3      # vLLM reasoning parser; omit or leave empty if not applicable
```

No SLURM changes needed. Submit with `./submit_tau2bench.sh Your/Model-Name`.

| Variable               | Default                       | Source                              | Description                                              |
| ---------------------- | ----------------------------- | ----------------------------------- | -------------------------------------------------------- |
| `MODEL`                | `Qwen/Qwen3.6-27B`            | `config_vllm.yaml` → submit arg     | HuggingFace model ID under evaluation                    |
| `MODEL_TP`             | `1`                           | `config_vllm.yaml models[MODEL].tp` | Tensor parallel size — set per model in config           |
| `JUDGE_MODEL`          | `openai/gpt-oss-120b`         | `config_vllm.yaml eval.judge_model` | Judge model for NL assertions                            |
| `JUDGE_TP`             | `1`                           | `config_vllm.yaml eval.judge_tp`    | Tensor parallel size for judge model                     |
| `DOMAIN`               | `all`                         | `config_vllm.yaml eval.domain`      | `airline` (50), `retail` (114), `telecom` (114), `all`  |
| `NUM_TRIALS`           | `1`                           | `config_vllm.yaml eval.num_trials`  | Trials per task for pass@k                               |
| `NUM_TASKS`            | *(all)*                       | `config_vllm.yaml eval.num_tasks`   | Cap tasks per domain                                     |
| `MAX_CONCURRENCY`      | `4`                           | `config_vllm.yaml eval.max_concurrency` | Concurrent simulations                               |
| `TASK_TIMEOUT`         | `300`                         | `config_vllm.yaml eval.task_timeout` | Max wallclock seconds per task before abandoned         |
| `OUTPUT`               | `<model basename>`            | SLURM script                        | Output name under `data/simulations/`                    |

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
