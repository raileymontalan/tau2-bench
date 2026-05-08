# $\tau^2$-Bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains (AISG Internal Fork)

> **This is an internal AISG fork of [sierra-research/tau2-bench](https://github.com/sierra-research/tau2-bench) for evaluating locally-served models. It is not connected to the official tau-bench leaderboard and results are not submitted externally.**
>
> Default model under evaluation: [`Qwen/Qwen3-32B`](https://huggingface.co/Qwen/Qwen3-32B), served locally via vLLM. Judge: [`openai/gpt-oss-120b`](https://huggingface.co/openai/gpt-oss-120b), also served locally.

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

Works with CUDA 12.4 (default driver on the cluster):

```bash
uv sync --extra vllm
```

```bash
.venv/bin/vllm serve Qwen/Qwen3-32B \
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

### Running on SLURM

For cluster jobs, use the provided script which handles vLLM startup, health-checking, and teardown:

```bash
# Default run (Qwen3-32B, all core domains)
sbatch run_tau2bench.slurm

# Quick smoke test, 5 tasks per domain
NUM_TASKS=5 sbatch run_tau2bench.slurm

# Single domain
DOMAIN=retail sbatch run_tau2bench.slurm

# Different model, more GPUs
MODEL=aisingapore/Other-Model TENSOR_PARALLEL_SIZE=4 sbatch --gres=gpu:4 run_tau2bench.slurm
```

| SLURM variable         | Default                   | Description                                                         |
| ---------------------- | ------------------------- | ------------------------------------------------------------------- |
| `MODEL`                | `Qwen/Qwen3-32B`          | HuggingFace model ID under evaluation                               |
| `MODEL_TP`             | `1`                       | Tensor parallel size for agent model                                |
| `JUDGE_MODEL`          | `openai/gpt-oss-120b`     | Judge model for NL assertions (served locally on GPU 1)             |
| `JUDGE_TP`             | `1`                       | Tensor parallel size for judge model                                |
| `DOMAIN`               | `all`                     | `airline` (50), `retail` (114), `telecom` (114), `mock` (10), `all` |
| `NUM_TRIALS`           | `1`                       | Trials per task for pass@k                                          |
| `NUM_TASKS`            | *(all)*                   | Cap tasks per domain                                                |
| `MAX_CONCURRENCY`      | `4`                      | Concurrent simulations (vLLM batches internally)                    |
| `TASK_TIMEOUT`         | `300`                     | Max wallclock seconds per task before it's abandoned                |
| `OUTPUT`               | `<model basename>`        | Output name under `data/simulations/`                               |

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
