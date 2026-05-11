#!/bin/bash
#
# Submit a tau2-bench SLURM job for a given model.
#
# Usage:
#   ./submit_tau2bench.sh [MODEL]
#
# MODEL defaults to eval.default_model in config_vllm.yaml.
# Additional SLURM overrides can be passed as env vars:
#   JUDGE_MODEL=openai/other-model ./submit_tau2bench.sh Qwen/Qwen3-32B
#   DOMAIN=retail NUM_TRIALS=3 ./submit_tau2bench.sh Qwen/Qwen3.6-27B

set -euo pipefail
cd "$(dirname "$0")"

MODEL="${1:-}"

# Resolve model and compute GPU count
_cfg=$(python3 - "$MODEL" <<'PYEOF'
import yaml, pathlib, sys

cfg = yaml.safe_load(pathlib.Path('config_vllm.yaml').read_text())
model = sys.argv[1] or cfg.get('eval', {}).get('default_model', '')
if not model:
    raise SystemExit("ERROR: MODEL not set and no default_model in config_vllm.yaml")

models = cfg.get('models', {})
if model not in models:
    available = list(models.keys())
    raise SystemExit(f"ERROR: Model '{model}' not found in config_vllm.yaml.\nAvailable: {available}")

mcfg = models[model]
model_tp = int(mcfg.get('tp', 1))
judge_tp = int(cfg.get('eval', {}).get('judge_tp', 1))
ngpus = model_tp + judge_tp
print(f"{model} {model_tp} {judge_tp} {ngpus}")
PYEOF
)
read -r MODEL MODEL_TP JUDGE_TP NGPUS <<< "$_cfg"

export MODEL
echo "Model:     ${MODEL}"
echo "Model TP:  ${MODEL_TP}"
echo "Judge TP:  ${JUDGE_TP}"
echo "GPUs:      ${NGPUS}"
echo ""

sbatch --gres=gpu:"${NGPUS}" run_tau2bench.slurm
