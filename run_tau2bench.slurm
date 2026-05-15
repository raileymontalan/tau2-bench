#!/bin/bash
#
# tau2-bench SLURM evaluation script
#
# Usage:
#   sbatch run_evaluation.slurm
#
# Override any variable at submit time:
#   MODEL=aisingapore/Other-Model DOMAIN=retail sbatch run_evaluation.slurm
#   DOMAIN=all NUM_TASKS=10 sbatch run_evaluation.slurm
#
# ── Variable reference ────────────────────────────────────────────────────────
#
# MODEL
#   HuggingFace model ID to evaluate.
#   Default: aisingapore/Gemma-SEA-LION-v4-27B-IT
#
# MODEL_TP
#   Tensor parallel size for the agent model (number of GPUs).
#   Qwen3-32B in BF16 is ~64GB — fits on 1x 80GB GPU.
#   Default: 1
#
# MODEL_PORT
#   Port for local vLLM server.
#   Default: auto-assigned free port
#
# JUDGE_MODEL
#   HuggingFace model ID for NL assertions judge. Served on a separate vLLM
#   instance (1 GPU). Set to empty string to disable and use default routing.
#   Default: openai/gpt-oss-120b
#
# JUDGE_TP
#   Tensor parallel size for the judge model.
#   gpt-oss-120b is sparse MoE — fits on 1x 140GB GPU. TP=2 triggers allreduce_rms
#   custom CUDA kernel that segfaults on CUDA 12.4 driver. Use TP=1.
#   Default: 1
#
# JUDGE_PORT
#   Port for judge vLLM server.
#   Default: auto-assigned free port
#
# DOMAIN
#   Domain(s) to evaluate. One of:
#     airline          — 50 tasks  (base split)
#     retail           — 114 tasks (base split)
#     telecom          — 114 tasks (base split; full set has 2285)
#     mock             — 10 tasks  (smoke test only, not for real evaluation)
#     all              — runs airline + retail + telecom sequentially
#   Default: all
#
# NUM_TRIALS
#   Evaluation trials per task. Higher = more robust pass@k estimate.
#   Default: 1
#
# NUM_TASKS
#   Cap tasks per domain. Omit (leave empty) to run the full domain.
#   Default: (empty — run all tasks)
#
# MAX_CONCURRENCY
#   Concurrent simulations. vLLM batches internally so raising this doesn't
#   overwhelm the GPU — it just keeps the queue fuller.
#   Default: 3
#
# MAX_STEPS
#   Maximum conversation turns per simulation before it's abandoned.
#   Default: 200
#
# SEED
#   Random seed for task sampling. Leaderboard submissions use 300.
#   Default: 300
#
# TASK_TIMEOUT
#   Max wallclock seconds per individual simulation before it's abandoned.
#   Prevents stuck tasks (e.g. Qwen3 thinking loops) from blocking workers.
#   Default: 300
#
# OUTPUT
#   Name for the output directory under data/simulations/.
#   Default: <model basename>
#
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=tau2bench
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
#SBATCH --nodelist=smc-pod-[14,12,11,10,8,5,3]
#SBATCH --output=logs/tau2bench-%j.out
#SBATCH --error=logs/tau2bench-%j.out

cd $SLURM_SUBMIT_DIR

module use "${MODULEFILES_PATH}"
module swap cuda/13.0.3

source .venv/bin/activate
export PATH="$HOME/.local/bin:$PATH"

SITE_PACKAGES=$(.venv/bin/python -c 'import site; print(site.getsitepackages()[0])')
export LD_LIBRARY_PATH="${SITE_PACKAGES}/nvidia/cudnn/lib:${SITE_PACKAGES}/nvidia/cusparselt/lib:${SITE_PACKAGES}/nvidia/nccl/lib:${SITE_PACKAGES}/nvidia/nvshmem/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# vLLM 0.20.2 bundles vllm.third_party.deep_gemm, making has_deep_gemm() return
# True even when the backend is broken. Disabling via env var is the clean fix.
export VLLM_USE_DEEP_GEMM=0

mkdir -p logs


free_port() { python3 -c "import socket; s=socket.socket(); s.bind(('',0)); p=s.getsockname()[1]; s.close(); print(p)"; }

# Resolve MODEL (env override > config default)
MODEL="${MODEL:-}"
if [ -z "${MODEL}" ]; then
    MODEL=$(python3 -c "
import yaml, pathlib
print(yaml.safe_load(pathlib.Path('config_vllm.yaml').read_text()).get('eval', {}).get('default_model', ''))
")
fi
if [ -z "${MODEL}" ]; then
    echo "ERROR: MODEL not set and no default_model in config_vllm.yaml"
    exit 1
fi

# Load model-specific settings
_model_vars=$(python3 - "$MODEL" <<'PYEOF'
import yaml, pathlib, shlex, sys
cfg = yaml.safe_load(pathlib.Path('config_vllm.yaml').read_text())
model = sys.argv[1]
models = cfg.get('models', {})
if model not in models:
    raise SystemExit(f"ERROR: Model '{model}' not found in config_vllm.yaml.\nAvailable: {list(models.keys())}")
mcfg = models[model]
print(f'_MODEL_TP={shlex.quote(str(mcfg.get("tp", 1)))}')
thinking = mcfg.get("enable_thinking", True)
print(f'_MODEL_THINKING={"true" if thinking else "false"}')
print(f'_TOOL_CALL_PARSER={shlex.quote(str(mcfg.get("tool_call_parser", "")))}')
print(f'_REASONING_PARSER={shlex.quote(str(mcfg.get("reasoning_parser", "")))}')
print(f'_MODEL_PATH={shlex.quote(str(mcfg.get("model_path", model)))}')
PYEOF
) || exit 1
eval "$_model_vars"

MODEL_TP=${MODEL_TP:-${_MODEL_TP:-"1"}}
MODEL_PORT=${MODEL_PORT:-$(free_port)}
ENABLE_THINKING=${ENABLE_THINKING:-${_MODEL_THINKING:-"true"}}
TOOL_CALL_PARSER=${TOOL_CALL_PARSER:-${_TOOL_CALL_PARSER:-""}}
REASONING_PARSER=${REASONING_PARSER:-${_REASONING_PARSER:-""}}
MODEL_PATH=${MODEL_PATH:-${_MODEL_PATH:-"${MODEL}"}}
MODEL_BASENAME="$(echo "${MODEL}" | awk -F/ '{print $NF}')"

# Load judge and eval defaults from config
eval "$(python3 - <<'PYEOF'
import yaml, pathlib, shlex
cfg = yaml.safe_load(pathlib.Path('config_vllm.yaml').read_text())
ecfg = cfg.get('eval', {})
print(f'_JUDGE_MODEL={shlex.quote(str(ecfg.get("judge_model", "openai/gpt-oss-120b")))}')
print(f'_JUDGE_TP={shlex.quote(str(ecfg.get("judge_tp", 1)))}')
v = ecfg.get("domain", "all"); print(f'_DOMAIN={shlex.quote(str(v))}')
v = ecfg.get("num_trials", 1); print(f'_NUM_TRIALS={shlex.quote(str(v))}')
v = ecfg.get("num_tasks", ""); v = "" if v is None else v; print(f'_NUM_TASKS={shlex.quote(str(v))}')
v = ecfg.get("max_concurrency", 3); print(f'_MAX_CONCURRENCY={shlex.quote(str(v))}')
v = ecfg.get("max_steps", 200); print(f'_MAX_STEPS={shlex.quote(str(v))}')
v = ecfg.get("seed", 300); print(f'_SEED={shlex.quote(str(v))}')
v = ecfg.get("task_timeout", 300); print(f'_TASK_TIMEOUT={shlex.quote(str(v))}')
PYEOF
)"

JUDGE_MODEL=${JUDGE_MODEL:-${_JUDGE_MODEL:-"openai/gpt-oss-120b"}}
JUDGE_TP=${JUDGE_TP:-${_JUDGE_TP:-"1"}}
JUDGE_PORT=${JUDGE_PORT:-$(free_port)}

DOMAIN=${DOMAIN:-${_DOMAIN:-"all"}}
NUM_TRIALS=${NUM_TRIALS:-${_NUM_TRIALS:-"1"}}
NUM_TASKS=${NUM_TASKS:-${_NUM_TASKS:-""}}
MAX_CONCURRENCY=${MAX_CONCURRENCY:-${_MAX_CONCURRENCY:-"3"}}
MAX_STEPS=${MAX_STEPS:-${_MAX_STEPS:-"200"}}
SEED=${SEED:-${_SEED:-"300"}}
TASK_TIMEOUT=${TASK_TIMEOUT:-${_TASK_TIMEOUT:-"300"}}
OUTPUT=${OUTPUT:-"${MODEL_BASENAME}"}

# Resolve DOMAIN=all to core text domains
if [ "${DOMAIN}" = "all" ]; then
    DOMAINS_TO_RUN=("airline" "retail" "telecom")
else
    DOMAINS_TO_RUN=("${DOMAIN}")
fi

# GPU UUID -> index mapping
UUIDS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n')
declare -a VISIBLE_DEVICES=()
if [ ! -z "$UUIDS" ]; then
    for UUID in $UUIDS; do
        if [[ $UUID == GPU-* ]] || [[ $UUID == MIG-* ]]; then
            echo "Processing UUID: $UUID"
            ID=$(nvidia-smi --id=$UUID --query-gpu=index --format=csv,noheader)
            if [ ! -z "$ID" ]; then
                VISIBLE_DEVICES+=($ID)
                echo "Mapped UUID $UUID to GPU ID: $ID"
            else
                echo "Could not map UUID $UUID to index. Keeping UUID."
                VISIBLE_DEVICES+=($UUID)
            fi
        else
            VISIBLE_DEVICES+=($UUID)
        fi
    done
    VISIBLE_DEVICES_STR=$( IFS=$','; echo "${VISIBLE_DEVICES[*]}" )
    echo "Visible devices mapped: $VISIBLE_DEVICES_STR"
    export CUDA_VISIBLE_DEVICES=$VISIBLE_DEVICES_STR
fi

# If VISIBLE_DEVICES wasn't populated (no UUID mapping), split CUDA_VISIBLE_DEVICES directly
if [ ${#VISIBLE_DEVICES[@]} -eq 0 ] && [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    IFS=',' read -ra VISIBLE_DEVICES <<< "$CUDA_VISIBLE_DEVICES"
fi

# Partition GPUs: first MODEL_TP for agent, next JUDGE_TP for judge
MODEL_GPUS=$(IFS=','; echo "${VISIBLE_DEVICES[*]:0:${MODEL_TP}}")
MODEL_GPUS=${MODEL_GPUS:-0}
JUDGE_GPUS=$(IFS=','; echo "${VISIBLE_DEVICES[*]:${MODEL_TP}:${JUDGE_TP}}")
JUDGE_GPUS=${JUDGE_GPUS:-1}

THINKING_FLAG=()
if [[ "$(echo "$ENABLE_THINKING" | tr '[:upper:]' '[:lower:]')" == "false" ]]; then
    THINKING_FLAG=("--default-chat-template-kwargs" '{"enable_thinking": false}')
fi

REASONING_PARSER_FLAG=""
if [ -n "${REASONING_PARSER}" ]; then
    REASONING_PARSER_FLAG="--reasoning-parser ${REASONING_PARSER}"
fi

# Start agent model vLLM server
echo "Starting agent vLLM server: ${MODEL} on port ${MODEL_PORT} (TP=${MODEL_TP})..."
CUDA_VISIBLE_DEVICES=${MODEL_GPUS} .venv/bin/vllm serve "${MODEL_PATH}" \
    --port "${MODEL_PORT}" \
    --tensor-parallel-size "${MODEL_TP}" \
    --served-model-name "$(echo "${MODEL}" | awk -F/ '{print $NF}')" \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.85 \
    --enable-prefix-caching \
    --enable-auto-tool-choice \
    --tool-call-parser "${TOOL_CALL_PARSER}" \
    ${REASONING_PARSER_FLAG} \
    "${THINKING_FLAG[@]}" &
MODEL_PID=$!

# Wait for agent model vLLM to be ready
echo "Waiting for model vLLM server to be ready..."
until curl -sf "http://localhost:${MODEL_PORT}/health" > /dev/null 2>&1 && kill -0 "${MODEL_PID}" 2>/dev/null; do
    if ! kill -0 "${MODEL_PID}" 2>/dev/null; then
        echo "ERROR: vLLM server (PID ${MODEL_PID}) died before becoming ready. Exiting."
        exit 1
    fi
    sleep 5
done
echo "Model vLLM server ready."

# Start NL assertions judge model vLLM server
JUDGE_PID=""
if [ -n "${JUDGE_MODEL}" ]; then
    echo "Starting judge vLLM server: ${JUDGE_MODEL} on port ${JUDGE_PORT} (TP=${JUDGE_TP})..."
    CUDA_VISIBLE_DEVICES=${JUDGE_GPUS} .venv/bin/vllm serve "${JUDGE_MODEL}" \
        --port "${JUDGE_PORT}" \
        --tensor-parallel-size "${JUDGE_TP}" \
        --served-model-name "$(echo ${JUDGE_MODEL} | sed 's|openai/||')" \
        --dtype bfloat16 \
        --enable-prefix-caching &
    JUDGE_PID=$!

    # Wait for judge model vLLM to be ready
    echo "Waiting for judge model vLLM server to be ready..."
    until curl -sf "http://localhost:${JUDGE_PORT}/health" > /dev/null 2>&1 && kill -0 "${JUDGE_PID}" 2>/dev/null; do
        if ! kill -0 "${JUDGE_PID}" 2>/dev/null; then
            echo "ERROR: vLLM server (PID ${JUDGE_PID}) died before becoming ready. Exiting."
            exit 1
        fi
        sleep 5
    done
    echo "Judge model vLLM server ready."

    # Tell tau2-bench to use the local judge (openai/ prefix = LiteLLM OpenAI-compatible)
    export TAU2_NL_JUDGE_MODEL="${JUDGE_MODEL}"
    export TAU2_NL_JUDGE_BASE_URL="http://localhost:${JUDGE_PORT}/v1"
fi

export OPENAI_API_BASE="http://localhost:${MODEL_PORT}/v1"
export OPENAI_API_KEY="dummy"

# Run tau2-bench for each domain
FINAL_EXIT=0
for CURRENT_DOMAIN in "${DOMAINS_TO_RUN[@]}"; do
    echo "=== Evaluating domain: ${CURRENT_DOMAIN} ==="

    tau2_args=(
        "uv" "run" "tau2" "run"
        "--domain"          "${CURRENT_DOMAIN}"
        "--agent-llm"       "openai/${MODEL_BASENAME}"
        "--user-llm"        "openai/${MODEL_BASENAME}"
        "--num-trials"      "${NUM_TRIALS}"
        "--save-to"         "${OUTPUT}_${CURRENT_DOMAIN}"
        "--max-concurrency" "${MAX_CONCURRENCY}"
        "--max-steps"       "${MAX_STEPS}"
        "--seed"            "${SEED}"
        "--timeout"         "${TASK_TIMEOUT}"
        "--auto-resume"
        "--log-level"       "INFO"
    )

    if [ ! -z "${NUM_TASKS}" ]; then
        tau2_args+=("--num-tasks" "${NUM_TASKS}")
    fi

    echo "Executing command: ${tau2_args[@]}"
    "${tau2_args[@]}"
    EXIT_CODE=$?

    if [ $EXIT_CODE -ne 0 ]; then
        echo "Domain ${CURRENT_DOMAIN} failed with exit code ${EXIT_CODE}."
        FINAL_EXIT=$EXIT_CODE
    fi
done

# Shut down vLLM servers
echo "Shutting down model vLLM server (PID ${MODEL_PID})..."
kill "${MODEL_PID}"
if [ -n "${JUDGE_PID}" ]; then
    echo "Shutting down judge vLLM server (PID ${JUDGE_PID})..."
    kill "${JUDGE_PID}"
fi

# Print and save results summary
echo ""
echo "=== Results Summary ==="
uv run python3 score_summary.py data/simulations/ 2>&1 | tee "data/simulations/${OUTPUT}_summary.txt"
echo "Summary saved to data/simulations/${OUTPUT}_summary.txt"

exit $FINAL_EXIT
