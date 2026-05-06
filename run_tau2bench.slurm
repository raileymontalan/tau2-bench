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
# DOMAIN
#   Domain(s) to evaluate. One of:
#     airline          — 50 tasks  (base split)
#     retail           — 114 tasks (base split)
#     telecom          — 114 tasks (base split; full set has 2285)
#     mock             — 10 tasks  (smoke test only, not for real evaluation)
#     banking_knowledge — requires `uv sync --extra knowledge` and --retrieval-config
#     all              — runs airline + retail + telecom sequentially (no banking_knowledge)
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
# TENSOR_PARALLEL_SIZE
#   Number of GPUs for vLLM tensor parallelism. Must match --gres=gpu:N.
#   Default: 1
#
# VLLM_PORT
#   Port for local vLLM server.
#   Default: 8000
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
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --nodelist=smc-pod-[3,5,8]
#SBATCH --output=logs/tau2bench-%j.out
#SBATCH --error=logs/tau2bench-%j.out

cd $SLURM_SUBMIT_DIR

export PATH="$HOME/.local/bin:$PATH"

MODEL=${MODEL:-"aisingapore/Gemma-SEA-LION-v4-27B-IT"}
DOMAIN=${DOMAIN:-"all"}
NUM_TRIALS=${NUM_TRIALS:-"1"}
NUM_TASKS=${NUM_TASKS:-""}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-"1"}
VLLM_PORT=${VLLM_PORT:-"8000"}
OUTPUT=${OUTPUT:-"$(echo ${MODEL} | awk -F/ '{print $NF}')"}

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

# Start vLLM server in background
echo "Starting vLLM server for ${MODEL} on port ${VLLM_PORT}..."
source .venv-vllm/bin/activate
vllm serve "${MODEL}" \
    --port "${VLLM_PORT}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --dtype bfloat16 \
    --enable-prefix-caching &
VLLM_PID=$!
deactivate

# Wait for vLLM to be ready
echo "Waiting for vLLM server to be ready..."
until curl -sf "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; do
    sleep 5
done
echo "vLLM server ready."

export OPENAI_API_BASE="http://localhost:${VLLM_PORT}/v1"
export OPENAI_API_KEY="dummy"

# Run evaluation for each domain
FINAL_EXIT=0
for CURRENT_DOMAIN in "${DOMAINS_TO_RUN[@]}"; do
    echo "=== Evaluating domain: ${CURRENT_DOMAIN} ==="

    tau2_args=(
        "uv" "run" "tau2" "run"
        "--domain"     "${CURRENT_DOMAIN}"
        "--agent-llm"  "openai/${MODEL}"
        "--user-llm"   "openai/${MODEL}"
        "--num-trials" "${NUM_TRIALS}"
        "--save-to"    "${OUTPUT}_${CURRENT_DOMAIN}"
        "--auto-resume"
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

# Shut down vLLM server
echo "Shutting down vLLM server (PID ${VLLM_PID})..."
kill "${VLLM_PID}"

exit $FINAL_EXIT
