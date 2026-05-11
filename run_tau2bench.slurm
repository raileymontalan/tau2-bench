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
#   Default: 8004
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
#   Default: 8005
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
# MAX_CONCURRENCY
#   Concurrent simulations. vLLM batches internally so raising this doesn't
#   overwhelm the GPU — it just keeps the queue fuller.
#   Default: 4
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
#SBATCH --nodelist=smc-pod-[3,5,8]
#SBATCH --output=logs/tau2bench-%j.out
#SBATCH --error=logs/tau2bench-%j.out

cd $SLURM_SUBMIT_DIR

source .venv/bin/activate
export PATH="$HOME/.local/bin:$PATH"

SITE_PACKAGES=$(.venv/bin/python -c 'import site; print(site.getsitepackages()[0])')
export LD_LIBRARY_PATH="${SITE_PACKAGES}/nvidia/cudnn/lib:${SITE_PACKAGES}/nvidia/cusparselt/lib:${SITE_PACKAGES}/nvidia/nccl/lib:${SITE_PACKAGES}/nvidia/nvshmem/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Validate flashinfer_cubin version attribute — broken install causes vLLM crash at import
# flashinfer_cubin has no prebuilt wheels for cu128/torch2.10 — remove it so
# flashinfer falls back to JIT mode (compiles kernels at runtime instead).
if python3 -c "import flashinfer_cubin" 2>/dev/null; then
    CUBIN_DIR=$(.venv/bin/python -c "import site; print(site.getsitepackages()[0])")/flashinfer_cubin
    echo "Removing flashinfer_cubin to force JIT mode..."
    rm -rf "${CUBIN_DIR}" "${CUBIN_DIR}"-*.dist-info 2>/dev/null || true
fi

mkdir -p logs


free_port() { python3 -c "import socket; s=socket.socket(); s.bind(('',0)); p=s.getsockname()[1]; s.close(); print(p)"; }

MODEL=${MODEL:-"Qwen/Qwen3-32B"}
MODEL_TP=${MODEL_TP:-"1"}
MODEL_PORT=${MODEL_PORT:-$(free_port)}
ENABLE_THINKING=${ENABLE_THINKING:-"true"}

JUDGE_MODEL=${JUDGE_MODEL:-"openai/gpt-oss-120b"}
JUDGE_TP=${JUDGE_TP:-"1"}
JUDGE_PORT=${JUDGE_PORT:-$(free_port)}

DOMAIN=${DOMAIN:-"all"}
NUM_TRIALS=${NUM_TRIALS:-"1"}
NUM_TASKS=${NUM_TASKS:-""}
MAX_CONCURRENCY=${MAX_CONCURRENCY:-"4"}
TASK_TIMEOUT=${TASK_TIMEOUT:-"300"}
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

# If VISIBLE_DEVICES wasn't populated (no UUID mapping), split CUDA_VISIBLE_DEVICES directly
if [ ${#VISIBLE_DEVICES[@]} -eq 0 ] && [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    IFS=',' read -ra VISIBLE_DEVICES <<< "$CUDA_VISIBLE_DEVICES"
fi

GPU0=${VISIBLE_DEVICES[0]:-0}
GPU1=${VISIBLE_DEVICES[1]:-1}
GPU2=${VISIBLE_DEVICES[2]:-2}

THINKING_FLAG=""
if [[ "$(echo "$ENABLE_THINKING" | tr '[:upper:]' '[:lower:]')" == "false" ]]; then
    THINKING_FLAG="--chat-template-kwargs {\"enable_thinking\": false}"
fi

# Start agent model vLLM server
echo "Starting agent vLLM server: ${MODEL} on port ${MODEL_PORT} (TP=${MODEL_TP})..."
CUDA_VISIBLE_DEVICES=${GPU0} .venv/bin/vllm serve "${MODEL}" \
    --port "${MODEL_PORT}" \
    --tensor-parallel-size "${MODEL_TP}" \
    --dtype bfloat16 \
    --enable-prefix-caching \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --attention-backend FLASH_ATTN \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.85 \
    ${THINKING_FLAG} &
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
    CUDA_VISIBLE_DEVICES=${GPU1} .venv/bin/vllm serve "${JUDGE_MODEL}" \
        --port "${JUDGE_PORT}" \
        --tensor-parallel-size "${JUDGE_TP}" \
        --dtype bfloat16 \
        --enable-prefix-caching \
        --max-model-len 32768 \
        --served-model-name "$(echo ${JUDGE_MODEL} | sed 's|openai/||')" &
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
        "--agent-llm"       "openai/${MODEL}"
        "--user-llm"        "openai/${MODEL}"
        "--num-trials"      "${NUM_TRIALS}"
        "--save-to"         "${OUTPUT}_${CURRENT_DOMAIN}"
        "--max-concurrency" "${MAX_CONCURRENCY}"
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

# Print and save results summary for each domain
echo ""
echo "=== Results Summary ==="
for CURRENT_DOMAIN in "${DOMAINS_TO_RUN[@]}"; do
    echo ""
    echo "--- Domain: ${CURRENT_DOMAIN} ---"
    RESULTS_FILE="data/simulations/${OUTPUT}_${CURRENT_DOMAIN}/results.json"
    SUMMARY_FILE="data/simulations/${OUTPUT}_${CURRENT_DOMAIN}/summary.txt"
    if [ -f "${RESULTS_FILE}" ]; then
        echo "Loading results from ${RESULTS_FILE}..."
        uv run python3 - "${RESULTS_FILE}" <<'PYEOF' | tee "${SUMMARY_FILE}"
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
if m.total_write_actions > 0:
    print(f"Write Acts:  {m.correct_write_actions}/{m.total_write_actions} ({m.correct_write_actions/m.total_write_actions*100:.1f}%)")
db_total = m.db_match_count + m.db_mismatch_count
if db_total > 0:
    print(f"DB Match:    {m.db_match_count}/{db_total} ({m.db_match_count/db_total*100:.1f}%)")
PYEOF
        echo "Summary saved to ${SUMMARY_FILE}"
    else
        echo "No results file found: ${RESULTS_FILE}"
        echo "Skipping domain ${CURRENT_DOMAIN}."
    fi
done

exit $FINAL_EXIT
