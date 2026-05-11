#!/bin/bash
#SBATCH --job-name=tau2-summary
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=logs/tau2-summary-%j.out
#SBATCH --error=logs/tau2-summary-%j.out

cd $SLURM_SUBMIT_DIR
source .venv/bin/activate

OUTPUT=${OUTPUT:-"Qwen3-32B"}
DOMAINS=${DOMAINS:-"airline retail telecom"}

mkdir -p logs

echo "=== tau2-bench Results Summary ==="
echo "Output prefix: ${OUTPUT}"
echo ""

for CURRENT_DOMAIN in ${DOMAINS}; do
    echo "--- Domain: ${CURRENT_DOMAIN} ---"
    RESULTS_FILE="data/simulations/${OUTPUT}_${CURRENT_DOMAIN}/results.json"
    SUMMARY_FILE="data/simulations/${OUTPUT}_${CURRENT_DOMAIN}/summary.txt"
    if [ -f "${RESULTS_FILE}" ]; then
        echo "Loading results from ${RESULTS_FILE}..."
        python3 - "${RESULTS_FILE}" <<'PYEOF' | tee "${SUMMARY_FILE}"
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
    echo ""
done
