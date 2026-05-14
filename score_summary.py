#!/usr/bin/env python3
"""Summarize tau2-bench results by model and domain.

Per-domain rows + aggregate row per model.
Pass columns:
  Pass@1  = fraction of tasks where ≥1 trial passed  (optimistic)
  Pass^1  = per-trial pass rate = avg_reward          (same as Pass@1 for 1-trial runs)
  Pass^2  = fraction of tasks where ≥2 trials passed
  Pass^3  = fraction of tasks where all 3 trials passed  (strict reliability)

Usage:
    python score_summary.py                      # reads data/simulations/
    python score_summary.py data/simulations/
    python score_summary.py data/simulations/Qwen3.6-27B_airline  # single dir
"""

import json
import sys
import io
from collections import defaultdict
from pathlib import Path

DOMAINS = ["airline", "retail", "telecom"]
DOMAIN_TASKS = {"airline": 50, "retail": 114, "telecom": 114}
PASS_THRESHOLD = 0.5
N_TRIALS = 3  # expected trials per task (matches config_vllm.yaml num_trials)


def _pass_metrics_from_json(results_file: Path) -> dict | None:
    """Compute per-domain pass metrics directly from results.json."""
    try:
        data = json.loads(results_file.read_text())
    except Exception as e:
        print(f"  WARNING: could not read {results_file}: {e}", file=sys.stderr)
        return None

    num_trials = data.get("info", {}).get("num_trials", 1)
    sims = data.get("simulations", [])

    # Group by task_id; if extra sims exist (retries), use last num_trials per task
    task_sim_map: dict[str, list[float]] = defaultdict(list)
    for s in sims:
        r = s.get("reward_info", {})
        reward = r.get("reward", 0.0) if isinstance(r, dict) else 0.0
        task_sim_map[s["task_id"]].append(float(reward or 0.0))

    n_tasks = len(task_sim_map)
    if n_tasks == 0:
        return None

    # Per-task: count passing trials (cap at num_trials most recent sims)
    task_n_pass = {
        tid: sum(1 for r in rewards[-num_trials:] if r >= PASS_THRESHOLD)
        for tid, rewards in task_sim_map.items()
    }

    # Aggregate metrics
    total_trials = sum(min(len(v), num_trials) for v in task_sim_map.values())
    total_passing_trials = sum(
        sum(1 for r in v[-num_trials:] if r >= PASS_THRESHOLD)
        for v in task_sim_map.values()
    )

    pass_at = {}   # Pass@k: fraction of tasks with >= k passing trials
    pass_hat = {}  # Pass^k: same definition (fraction of tasks with >= k passing)
    for k in range(1, num_trials + 1):
        n = sum(1 for np_ in task_n_pass.values() if np_ >= k)
        pass_at[k] = n / n_tasks
        pass_hat[k] = n / n_tasks

    # Pass^1 = per-trial pass rate (distinguished from Pass@1 when num_trials > 1)
    pass_hat[1] = total_passing_trials / total_trials if total_trials else 0.0

    return {
        "tasks": n_tasks,
        "avg_reward": total_passing_trials / total_trials if total_trials else 0.0,
        "pass_at": pass_at,
        "pass_hat": pass_hat,
        "num_trials": num_trials,
    }


def load_domain(sim_dir: Path) -> dict | None:
    """Load metrics for a single domain simulation directory."""
    results_file = sim_dir / "results.json"
    if not results_file.exists():
        return None
    try:
        _stdout = sys.stdout
        sys.stdout = io.StringIO()
        from tau2.data_model.simulation import Results
        from tau2.metrics.agent_metrics import compute_metrics
        sys.stdout = _stdout
        results = Results.load(results_file)
        m = compute_metrics(results)
    except Exception as e:
        sys.stdout = sys.__stdout__
        print(f"  WARNING: {sim_dir.name}: {e}", file=sys.stderr)
        return None

    raw = _pass_metrics_from_json(results_file)
    if raw is None:
        return None

    return {
        "tasks": m.total_tasks,
        "simulations": m.total_simulations,
        "avg_reward": m.avg_reward,
        "pass_at": raw["pass_at"],
        "pass_hat": raw["pass_hat"],
        "num_trials": raw["num_trials"],
    }


def find_model_domains(root: Path) -> dict[str, dict[str, Path]]:
    """Group dirs named <MODEL>_<DOMAIN> by model."""
    model_domains: dict[str, dict[str, Path]] = defaultdict(dict)
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        for domain in DOMAINS + ["mock"]:
            if d.name.endswith(f"_{domain}"):
                model = d.name[: -(len(domain) + 1)]
                model_domains[model][domain] = d
                break
    return dict(model_domains)


def _fmt(val, width=7) -> str:
    return f"{val:.3f}".rjust(width) if val is not None else "—".rjust(width)


def _weighted_avg(vals_and_weights: list[tuple[float | None, int]]) -> float | None:
    pairs = [(v, w) for v, w in vals_and_weights if v is not None and w > 0]
    if not pairs:
        return None
    total_w = sum(w for _, w in pairs)
    return sum(v * w for v, w in pairs) / total_w if total_w else None


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/simulations")
    if not root.exists():
        print(f"Directory not found: {root}")
        sys.exit(1)

    # Single dir mode
    if (root / "results.json").exists():
        r = load_domain(root)
        if r:
            print(f"Dir:         {root}")
            print(f"Tasks:       {r['tasks']}")
            print(f"Avg Reward:  {r['avg_reward']:.4f}")
            for k in sorted(set(r["pass_at"]) | set(r["pass_hat"])):
                at = r["pass_at"].get(k)
                hat = r["pass_hat"].get(k)
                print(f"  Pass@{k}: {at:.4f}   Pass^{k}: {hat:.4f}")
        return

    model_domains = find_model_domains(root)
    if not model_domains:
        print(f"No simulation dirs found in {root}")
        sys.exit(1)

    sorted_models = sorted(model_domains.keys())

    # Detect max k across all loaded data
    max_k = 1
    cache: dict[str, dict[str, dict]] = defaultdict(dict)
    for model in sorted_models:
        for domain, sim_dir in model_domains[model].items():
            r = load_domain(sim_dir)
            if r and r["tasks"] > 0:
                cache[model][domain] = r
                max_k = max(max_k, r["num_trials"])
    max_k = max(max_k, N_TRIALS)

    k_values = list(range(1, max_k + 1))

    # ── Header ────────────────────────────────────────────────────────────────
    pass_header = "  ".join(
        f"{'Pass@'+str(k):>7s}  {'Pass^'+str(k):>7s}" for k in k_values
    )
    header = f"{'Model':<25s}  {'Domain':<8s}  {'Tasks':>5s}  {'Avg':>5s}  {pass_header}"
    print(header)
    print("─" * len(header))

    json_out = []

    for model in sorted_models:
        domain_data = cache.get(model, {})
        first_row = True
        model_json = {"model": model, "domains": {}, "aggregate": None}

        # Per-domain rows
        for domain in DOMAINS:
            r = domain_data.get(domain)
            label = model if first_row else ""
            first_row = False

            if r is None:
                pass_vals = "  ".join("—".rjust(7) + "  " + "—".rjust(7) for _ in k_values)
                expected = DOMAIN_TASKS.get(domain, "?")
                print(f"{label:<25s}  {domain:<8s}  {'—':>5s}  {'—':>5s}  {pass_vals}")
                model_json["domains"][domain] = None
            else:
                expected = DOMAIN_TASKS.get(domain, r["tasks"])
                incomplete = "*" if r["tasks"] < expected else " "
                pass_vals = "  ".join(
                    _fmt(r["pass_at"].get(k)) + "  " + _fmt(r["pass_hat"].get(k))
                    for k in k_values
                )
                print(
                    f"{label:<25s}  {domain:<8s}  {r['tasks']:>4d}{incomplete}"
                    f"  {r['avg_reward']:>5.3f}  {pass_vals}"
                )
                model_json["domains"][domain] = {
                    "tasks": r["tasks"],
                    "avg_reward": round(r["avg_reward"], 4),
                    "pass_at": {str(k): round(v, 4) for k, v in r["pass_at"].items()},
                    "pass_hat": {str(k): round(v, 4) for k, v in r["pass_hat"].items()},
                }

        # Aggregate row (weighted by task count across completed domains)
        completed = [(r, r["tasks"]) for r in domain_data.values() if r]
        if completed:
            agg_tasks = sum(w for _, w in completed)
            agg_avg = _weighted_avg([(r["avg_reward"], r["tasks"]) for r, _ in completed])
            agg_pass_at = {
                k: _weighted_avg([(r["pass_at"].get(k), r["tasks"]) for r, _ in completed])
                for k in k_values
            }
            agg_pass_hat = {
                k: _weighted_avg([(r["pass_hat"].get(k), r["tasks"]) for r, _ in completed])
                for k in k_values
            }
            n_domains = len(completed)
            pass_vals = "  ".join(
                _fmt(agg_pass_at.get(k)) + "  " + _fmt(agg_pass_hat.get(k))
                for k in k_values
            )
            print(
                f"{'':25s}  {'TOTAL':8s}  {agg_tasks:>5d}"
                f"  {agg_avg:>5.3f}  {pass_vals}"
                f"  ({n_domains} domains)"
            )
            model_json["aggregate"] = {
                "tasks": agg_tasks,
                "domains_completed": n_domains,
                "avg_reward": round(agg_avg, 4),
                "pass_at": {str(k): round(v, 4) for k, v in agg_pass_at.items() if v is not None},
                "pass_hat": {str(k): round(v, 4) for k, v in agg_pass_hat.items() if v is not None},
            }

        print()
        json_out.append(model_json)

    if any(
        d.get("tasks", 0) < DOMAIN_TASKS.get(dom, 0)
        for m_json in json_out
        for dom, d in m_json["domains"].items()
        if d
        for dom in DOMAIN_TASKS
    ):
        print("* = incomplete (fewer tasks than expected)")
    print()
    print(f"Pass@k = fraction of tasks where ≥k trials passed  (optimistic)")
    print(f"Pass^k = per-trial pass rate (k=1) or fraction with ≥k passes (k>1)")

    # ── Save JSON ──────────────────────────────────────────────────────────────
    out_file = root / "score_summary.json"
    with open(out_file, "w") as f:
        json.dump(json_out, f, indent=2, ensure_ascii=False)
    print(f"\nJSON saved to {out_file}")


if __name__ == "__main__":
    main()
