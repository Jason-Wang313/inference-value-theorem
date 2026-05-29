"""
Adaptive compute-allocation experiment.

Compares fixed-budget sample allocation policies on held-out response pools:
uniform, p-only, AUC/kappa, moment-law, and oracle marginal allocation.
The pilot split is used only to estimate allocation priorities; reported
accuracy is evaluated on held-out samples under identical total budgets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
except Exception:  # pragma: no cover
    Ridge = None
    make_pipeline = None
    StandardScaler = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import RESULTS_DIR  # noqa: E402


MEASUREMENTS_DIR = RESULTS_DIR / "measurements"
OUT_DIR = RESULTS_DIR / "du_aligned"
TABLE_DIR = OUT_DIR / "tables"
FIGURE_DIR = OUT_DIR / "figures"

DEFAULT_MODELS = ["3B", "70B", "Qwen397B", "Super120B", "MistralSmall119B"]
DEFAULT_MEAN_BUDGETS = [1, 2, 4, 8, 16, 32, 48, 64, 96, 128]


def stable_seed(*parts: object) -> int:
    text = "|".join(str(p) for p in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16) % (2**32)


def load_measurements(models: list[str]) -> dict[str, list[dict]]:
    data: dict[str, list[dict]] = {}
    for model in models:
        rows = []
        for path in sorted((MEASUREMENTS_DIR / model).glob("problem_*.json")):
            try:
                rec = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if len(rec.get("all_scores", [])) != len(rec.get("all_correct", [])):
                continue
            rec["model_key"] = model
            rows.append(rec)
        if rows:
            data[model] = rows
    return data


def train_split(problem_idx: int) -> bool:
    return problem_idx % 5 != 0


def exact_curve(scores: np.ndarray, correct: np.ndarray, n_values: list[int]) -> dict[int, float]:
    scores = np.asarray(scores, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    n = len(scores)
    if n == 0:
        return {int(N): float("nan") for N in n_values}
    p = float(np.mean(correct))
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_correct = correct[order]
    groups = []
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_scores[j] == sorted_scores[i]:
            j += 1
        k_g = j - i
        c_g = int(np.sum(sorted_correct[i:j]))
        if c_g:
            groups.append((i + 1, j, k_g, c_g))
        i = j
    out = {}
    for N in n_values:
        N = int(N)
        if N == 1:
            out[N] = p
            continue
        if p == 0.0:
            out[N] = 0.0
            continue
        f = 0.0
        for r_min, r_max, k_g, c_g in groups:
            mass = (r_max / n) ** N - ((r_min - 1) / n) ** N
            f += (c_g / k_g) * mass
        out[N] = float(f)
    return out


def auc_score(scores: np.ndarray, correct: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    pos = scores[correct]
    neg = scores[~correct]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    wins = 0.0
    total = 0
    for s in pos:
        wins += float(np.sum(s > neg)) + 0.5 * float(np.sum(s == neg))
        total += len(neg)
    return float(wins / total) if total else 0.5


def _features(p: float, kappa: float, n: int) -> list[float]:
    logn = math.log(max(n, 1))
    return [p, kappa, n, logn, p * kappa, p * logn, kappa * logn]


def train_auc_predictors(data: dict[str, list[dict]], max_n: int) -> dict[str, object]:
    predictors: dict[str, object] = {}
    if Ridge is None or make_pipeline is None or StandardScaler is None:
        return predictors
    n_values = list(range(1, max_n + 1))
    for model, rows in data.items():
        X, y = [], []
        for rec in rows:
            if not train_split(int(rec["problem_idx"])):
                continue
            scores = np.asarray(rec["all_scores"], dtype=float)
            correct = np.asarray(rec["all_correct"], dtype=bool)
            usable_n = [n for n in n_values if n <= len(scores)]
            if not usable_n:
                continue
            p = float(np.mean(correct))
            kappa = auc_score(scores, correct)
            curve = exact_curve(scores, correct, usable_n)
            for n in usable_n:
                X.append(_features(p, kappa, n))
                y.append(curve[n])
        if len(X) >= 20:
            model_pipe = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            model_pipe.fit(np.asarray(X, dtype=float), np.asarray(y, dtype=float))
            predictors[model] = model_pipe
    return predictors


def p_only_curve(p: float, max_n: int) -> np.ndarray:
    ns = np.arange(1, max_n + 1, dtype=float)
    return 1.0 - np.power(1.0 - np.clip(p, 0.0, 1.0), ns)


def auc_curve(model_pipe: object | None, p: float, kappa: float, max_n: int) -> np.ndarray:
    if model_pipe is None:
        return p_only_curve(p, max_n)
    X = np.asarray([_features(p, kappa, n) for n in range(1, max_n + 1)], dtype=float)
    pred = np.asarray(model_pipe.predict(X), dtype=float)
    pred = np.maximum.accumulate(np.clip(pred, 0.0, 1.0))
    return pred


def allocate_greedy(pred_curves: list[np.ndarray], total_budget: int) -> list[int]:
    n_problems = len(pred_curves)
    alloc = np.ones(n_problems, dtype=int)
    remaining = max(0, int(total_budget) - n_problems)
    maxes = np.asarray([len(c) for c in pred_curves], dtype=int)
    while remaining > 0:
        gains = np.full(n_problems, -np.inf, dtype=float)
        for i, curve in enumerate(pred_curves):
            if alloc[i] < maxes[i]:
                gains[i] = curve[alloc[i]] - curve[alloc[i] - 1]
        best = int(np.argmax(gains))
        if not np.isfinite(gains[best]) or gains[best] <= 0:
            break
        alloc[best] += 1
        remaining -= 1
    return alloc.tolist()


def allocate_uniform(n_problems: int, total_budget: int, max_n: int) -> list[int]:
    alloc = np.ones(n_problems, dtype=int)
    remaining = max(0, int(total_budget) - n_problems)
    idx = 0
    while remaining > 0 and np.any(alloc < max_n):
        if alloc[idx] < max_n:
            alloc[idx] += 1
            remaining -= 1
        idx = (idx + 1) % n_problems
    return alloc.tolist()


def run_allocation(
    data: dict[str, list[dict]],
    pilot_k: int,
    seeds: int,
    mean_budgets: list[int],
) -> tuple[pd.DataFrame, dict]:
    max_possible = max((len(rec["all_scores"]) - pilot_k for rows in data.values() for rec in rows), default=0)
    max_budget = min(max(mean_budgets), max_possible)
    mean_budgets = [b for b in mean_budgets if 1 <= b <= max_budget]
    predictors = train_auc_predictors(data, max_budget)
    rows = []

    for model, records in sorted(data.items()):
        test_records = [r for r in records if not train_split(int(r["problem_idx"])) and len(r["all_scores"]) > pilot_k + 1]
        if not test_records:
            continue
        for seed_id in range(seeds):
            eval_items = []
            for rec in test_records:
                scores = np.asarray(rec["all_scores"], dtype=float)
                correct = np.asarray(rec["all_correct"], dtype=bool)
                rng = np.random.default_rng(stable_seed("allocation", seed_id, model, rec["problem_idx"], pilot_k))
                perm = rng.permutation(len(scores))
                pilot_idx = perm[:pilot_k]
                held_idx = perm[pilot_k:]
                held_max = min(max_budget, len(held_idx))
                if held_max < 1:
                    continue

                pilot_scores = scores[pilot_idx]
                pilot_correct = correct[pilot_idx]
                held_scores = scores[held_idx]
                held_correct = correct[held_idx]
                n_values = list(range(1, held_max + 1))
                pilot_ns = sorted(set(min(n, pilot_k) for n in n_values))
                pilot_curve_lookup = exact_curve(pilot_scores, pilot_correct, pilot_ns)
                pilot_curve = np.asarray([pilot_curve_lookup[min(n, pilot_k)] for n in n_values], dtype=float)
                pilot_curve = np.maximum.accumulate(pilot_curve)
                held_curve_lookup = exact_curve(held_scores, held_correct, n_values)
                held_curve = np.asarray([held_curve_lookup[n] for n in n_values], dtype=float)
                p_hat = float(np.mean(pilot_correct))
                kappa_hat = auc_score(pilot_scores, pilot_correct)
                eval_items.append(
                    {
                        "problem_idx": int(rec["problem_idx"]),
                        "held_curve": held_curve,
                        "moment_curve": pilot_curve,
                        "p_curve": p_only_curve(p_hat, held_max),
                        "auc_curve": auc_curve(predictors.get(model), p_hat, kappa_hat, held_max),
                        "oracle_curve": held_curve,
                        "max_n": held_max,
                    }
                )
            if not eval_items:
                continue

            n_problems = len(eval_items)
            for mean_budget in mean_budgets:
                total_budget = n_problems * int(mean_budget)
                policy_curves = {
                    "uniform": None,
                    "p_based": [x["p_curve"] for x in eval_items],
                    "auc_kappa_based": [x["auc_curve"] for x in eval_items],
                    "moment_law": [x["moment_curve"] for x in eval_items],
                    "oracle": [x["oracle_curve"] for x in eval_items],
                }
                for policy, curves in policy_curves.items():
                    if policy == "uniform":
                        alloc = allocate_uniform(n_problems, total_budget, min(x["max_n"] for x in eval_items))
                    else:
                        alloc = allocate_greedy(curves or [], total_budget)
                    actuals = []
                    for item, n_alloc in zip(eval_items, alloc):
                        n = max(1, min(int(n_alloc), len(item["held_curve"])))
                        actuals.append(float(item["held_curve"][n - 1]))
                    rows.append(
                        {
                            "model": model,
                            "seed": seed_id,
                            "policy": policy,
                            "mean_budget": int(mean_budget),
                            "total_budget": int(total_budget),
                            "accuracy": float(np.mean(actuals)),
                            "mean_allocated_samples": float(np.mean(alloc)),
                            "num_problems": n_problems,
                            "pilot_k": int(pilot_k),
                        }
                    )

    df = pd.DataFrame(rows)
    summary: dict[str, object] = {
        "status": "completed" if not df.empty else "empty",
        "pilot_k": pilot_k,
        "seeds": seeds,
        "models": sorted(data.keys()),
        "mean_budgets": mean_budgets,
        "budget_note": "Budgets are evaluated post-pilot on held-out response pools; pilot samples estimate policy priorities.",
    }
    if not df.empty:
        overall = df.groupby(["policy", "mean_budget"]).agg(accuracy=("accuracy", "mean")).reset_index()
        summary["policies"] = sorted(df["policy"].unique())
        summary["best_by_budget"] = (
            overall.sort_values(["mean_budget", "accuracy"], ascending=[True, False])
            .groupby("mean_budget")
            .head(1)
            .to_dict(orient="records")
        )
        n48 = overall[overall["mean_budget"] == min(48, max(mean_budgets))]
        if not n48.empty:
            summary["ranking_at_reference_budget"] = n48.sort_values("accuracy", ascending=False).to_dict(orient="records")
    return df, summary


def write_outputs(df: pd.DataFrame, summary: dict) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLE_DIR / "adaptive_allocation.csv", index=False)
    (OUT_DIR / "adaptive_allocation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plt.figure(figsize=(7.4, 4.8))
    if not df.empty:
        overall = (
            df.groupby(["policy", "mean_budget"])
            .agg(accuracy=("accuracy", "mean"))
            .reset_index()
            .sort_values(["policy", "mean_budget"])
        )
        for policy, sub in overall.groupby("policy"):
            plt.plot(sub["mean_budget"], sub["accuracy"], marker="o", label=policy.replace("_", " "))
    plt.xscale("log", base=2)
    plt.xlabel("Mean samples per problem")
    plt.ylabel("Held-out best-of-N accuracy")
    plt.title("Adaptive allocation under fixed inference budget")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "adaptive_allocation_accuracy.pdf")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run adaptive compute-allocation experiment.")
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--pilot-k", type=int, default=16)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--mean-budgets", nargs="*", type=int, default=DEFAULT_MEAN_BUDGETS)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading measurements...")
    data = load_measurements(args.models)
    print(f"Loaded {sum(len(v) for v in data.values())} records across {len(data)} models.")
    df, summary = run_allocation(data, args.pilot_k, args.seeds, args.mean_budgets)
    write_outputs(df, summary)
    print(f"Done. Rows: {len(df)}. Output: {TABLE_DIR / 'adaptive_allocation.csv'}")


if __name__ == "__main__":
    main()
