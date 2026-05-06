"""
Usage: python experiments/06_diagnostic.py

From K=8 pilot samples per problem:
1. Estimate p̂ and κ̂ from pilot samples
2. Use exact discrete formula to predict gain from best-of-16
3. Predict: "sample more" if predicted gain > threshold, else "stop"
4. Compare prediction against actual gain from full samples
5. Report precision/recall of the diagnostic

Uses multiprocessing for per-problem parallelism.
"""

import sys
import json
import hashlib
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import MODELS, N_SAMPLES, PILOT_K, RESULTS_DIR, RAW_DIR
from src.diagnostic import estimate_from_pilot, GAIN_THRESHOLD

MEASUREMENTS_DIR = RESULTS_DIR / "measurements"
ACTUALS_DIR = RESULTS_DIR / "actuals"


def _cache_path(model_nim, problem, sample_idx):
    h = hashlib.md5(f"{model_nim}:{problem}:{sample_idx}".encode()).hexdigest()
    return RAW_DIR / f"{h}.json"


def _eval_one(args):
    """Worker: evaluate diagnostic for one problem."""
    prob_idx, label, model_nim, problem_text, ground_truth, pilot_k, p_full, kappa_full, actual_acc = args

    pilot_responses = []
    for s in range(pilot_k):
        p = _cache_path(model_nim, problem_text, s)
        if p.exists():
            try:
                with open(p) as fh:
                    pilot_responses.append(json.load(fh))
            except (json.JSONDecodeError, IOError):
                continue

    if len(pilot_responses) < pilot_k:
        return None

    p_hat, kappa_hat, recommendation = estimate_from_pilot(
        pilot_responses, ground_truth, K=pilot_k
    )

    actual_gain = actual_acc - p_full
    predicted_positive = (recommendation == "sample_more")
    actual_positive = (actual_gain > GAIN_THRESHOLD)

    return {
        "model": label, "prob_idx": prob_idx,
        "p_hat": p_hat, "kappa_hat": kappa_hat,
        "kappa_full": kappa_full,
        "recommendation": recommendation, "p_full": p_full,
        "actual_gain": actual_gain,
        "predicted_positive": predicted_positive,
        "actual_positive": actual_positive,
    }


def load_problems():
    jsonl_path = PROJECT_ROOT / "data" / "math500.jsonl"
    problems = []
    if jsonl_path.exists():
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if "problem" in rec and "answer" in rec:
                    problems.append(rec)
    return problems


def run_diagnostic():
    problems = load_problems()
    if not problems:
        print("No MATH-500 problems found.")
        return

    MEASUREMENTS_DIR.mkdir(parents=True, exist_ok=True)
    n_workers = min(cpu_count(), 4)
    print(f"Using {n_workers} worker processes.")

    all_results = []

    for label, model_nim in MODELS.items():
        meas_dir = MEASUREMENTS_DIR / label
        if not meas_dir.exists():
            print(f"{label}: no measurements, skipping")
            continue

        actual_by_prob = {}
        actuals_path = ACTUALS_DIR / label / "validation.json"
        if actuals_path.exists():
            with open(actuals_path) as f:
                val_data = json.load(f)
            for rec in val_data.get("per_problem", []):
                idx = rec["problem_idx"]
                n_key = str(N_SAMPLES)
                if n_key in rec.get("n_values", {}):
                    actual_by_prob[idx] = rec["n_values"][n_key]["actual_acc"]

        work_items = []
        for prob_idx, prob in enumerate(problems):
            meas_path = meas_dir / f"problem_{prob_idx}.json"
            if not meas_path.exists():
                continue
            try:
                with open(meas_path) as f:
                    meas = json.load(f)
            except (json.JSONDecodeError, IOError):
                continue

            p_full = meas["p"]
            kappa_full = meas.get("kappa")
            actual_acc = actual_by_prob.get(prob_idx, p_full)

            work_items.append((
                prob_idx, label, model_nim,
                prob["problem"], prob["answer"],
                PILOT_K, p_full, kappa_full, actual_acc,
            ))

        print(f"\n=== Model: {label} ({model_nim}) ===")

        model_results = []
        with Pool(n_workers) as pool:
            for result in pool.imap_unordered(_eval_one, work_items):
                if result is not None:
                    model_results.append(result)

        tp = sum(1 for r in model_results if r['predicted_positive'] and r['actual_positive'])
        fp = sum(1 for r in model_results if r['predicted_positive'] and not r['actual_positive'])
        tn = sum(1 for r in model_results if not r['predicted_positive'] and not r['actual_positive'])
        fn = sum(1 for r in model_results if not r['predicted_positive'] and r['actual_positive'])
        n_valid = len(model_results)

        precision = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
        recall = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else float('nan'))
        accuracy = (tp + tn) / n_valid if n_valid > 0 else float('nan')

        print(f"  Valid: {n_valid}  TP={tp} FP={fp} TN={tn} FN={fn}")
        print(f"  Precision: {precision:.4f}  Recall: {recall:.4f}  F1: {f1:.4f}  Accuracy: {accuracy:.4f}")

        all_results.extend(model_results)

    if all_results:
        total_tp = sum(1 for r in all_results if r['predicted_positive'] and r['actual_positive'])
        total_fp = sum(1 for r in all_results if r['predicted_positive'] and not r['actual_positive'])
        total_tn = sum(1 for r in all_results if not r['predicted_positive'] and not r['actual_positive'])
        total_fn = sum(1 for r in all_results if not r['predicted_positive'] and r['actual_positive'])
        n_total = len(all_results)

        agg_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else float('nan')
        agg_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else float('nan')
        agg_f1 = (2 * agg_prec * agg_rec / (agg_prec + agg_rec) if (agg_prec + agg_rec) > 0 else float('nan'))

        print(f"\n=== Aggregate ===")
        print(f"  Total: {n_total}  TP={total_tp} FP={total_fp} TN={total_tn} FN={total_fn}")
        print(f"  Precision: {agg_prec:.4f}  Recall: {agg_rec:.4f}  F1: {agg_f1:.4f}")

        informative = [r for r in all_results if 0 < r['p_hat'] < 1]
        inf_tp = sum(1 for r in informative if r['predicted_positive'] and r['actual_positive'])
        inf_fp = sum(1 for r in informative if r['predicted_positive'] and not r['actual_positive'])
        inf_tn = sum(1 for r in informative if not r['predicted_positive'] and not r['actual_positive'])
        inf_fn = sum(1 for r in informative if not r['predicted_positive'] and r['actual_positive'])
        inf_prec = inf_tp / (inf_tp + inf_fp) if (inf_tp + inf_fp) > 0 else float('nan')
        inf_rec = inf_tp / (inf_tp + inf_fn) if (inf_tp + inf_fn) > 0 else float('nan')
        inf_f1 = 2*inf_prec*inf_rec/(inf_prec+inf_rec) if (inf_prec+inf_rec) > 0 else float('nan')

        edge_p0 = [r for r in all_results if r['p_hat'] == 0.0]
        edge_p1 = [r for r in all_results if r['p_hat'] == 1.0]
        p0_fn = sum(1 for r in edge_p0 if not r['predicted_positive'] and r['actual_positive'])
        p1_fn = sum(1 for r in edge_p1 if not r['predicted_positive'] and r['actual_positive'])

        print(f"\n  --- Informative pilots (0 < p_hat < 1) ---")
        print(f"  N={len(informative)}  TP={inf_tp} FP={inf_fp} TN={inf_tn} FN={inf_fn}")
        print(f"  Precision: {inf_prec:.4f}  Recall: {inf_rec:.4f}  F1: {inf_f1:.4f}")
        print(f"\n  --- Edge cases ---")
        print(f"  p_hat=0: {len(edge_p0)} problems, {p0_fn} FN")
        print(f"  p_hat=1: {len(edge_p1)} problems, {p1_fn} FN")

        summary = {
            "pilot_K": PILOT_K,
            "n_total": n_total,
            "aggregate": {"TP": total_tp, "FP": total_fp, "TN": total_tn, "FN": total_fn,
                          "precision": agg_prec, "recall": agg_rec, "f1": agg_f1},
            "informative": {"n": len(informative), "TP": inf_tp, "FP": inf_fp, "TN": inf_tn, "FN": inf_fn,
                            "precision": inf_prec, "recall": inf_rec, "f1": inf_f1},
            "edge_p0": {"n": len(edge_p0), "FN": p0_fn},
            "edge_p1": {"n": len(edge_p1), "FN": p1_fn},
            "per_problem": all_results,
        }
        out_path = MEASUREMENTS_DIR / "diagnostic_results.json"
        with open(out_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    run_diagnostic()
