"""
Answer extraction audit: sample problems from models with highest phase
violation rates, compare extracted answers against ground truth, flag
suspicious mismatches.

Usage: python scripts/audit_answers.py [--n_samples 50] [--models 3B,8B,Gemma3n2B]
"""
import sys
import json
import random
import argparse
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import MODELS, N_SAMPLES, RESULTS_DIR
from src.nim_client import _cache_path
from src.utils import extract_boxed_answer, normalize_answer, is_equiv, check_correctness


def load_problems():
    jsonl_path = PROJECT_ROOT / "data" / "math500.jsonl"
    problems = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "problem" in rec and "answer" in rec:
                problems.append(rec)
    return problems


def find_worst_models(n_models=3):
    phase_path = RESULTS_DIR / "measurements" / "phase_transition_summary.json"
    if not phase_path.exists():
        return list(MODELS.keys())[:n_models]
    with open(phase_path) as f:
        data = json.load(f)
    violations = {}
    per_model = data.get("per_model", {})
    if isinstance(per_model, dict):
        for label, stats in per_model.items():
            violations[label] = stats.get("violations", 0)
    else:
        for rec in per_model:
            label = rec.get("model", rec.get("label"))
            violations[label] = rec.get("violations", 0)
    sorted_models = sorted(violations, key=violations.get, reverse=True)
    return sorted_models[:n_models]


def audit_model(label, model_nim, problems, n_audit=50, rng=None):
    if rng is None:
        rng = random.Random(42)

    meas_dir = RESULTS_DIR / "measurements" / label
    indices = list(range(len(problems)))
    rng.shuffle(indices)

    results = []
    for prob_idx in indices:
        if len(results) >= n_audit:
            break
        prob = problems[prob_idx]
        meas_path = meas_dir / f"problem_{prob_idx}.json"
        if not meas_path.exists():
            continue

        with open(meas_path) as f:
            meas = json.load(f)

        ground_truth = prob["answer"]
        gt_norm = normalize_answer(ground_truth)

        sample_indices = list(range(N_SAMPLES))
        rng.shuffle(sample_indices)
        for s_idx in sample_indices[:3]:
            cache_file = _cache_path(model_nim, prob["problem"], s_idx)
            if not cache_file.exists():
                continue
            with open(cache_file) as f:
                resp = json.load(f)

            content = resp["content"]
            is_correct, extracted = check_correctness(content, ground_truth)
            extracted_norm = normalize_answer(extracted) if extracted else None

            flag = ""
            if extracted is None:
                flag = "NO_ANSWER"
            elif is_correct and gt_norm != extracted_norm:
                flag = "SOFT_MATCH"
            elif not is_correct and extracted is not None:
                flag = "REJECTED"

            results.append({
                "prob_idx": prob_idx,
                "sample_idx": s_idx,
                "level": prob.get("level"),
                "ground_truth": ground_truth,
                "gt_normalized": gt_norm,
                "extracted": extracted,
                "extracted_normalized": extracted_norm,
                "is_correct": is_correct,
                "flag": flag,
                "p": meas["p"],
            })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model labels (default: 3 worst)")
    args = parser.parse_args()

    problems = load_problems()
    if not problems:
        print("No problems found.")
        return

    if args.models:
        model_labels = args.models.split(",")
    else:
        model_labels = find_worst_models(3)

    print(f"Auditing models: {model_labels}")
    print(f"Sampling {args.n_samples} problems per model")
    print()

    all_flags = defaultdict(int)

    for label in model_labels:
        if label not in MODELS:
            print(f"  {label}: not in config, skipping")
            continue
        model_nim = MODELS[label]
        results = audit_model(label, model_nim, problems, args.n_samples)

        n_no_answer = sum(1 for r in results if r["flag"] == "NO_ANSWER")
        n_soft = sum(1 for r in results if r["flag"] == "SOFT_MATCH")
        n_rejected = sum(1 for r in results if r["flag"] == "REJECTED")
        n_exact = sum(1 for r in results if r["is_correct"] and r["flag"] == "")

        print(f"=== {label} ({len(results)} samples) ===")
        print(f"  Exact match:  {n_exact}")
        print(f"  Soft match:   {n_soft} (correct via sympy, different string)")
        print(f"  Rejected:     {n_rejected} (extracted but marked wrong)")
        print(f"  No answer:    {n_no_answer} (no \\boxed{{}} found)")
        print()

        all_flags["exact"] += n_exact
        all_flags["soft"] += n_soft
        all_flags["rejected"] += n_rejected
        all_flags["no_answer"] += n_no_answer

        if n_rejected > 0:
            print(f"  --- Rejected samples (potential false negatives) ---")
            for r in results:
                if r["flag"] == "REJECTED":
                    print(f"  prob={r['prob_idx']} sample={r['sample_idx']} "
                          f"gt=\"{r['gt_normalized']}\" ext=\"{r['extracted_normalized']}\"")
            print()

        if n_soft > 0:
            print(f"  --- Soft matches (verify these are truly equivalent) ---")
            for r in results:
                if r["flag"] == "SOFT_MATCH":
                    print(f"  prob={r['prob_idx']} sample={r['sample_idx']} "
                          f"gt=\"{r['gt_normalized']}\" ext=\"{r['extracted_normalized']}\"")
            print()

    total = sum(all_flags.values())
    if total > 0:
        print(f"=== AGGREGATE ({total} samples) ===")
        for k, v in all_flags.items():
            print(f"  {k}: {v} ({100*v/total:.1f}%)")


if __name__ == "__main__":
    main()
