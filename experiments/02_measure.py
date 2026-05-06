"""
Usage: python experiments/02_measure.py [--model 8B]

For each model (or specified model), for each problem:
1. Load all cached responses from data/raw/
2. Check correctness, compute scores
3. Compute p, kappa, F+, F-
4. Save to results/measurements/{model}/{problem_idx}.json

Uses multiprocessing for per-problem parallelism.
"""

import sys
import json
import csv
import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import MODELS, N_SAMPLES, N_PROBLEMS, RESULTS_DIR
from src.nim_client import _cache_path
from src.scorer import compute_problem_stats


def _measure_one(args):
    """Process a single problem: load cache, compute stats, return result."""
    prob_idx, model_key, model_name, problem_text, ground_truth, level, prob_type, n_samples = args

    responses = []
    for sample_idx in range(n_samples):
        cache_file = _cache_path(model_name, problem_text, sample_idx)
        if not cache_file.exists():
            return None
        try:
            resp = json.loads(cache_file.read_text(encoding="utf-8"))
            responses.append(resp)
        except (json.JSONDecodeError, IOError):
            return None

    try:
        stats = compute_problem_stats(responses, ground_truth)
    except Exception:
        return None

    return {
        "problem_idx": prob_idx,
        "model_key": model_key,
        "model_name": model_name,
        "p": stats["p"],
        "kappa": stats["kappa"],
        "n_correct": stats["n_correct"],
        "n_incorrect": stats["n_incorrect"],
        "scores_correct": stats["scores_correct"],
        "scores_incorrect": stats["scores_incorrect"],
        "all_scores": stats["all_scores"],
        "all_correct": stats["all_correct"],
        "level": level,
        "type": prob_type,
        "ground_truth": ground_truth,
    }


def load_math500(data_path: Path, n_problems: int) -> list[dict]:
    """Load MATH500 problems from JSONL file."""
    problems = []
    if not data_path.exists():
        print(f"ERROR: Data file not found: {data_path}")
        sys.exit(1)

    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "problem" not in obj or "answer" not in obj:
                continue
            problems.append(obj)
            if len(problems) >= n_problems:
                break
    return problems


def main():
    parser = argparse.ArgumentParser(
        description="Compute p, kappa, F+, F- from collected responses."
    )
    parser.add_argument(
        "--model", type=str, default=None, choices=list(MODELS.keys()),
        help="Model to measure (default: all models)",
    )
    parser.add_argument(
        "--n_samples", type=int, default=N_SAMPLES,
        help=f"Number of samples expected per problem (default: {N_SAMPLES})",
    )
    parser.add_argument(
        "--n_problems", type=int, default=N_PROBLEMS,
        help=f"Number of problems (default: {N_PROBLEMS})",
    )
    args = parser.parse_args()

    if args.model:
        model_keys = [args.model]
    else:
        model_keys = list(MODELS.keys())

    n_samples = args.n_samples
    n_problems = args.n_problems

    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "math500.jsonl"
    measurements_dir = RESULTS_DIR / "measurements"

    print(f"Loading MATH500 from {data_path} ...")
    problems = load_math500(data_path, n_problems)
    print(f"Loaded {len(problems)} problems.")

    if len(problems) == 0:
        print("No problems loaded. Exiting.")
        sys.exit(1)

    n_workers = min(cpu_count(), 4)
    print(f"Using {n_workers} worker processes.")

    for model_key in model_keys:
        model_name = MODELS[model_key]
        model_dir = measurements_dir / model_key
        model_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 60}")
        print(f"Measuring model: {model_key} ({model_name})")
        print(f"{'=' * 60}")

        work_items = []
        for prob_idx in range(len(problems)):
            prob = problems[prob_idx]
            work_items.append((
                prob_idx, model_key, model_name,
                prob["problem"], prob["answer"],
                prob.get("level"), prob.get("type"),
                n_samples,
            ))

        summary_rows = []
        n_complete = 0
        n_skipped = 0

        with Pool(n_workers) as pool:
            for i, result in enumerate(pool.imap_unordered(_measure_one, work_items)):
                if (i + 1) % 50 == 0 or (i + 1) == len(work_items):
                    print(f"\r  {model_key}: {i + 1}/{len(work_items)}", end="", flush=True)

                if result is None:
                    n_skipped += 1
                    continue

                n_complete += 1
                output_file = model_dir / f"problem_{result['problem_idx']}.json"
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2)

                summary_rows.append({
                    "problem_idx": result["problem_idx"],
                    "p": result["p"],
                    "kappa": result["kappa"],
                    "n_correct": result["n_correct"],
                    "n_incorrect": result["n_incorrect"],
                    "level": result["level"] if result["level"] is not None else "",
                    "type": result["type"] if result["type"] is not None else "",
                })

        print()

        summary_path = measurements_dir / f"{model_key}_summary.csv"
        if summary_rows:
            fieldnames = ["problem_idx", "p", "kappa", "n_correct", "n_incorrect", "level", "type"]
            with open(summary_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(summary_rows)
            print(f"Saved summary CSV: {summary_path}")

        print(f"\n--- {model_key} Summary ---")
        print(f"Complete problems: {n_complete}")
        print(f"Skipped (missing cache): {n_skipped}")

        if summary_rows:
            p_values = [r["p"] for r in summary_rows]
            kappa_values = [r["kappa"] for r in summary_rows if r["kappa"] is not None]
            print(f"Mean p: {sum(p_values) / len(p_values):.4f}")
            if kappa_values:
                print(f"Mean kappa: {sum(kappa_values) / len(kappa_values):.4f}")
            print(f"Problems with p=0: {sum(1 for p in p_values if p == 0.0)}")
            print(f"Problems with p=1: {sum(1 for p in p_values if p == 1.0)}")
            print(f"Problems with 0<p<1: {sum(1 for p in p_values if 0 < p < 1)}")


if __name__ == "__main__":
    main()
