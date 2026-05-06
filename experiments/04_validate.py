"""
Usage: python experiments/04_validate.py [--model 8B]

For each model, for each problem, for each N:
1. Simulate actual best-of-N 10000 times
2. Compare with theoretical prediction
3. Aggregate results per model, per difficulty level
4. Save to results/actuals/{model}/validation.json
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------------------------------
# sys.path manipulation to import from project root
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import MODELS, EVAL_N_VALUES, RESULTS_DIR
from src.theorem import simulate_best_of_n

try:
    from tqdm import tqdm
except ImportError:
    class tqdm:
        def __init__(self, iterable=None, total=0, desc="", **kwargs):
            self.iterable = iterable
            self.total = total
            self.desc = desc
            self.n = 0
        def __iter__(self):
            for item in self.iterable:
                yield item
                self.n += 1
                print(f"\r{self.desc} {self.n}/{self.total}", end="", flush=True)
            print()
        def __enter__(self):
            return self
        def __exit__(self, *args):
            print()
        def update(self, n=1):
            self.n += n
            print(f"\r{self.desc} {self.n}/{self.total}", end="", flush=True)
        def set_postfix_str(self, s):
            pass


def load_measurement(path: Path) -> dict | None:
    """Load a single problem measurement JSON."""
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def load_prediction(path: Path) -> dict | None:
    """Load a single problem prediction JSON."""
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Compare predicted vs actual best-of-N accuracy."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=list(MODELS.keys()),
        help="Model to validate (default: all models)",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=100000,
        help="Number of simulation trials per (problem, N) pair (default: 100000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    if args.model:
        model_keys = [args.model]
    else:
        model_keys = list(MODELS.keys())

    n_trials = args.n_trials
    seed = args.seed

    measurements_dir = RESULTS_DIR / "measurements"
    predictions_dir = RESULTS_DIR / "predictions"
    actuals_dir = RESULTS_DIR / "actuals"

    for model_key in model_keys:
        model_meas_dir = measurements_dir / model_key
        model_pred_dir = predictions_dir / model_key
        model_act_dir = actuals_dir / model_key
        model_act_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 60}")
        print(f"Validating model: {model_key}")
        print(f"{'=' * 60}")

        if not model_meas_dir.exists():
            print(f"No measurements found at {model_meas_dir}. Run 02_measure.py first.")
            continue
        if not model_pred_dir.exists():
            print(f"No predictions found at {model_pred_dir}. Run 03_predict.py first.")
            continue

        # Discover prediction files (these are the problems with 0 < p < 1)
        pred_files = sorted(model_pred_dir.glob("problem_*.json"))
        if not pred_files:
            print(f"No prediction files found in {model_pred_dir}.")
            continue

        print(f"Found {len(pred_files)} prediction files.")

        rng = np.random.default_rng(seed)

        # Per-problem validation results
        per_problem_results = []
        n_validated = 0
        n_errors = 0

        # Aggregation structures
        # Per N: collect (predicted_f, actual_f) pairs
        per_n_predicted = defaultdict(list)
        per_n_actual = defaultdict(list)

        # Per level per N: collect (predicted_f, actual_f) pairs
        per_level_n_predicted = defaultdict(lambda: defaultdict(list))
        per_level_n_actual = defaultdict(lambda: defaultdict(list))

        for pred_file in tqdm(pred_files, desc=f"Validating {model_key}", total=len(pred_files)):
            pred = load_prediction(pred_file)
            if pred is None:
                n_errors += 1
                continue

            prob_idx = pred["problem_idx"]
            level = pred.get("level", None)
            prob_type = pred.get("type", None)
            predictions = pred["predictions"]  # {str(N): f_val}

            # Load corresponding measurement for scores and correctness
            meas_path = model_meas_dir / f"problem_{prob_idx}.json"
            meas = load_measurement(meas_path)
            if meas is None:
                print(f"\nWARNING: No measurement for problem {prob_idx}, skipping.")
                n_errors += 1
                continue

            all_scores = meas["all_scores"]
            all_correct = meas["all_correct"]

            if len(all_scores) == 0:
                n_errors += 1
                continue

            # Simulate best-of-N for each N value
            problem_result = {
                "problem_idx": prob_idx,
                "p": pred["p"],
                "kappa": pred["kappa"],
                "level": level,
                "type": prob_type,
                "n_values": {},
            }

            for n_val in EVAL_N_VALUES:
                n_key = str(n_val)
                predicted_f = predictions.get(n_key, None)
                if predicted_f is None:
                    continue

                try:
                    actual_acc = simulate_best_of_n(
                        scores=all_scores,
                        correct=all_correct,
                        N=n_val,
                        n_trials=n_trials,
                        rng=rng,
                    )
                except Exception as e:
                    print(f"\nERROR: simulate_best_of_n failed for problem {prob_idx}, N={n_val}: {e}")
                    continue

                abs_error = abs(predicted_f - actual_acc)

                problem_result["n_values"][n_key] = {
                    "predicted_f": predicted_f,
                    "actual_acc": actual_acc,
                    "abs_error": abs_error,
                }

                # Collect for aggregation
                per_n_predicted[n_val].append(predicted_f)
                per_n_actual[n_val].append(actual_acc)

                if level is not None:
                    per_level_n_predicted[level][n_val].append(predicted_f)
                    per_level_n_actual[level][n_val].append(actual_acc)

            per_problem_results.append(problem_result)
            n_validated += 1

        # Build aggregate results
        aggregate = {
            "model_key": model_key,
            "n_problems_validated": n_validated,
            "n_errors": n_errors,
            "n_trials_per_simulation": n_trials,
            "seed": seed,
            "per_n": {},
            "per_level": {},
        }

        # Per-N aggregation
        print(f"\n--- Per-N Validation Results ---")
        print(f"{'N':>5s}  {'Mean Pred':>10s}  {'Mean Actual':>11s}  {'Mean |Err|':>10s}  {'Max |Err|':>9s}  {'Corr':>6s}")
        print("-" * 60)

        for n_val in EVAL_N_VALUES:
            if n_val not in per_n_predicted or len(per_n_predicted[n_val]) == 0:
                continue

            pred_arr = np.array(per_n_predicted[n_val])
            act_arr = np.array(per_n_actual[n_val])
            abs_err = np.abs(pred_arr - act_arr)

            mean_pred = float(np.mean(pred_arr))
            mean_act = float(np.mean(act_arr))
            mean_err = float(np.mean(abs_err))
            max_err = float(np.max(abs_err))

            # Correlation
            if len(pred_arr) > 1 and np.std(pred_arr) > 0 and np.std(act_arr) > 0:
                corr = float(np.corrcoef(pred_arr, act_arr)[0, 1])
            else:
                corr = None

            aggregate["per_n"][str(n_val)] = {
                "n_problems": len(pred_arr),
                "mean_predicted": mean_pred,
                "mean_actual": mean_act,
                "mean_abs_error": mean_err,
                "max_abs_error": max_err,
                "correlation": corr,
            }

            corr_str = f"{corr:.4f}" if corr is not None else "N/A"
            print(f"{n_val:5d}  {mean_pred:10.4f}  {mean_act:11.4f}  {mean_err:10.4f}  {max_err:9.4f}  {corr_str:>6s}")

        # Per-level aggregation
        levels = sorted(per_level_n_predicted.keys())
        if levels:
            print(f"\n--- Per-Level Breakdown ---")
            for level in levels:
                print(f"\nLevel: {level}")
                print(f"  {'N':>5s}  {'Mean Pred':>10s}  {'Mean Actual':>11s}  {'Mean |Err|':>10s}  {'Count':>6s}")
                print(f"  {'-' * 50}")

                level_data = {}
                for n_val in EVAL_N_VALUES:
                    if n_val not in per_level_n_predicted[level]:
                        continue
                    pred_arr = np.array(per_level_n_predicted[level][n_val])
                    act_arr = np.array(per_level_n_actual[level][n_val])

                    if len(pred_arr) == 0:
                        continue

                    mean_pred = float(np.mean(pred_arr))
                    mean_act = float(np.mean(act_arr))
                    mean_err = float(np.mean(np.abs(pred_arr - act_arr)))

                    level_data[str(n_val)] = {
                        "n_problems": len(pred_arr),
                        "mean_predicted": mean_pred,
                        "mean_actual": mean_act,
                        "mean_abs_error": mean_err,
                    }

                    print(f"  {n_val:5d}  {mean_pred:10.4f}  {mean_act:11.4f}  {mean_err:10.4f}  {len(pred_arr):6d}")

                aggregate["per_level"][str(level)] = level_data

        # Save comprehensive validation results
        validation_output = {
            "aggregate": aggregate,
            "per_problem": per_problem_results,
        }

        validation_path = model_act_dir / "validation.json"
        with open(validation_path, "w", encoding="utf-8") as f:
            json.dump(validation_output, f, indent=2)
        print(f"\nSaved validation results: {validation_path}")

        # Print final summary
        print(f"\n--- {model_key} Validation Summary ---")
        print(f"Problems validated: {n_validated}")
        print(f"Errors: {n_errors}")
        if aggregate["per_n"]:
            overall_errors = [
                v["mean_abs_error"] for v in aggregate["per_n"].values()
            ]
            print(f"Mean absolute error across all N: {np.mean(overall_errors):.4f}")


if __name__ == "__main__":
    main()
