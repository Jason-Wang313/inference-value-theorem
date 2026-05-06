"""
Usage: python experiments/03_predict.py [--model 8B]

For each model, for each problem with 0 < p < 1:
1. Load measurements from results/measurements/{model}/
2. Compute theoretical f for N in [1, 2, 4, 8, 16, 32, 64]
3. Also compute N=2 closed form and verify consistency
4. Save to results/predictions/{model}/{problem_idx}.json
"""

import sys
import json
import csv
import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# sys.path manipulation to import from project root
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import MODELS, EVAL_N_VALUES, RESULTS_DIR
from src.theorem import compute_f_theoretical, compute_f_n2_closed, predict_all_n

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


def load_measurement(measurement_path: Path) -> dict | None:
    """Load a single problem measurement JSON. Returns None on failure."""
    if not measurement_path.exists():
        return None
    try:
        with open(measurement_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"WARNING: Failed to load {measurement_path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Predict theoretical f at all N values from measurements."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=list(MODELS.keys()),
        help="Model to predict for (default: all models)",
    )
    args = parser.parse_args()

    # Determine which models to process
    if args.model:
        model_keys = [args.model]
    else:
        model_keys = list(MODELS.keys())

    measurements_dir = RESULTS_DIR / "measurements"
    predictions_dir = RESULTS_DIR / "predictions"

    for model_key in model_keys:
        model_meas_dir = measurements_dir / model_key
        model_pred_dir = predictions_dir / model_key
        model_pred_dir.mkdir(parents=True, exist_ok=True)
        for old_pred in model_pred_dir.glob("problem_*.json"):
            old_pred.unlink()

        print(f"\n{'=' * 60}")
        print(f"Predicting for model: {model_key}")
        print(f"{'=' * 60}")

        if not model_meas_dir.exists():
            print(f"No measurements found at {model_meas_dir}. Run 02_measure.py first.")
            continue

        # Discover all measurement files
        measurement_files = sorted(model_meas_dir.glob("problem_*.json"))
        if not measurement_files:
            print(f"No measurement files found in {model_meas_dir}.")
            continue

        print(f"Found {len(measurement_files)} measurement files.")

        summary_rows = []
        n_predicted = 0
        n_skipped_p0 = 0
        n_skipped_p1 = 0
        n_errors = 0

        for meas_file in tqdm(measurement_files, desc=f"Predicting {model_key}", total=len(measurement_files)):
            meas = load_measurement(meas_file)
            if meas is None:
                n_errors += 1
                continue

            prob_idx = meas["problem_idx"]
            p = meas["p"]
            kappa = meas["kappa"]
            all_scores = meas["all_scores"]
            all_correct = meas["all_correct"]
            level = meas.get("level", None)
            prob_type = meas.get("type", None)

            # Skip problems with p=0 or p=1
            if p == 0.0:
                n_skipped_p0 += 1
                continue
            if p == 1.0:
                n_skipped_p1 += 1
                continue

            # Compute theoretical f for all N values
            try:
                predictions = predict_all_n(
                    all_scores=all_scores,
                    all_correct=all_correct,
                    p=p,
                    kappa=kappa,
                    n_values=EVAL_N_VALUES,
                )
            except Exception as e:
                print(f"\nERROR: predict_all_n failed for problem {prob_idx}: {e}")
                n_errors += 1
                continue

            # Also compute N=2 closed form for verification
            f2_closed = None
            if kappa is not None:
                f2_closed = compute_f_n2_closed(p, kappa)

            n_predicted += 1

            # Save per-problem predictions
            result = {
                "problem_idx": prob_idx,
                "model_key": model_key,
                "p": p,
                "kappa": kappa,
                "predictions": {str(n): f_val for n, f_val in predictions.items()},
                "f2_closed_form": f2_closed,
                "f2_general": predictions.get(2, None),
                "level": level,
                "type": prob_type,
            }

            output_file = model_pred_dir / f"problem_{prob_idx}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)

            # Summary row
            row = {
                "problem_idx": prob_idx,
                "p": p,
                "kappa": kappa if kappa is not None else "",
                "level": level if level is not None else "",
                "type": prob_type if prob_type is not None else "",
            }
            for n_val in EVAL_N_VALUES:
                row[f"f_N{n_val}"] = predictions.get(n_val, "")
            if f2_closed is not None:
                row["f2_closed"] = f2_closed
            else:
                row["f2_closed"] = ""
            summary_rows.append(row)

        # Save summary CSV
        summary_path = predictions_dir / f"{model_key}_summary.csv"
        if summary_rows:
            fieldnames = ["problem_idx", "p", "kappa", "level", "type"]
            for n_val in EVAL_N_VALUES:
                fieldnames.append(f"f_N{n_val}")
            fieldnames.append("f2_closed")

            with open(summary_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(summary_rows)
            print(f"\nSaved summary CSV: {summary_path}")

        # Print summary
        print(f"\n--- {model_key} Prediction Summary ---")
        print(f"Predicted:       {n_predicted}")
        print(f"Skipped (p=0):   {n_skipped_p0}")
        print(f"Skipped (p=1):   {n_skipped_p1}")
        print(f"Errors:          {n_errors}")

        if summary_rows:
            # Show mean predicted f at each N
            print(f"\nMean predicted f across {n_predicted} problems:")
            for n_val in EVAL_N_VALUES:
                col_key = f"f_N{n_val}"
                values = [r[col_key] for r in summary_rows if r[col_key] != ""]
                if values:
                    mean_f = sum(values) / len(values)
                    print(f"  N={n_val:3d}: f = {mean_f:.4f}")

            # N=2 consistency check
            f2_general_vals = [r[f"f_N2"] for r in summary_rows if r[f"f_N2"] != ""]
            f2_closed_vals = [r["f2_closed"] for r in summary_rows if r["f2_closed"] != ""]
            if f2_general_vals and f2_closed_vals and len(f2_general_vals) == len(f2_closed_vals):
                max_diff = max(
                    abs(g - c) for g, c in zip(f2_general_vals, f2_closed_vals)
                )
                mean_diff = sum(
                    abs(g - c) for g, c in zip(f2_general_vals, f2_closed_vals)
                ) / len(f2_general_vals)
                print(f"\nN=2 closed-form vs general formula:")
                print(f"  Max |diff|:  {max_diff:.6f}")
                print(f"  Mean |diff|: {mean_diff:.6f}")


if __name__ == "__main__":
    main()
