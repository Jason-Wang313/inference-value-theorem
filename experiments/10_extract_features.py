"""
Runner for raw-cache feature extraction.

Streams through all raw JSON caches and writes per-response features to CSV.
Memory-safe: processes one file at a time.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import MODELS, N_SAMPLES, RESULTS_DIR, DATA_DIR
from src.feature_extraction import extract_all_features, load_math500

FOCUS_MODELS = ["3B", "8B", "70B", "Qwen397B", "Mixtral8x22B"]


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Extract for all models (slow)")
    parser.add_argument("--models", nargs="*", default=None, help="Specific models")
    parser.add_argument("--append", action="store_true", help="Append to existing CSV")
    args = parser.parse_args()

    measurements_dir = RESULTS_DIR / "measurements"
    output_csv = RESULTS_DIR / "du_aligned" / "cache" / "raw_response_features.csv"

    if args.models:
        target_models = {k: MODELS[k] for k in args.models if k in MODELS}
    elif args.all:
        target_models = MODELS
    else:
        target_models = {k: MODELS[k] for k in FOCUS_MODELS if k in MODELS}

    print("Loading MATH500 problems...")
    problems = load_math500(DATA_DIR)
    print(f"Loaded {len(problems)} problems.")

    print(f"Extracting features for {len(target_models)} models ({list(target_models.keys())}), {N_SAMPLES} samples/problem...")
    row_count = extract_all_features(
        models=target_models,
        problems=problems,
        n_samples=N_SAMPLES,
        measurements_dir=measurements_dir,
        output_csv_path=output_csv,
        progress=True,
        append=args.append,
    )
    print(f"Done. Wrote {row_count} rows to {output_csv}")
    print(f"File size: {output_csv.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
