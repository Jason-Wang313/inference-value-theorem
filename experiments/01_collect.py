"""
Usage: python experiments/01_collect.py --model 8B [--n_samples 16] [--n_problems 500]

Uses n=N_SAMPLES batch completions: 1 API call = all samples for 1 problem.
17 keys in parallel. 500 problems = 500 API calls per model.
"""

import sys
import json
import argparse
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import MODELS, N_SAMPLES, N_PROBLEMS
from src.nim_client import NIMClient, collect_parallel

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def load_math500(data_path, n_problems):
    problems = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "problem" in obj and "answer" in obj:
                problems.append(obj)
            if len(problems) >= n_problems:
                break
    return problems


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()))
    parser.add_argument("--n_samples", type=int, default=N_SAMPLES)
    parser.add_argument("--n_problems", type=int, default=N_PROBLEMS)
    args = parser.parse_args()

    model_key = args.model
    model_name = MODELS[model_key]

    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "math500.jsonl"

    print(f"Loading MATH500 from {data_path} ...")
    problems = load_math500(data_path, args.n_problems)
    print(f"Loaded {len(problems)} problems.\n")

    print(f"Model: {model_key} ({model_name})")
    print(f"Problems: {len(problems)}, Samples/problem: {args.n_samples}")
    print(f"API calls needed: {len(problems)} (n={args.n_samples} per call)")
    print(f"Total responses: {len(problems) * args.n_samples}\n")

    client = NIMClient()
    start_time = time.time()

    pbar = tqdm(total=len(problems), desc=f"Collecting {model_key}") if tqdm else None
    last_done = 0

    def progress_cb(rc, total):
        nonlocal last_done
        done = rc["done"]
        if pbar and done > last_done:
            pbar.update(done - last_done)
            pbar.set_postfix_str(
                f"cached={rc['cached']} new={rc['new']} err={rc['errors']} dead={len(rc['dead_keys'])}"
            )
            last_done = done

    rc = collect_parallel(client, model_name, problems, args.n_samples, progress_cb)

    if pbar:
        pbar.update(rc["done"] - last_done)
        pbar.close()

    elapsed = time.time() - start_time

    print(f"\n{'='*60}")
    print("COLLECTION SUMMARY")
    print(f"{'='*60}")
    print(f"Model:        {model_key} ({model_name})")
    print(f"Problems:     {len(problems)} (n={args.n_samples} per call)")
    print(f"Cached:       {rc['cached']} problems (all samples existed)")
    print(f"New calls:    {rc['new']} API calls ({rc['new'] * args.n_samples} responses)")
    print(f"Errors:       {rc['errors']}")
    print(f"Dead keys:    {len(rc['dead_keys'])}")
    print(f"Elapsed:      {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
