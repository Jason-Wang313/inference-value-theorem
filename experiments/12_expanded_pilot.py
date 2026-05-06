"""
Expanded pilot sample-complexity experiment.

Part A: Better analysis of existing 48-sample data with representative models.
Part B: Collection pipeline for additional samples (48→128) if API keys work.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import MODELS, N_SAMPLES, RESULTS_DIR, DATA_DIR, NIM_API_KEYS
from src.feature_extraction import load_math500

MEASUREMENTS_DIR = RESULTS_DIR / "measurements"
OUT_DIR = RESULTS_DIR / "du_aligned"
TABLE_DIR = OUT_DIR / "tables"
FIGURE_DIR = OUT_DIR / "figures"

TARGET_MODELS = ["3B", "8B", "70B", "Qwen397B", "Mixtral8x22B"]
TARGET_N_PROBLEMS = 100
EXPANDED_TARGET_SAMPLES = 128

EXPANDED_K_VALUES = [8, 16, 24, 32, 48, 64, 96, 128]
EXPANDED_N_VALUES = [2, 8, 16, 32, 48, 64, 96]
N_SEEDS = 20


def select_stratified_problems(
    models: list[str],
    n_problems: int = 100,
    seed: int = 42,
) -> list[int]:
    rng = np.random.default_rng(seed)
    bins: dict[str, list[int]] = {"hard": [], "medium": [], "easy": []}

    p_by_idx: dict[int, list[float]] = defaultdict(list)
    for model_key in models:
        model_dir = MEASUREMENTS_DIR / model_key
        for pidx in range(500):
            meas_path = model_dir / f"problem_{pidx}.json"
            if not meas_path.exists():
                continue
            try:
                meas = json.loads(meas_path.read_text(encoding="utf-8"))
                p_by_idx[pidx].append(meas.get("p", 0.5))
            except (json.JSONDecodeError, OSError):
                continue

    for pidx, ps in p_by_idx.items():
        avg_p = np.mean(ps)
        if avg_p < 0.25:
            bins["hard"].append(pidx)
        elif avg_p < 0.75:
            bins["medium"].append(pidx)
        else:
            bins["easy"].append(pidx)

    per_bin = n_problems // 3
    remainder = n_problems - 3 * per_bin
    selected = []
    for i, (bin_name, indices) in enumerate(bins.items()):
        n = per_bin + (1 if i < remainder else 0)
        n = min(n, len(indices))
        if n > 0 and len(indices) > 0:
            chosen = rng.choice(indices, size=n, replace=False).tolist()
            selected.extend(chosen)

    return sorted(selected)[:n_problems]


def _stable_seed(*parts) -> int:
    import hashlib
    text = "|".join(str(p) for p in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16) % (2**32)


def _exact_curve(scores, correct, n_values):
    scores = np.asarray(scores, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    n = len(scores)
    out = {}
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


def run_expanded_analysis_existing(
    selected_problems: list[int],
    models: list[str],
) -> tuple[list[dict], dict]:
    import pandas as pd

    all_data = {}
    for model_key in models:
        records = []
        for pidx in selected_problems:
            meas_path = MEASUREMENTS_DIR / model_key / f"problem_{pidx}.json"
            if not meas_path.exists():
                continue
            try:
                rec = json.loads(meas_path.read_text(encoding="utf-8"))
                rec["model_key"] = model_key
                records.append(rec)
            except (json.JSONDecodeError, OSError):
                continue
        all_data[model_key] = records

    rows = []
    for seed_id in range(N_SEEDS):
        for model_key, records in all_data.items():
            for rec in records:
                scores = np.asarray(rec["all_scores"], dtype=float)
                correct = np.asarray(rec["all_correct"], dtype=bool)
                n = len(scores)
                if n < 4:
                    continue
                possible_k = [K for K in EXPANDED_K_VALUES if K < n]
                for K in possible_k:
                    rng = np.random.default_rng(
                        _stable_seed("expanded_pilot", seed_id, model_key, rec["problem_idx"], K)
                    )
                    perm = rng.permutation(n)
                    pilot_idx = perm[:K]
                    held_idx = perm[K:]
                    held_n = len(held_idx)
                    eval_ns = [N for N in EXPANDED_N_VALUES if N <= held_n]
                    if not eval_ns:
                        continue
                    pred_curve = _exact_curve(scores[pilot_idx], correct[pilot_idx], eval_ns)
                    held_curve = _exact_curve(scores[held_idx], correct[held_idx], eval_ns)
                    for N in eval_ns:
                        mae = abs(pred_curve[N] - held_curve[N])
                        rows.append({
                            "K": K,
                            "N": N,
                            "model": model_key,
                            "seed": seed_id,
                            "problem_idx": rec["problem_idx"],
                            "heldout_MAE": mae,
                        })

    agg_rows = []
    grouped: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        key = (r["model"], r["K"], r["N"])
        grouped[key].append(r["heldout_MAE"])

    for (model, K, N), maes in sorted(grouped.items()):
        maes_arr = np.array(maes, dtype=float)
        mean_mae = float(np.mean(maes_arr))
        std_mae = float(np.std(maes_arr, ddof=1)) if len(maes_arr) > 1 else 0.0
        ci = 1.96 * std_mae / math.sqrt(len(maes_arr)) if len(maes_arr) > 1 else 0.0
        agg_rows.append({
            "K": K,
            "N": N,
            "model": model,
            "heldout_MAE_mean": mean_mae,
            "heldout_MAE_std": std_mae,
            "heldout_MAE_ci_low": mean_mae - ci,
            "heldout_MAE_ci_high": mean_mae + ci,
            "num_problems_used": len(set(r["problem_idx"] for r in rows
                                         if r["model"] == model and r["K"] == K and r["N"] == N)),
            "num_splits": len(maes_arr),
        })

    summary = {
        "models": models,
        "n_problems": len(selected_problems),
        "K_values_tested": sorted(set(r["K"] for r in agg_rows)),
        "N_values_tested": sorted(set(r["N"] for r in agg_rows)),
        "n_seeds": N_SEEDS,
        "data_source": "existing_48_samples",
    }

    return agg_rows, summary


def check_api_connectivity() -> bool:
    try:
        from openai import OpenAI
        client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=NIM_API_KEYS[0],
        )
        models = client.models.list()
        return True
    except Exception as e:
        print(f"API connectivity check failed: {e}")
        return False


def collect_additional_samples(
    models: list[str],
    problem_indices: list[int],
    problems: list[dict],
    target_n: int = 128,
    existing_n: int = 48,
) -> int:
    from src.nim_client import NIMClient, _cache_path, _call_batch
    import time

    nim = NIMClient()
    collected = 0

    for model_key in models:
        nim_name = MODELS[model_key]
        print(f"\n  Collecting for {model_key} ({nim_name})...")

        for pidx in problem_indices:
            problem_text = problems[pidx]["problem"]

            needed = []
            for s_idx in range(existing_n, target_n):
                path = _cache_path(nim_name, problem_text, s_idx)
                if not path.exists():
                    needed.append(s_idx)

            if not needed:
                continue

            batch_size = min(16, len(needed))
            for batch_start in range(0, len(needed), batch_size):
                batch_indices = needed[batch_start : batch_start + batch_size]
                n_batch = len(batch_indices)

                for key_idx, key in enumerate(NIM_API_KEYS):
                    try:
                        from openai import OpenAI
                        client = OpenAI(
                            base_url="https://integrate.api.nvidia.com/v1",
                            api_key=key,
                        )
                        response = client.chat.completions.create(
                            model=nim_name,
                            messages=[{"role": "user", "content": problem_text}],
                            temperature=0.7,
                            max_tokens=4096,
                            n=n_batch,
                            logprobs=True,
                            top_logprobs=1,
                        )

                        for choice_idx, choice in enumerate(response.choices):
                            if choice_idx >= len(batch_indices):
                                break
                            s_idx = batch_indices[choice_idx]
                            content = choice.message.content or ""
                            lp = choice.logprobs
                            logprobs_list = []
                            if lp and lp.content:
                                logprobs_list = [
                                    {"token": t.token, "logprob": t.logprob}
                                    for t in lp.content
                                ]
                            cache_data = {
                                "content": content,
                                "logprobs": logprobs_list,
                                "model": nim_name,
                                "problem": problem_text,
                                "sample_idx": s_idx,
                            }
                            path = _cache_path(nim_name, problem_text, s_idx)
                            path.write_text(json.dumps(cache_data), encoding="utf-8")
                            collected += 1

                        break
                    except Exception as e:
                        if key_idx < len(NIM_API_KEYS) - 1:
                            time.sleep(2)
                            continue
                        print(f"    Failed for problem {pidx}: {e}")
                        break

                time.sleep(0.5)

            if collected % 100 == 0 and collected > 0:
                print(f"    Collected {collected} new samples so far...")

    return collected


def make_expanded_pilot_figure(rows: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    if df.empty:
        return

    overall = (
        df.groupby(["K", "N"])
        .agg(heldout_MAE_mean=("heldout_MAE_mean", "mean"))
        .reset_index()
        .sort_values(["N", "K"])
    )

    plt.figure(figsize=(7.0, 4.6))
    for N, sub in overall.groupby("N"):
        plt.plot(sub["K"], sub["heldout_MAE_mean"], marker="o", label=f"N={int(N)}")
    plt.xlabel("Pilot samples K")
    plt.ylabel("Held-out MAE")
    plt.title("Expanded pilot sample-complexity (representative models)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "expanded_pilot_sample_complexity.pdf")
    plt.close()


def main() -> None:
    import pandas as pd

    parser = argparse.ArgumentParser()
    parser.add_argument("--collect", action="store_true", help="Attempt to collect additional samples.")
    parser.add_argument("--target-samples", type=int, default=EXPANDED_TARGET_SAMPLES)
    parser.add_argument("--models", nargs="*", default=TARGET_MODELS)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    available_models = [m for m in args.models if m in MODELS]
    if not available_models:
        print("No target models found in MODELS catalogue.")
        return

    print(f"Selecting {TARGET_N_PROBLEMS} stratified problems...")
    selected = select_stratified_problems(available_models, TARGET_N_PROBLEMS)
    print(f"Selected {len(selected)} problems.")

    collection_ran = False
    if args.collect:
        print("\nChecking API connectivity...")
        if check_api_connectivity():
            print("API keys work. Collecting additional samples...")
            problems = load_math500(DATA_DIR)
            collected = collect_additional_samples(
                available_models, selected, problems,
                target_n=args.target_samples,
            )
            print(f"Collected {collected} additional samples.")
            collection_ran = collected > 0
        else:
            print("API keys failed. Writing NOT_RUN report.")
            not_run_path = OUT_DIR / "expanded_pilot_NOT_RUN.txt"
            not_run_path.write_text(
                f"""EXPANDED PILOT SAMPLE COLLECTION NOT RUN

API connectivity check failed. To collect additional samples:

1. Ensure NIM API keys are valid in config.py or set NIM_API_KEYS env var
2. Run: python experiments/12_expanded_pilot.py --collect --target-samples {args.target_samples}

Target: {args.target_samples} samples/problem for {len(available_models)} models x {len(selected)} problems
Models: {available_models}
Problems: {selected[:10]}... ({len(selected)} total)
""",
                encoding="utf-8",
            )

    print("\nRunning expanded analysis on existing 48-sample data...")
    agg_rows, summary = run_expanded_analysis_existing(selected, available_models)

    if collection_ran:
        summary["data_source"] = f"existing_48_plus_collected_up_to_{args.target_samples}"

    df = pd.DataFrame(agg_rows)
    df.to_csv(TABLE_DIR / "expanded_pilot_sample_complexity.csv", index=False)
    make_expanded_pilot_figure(agg_rows)

    print(f"\nExpanded pilot results:")
    print(f"  Models: {available_models}")
    print(f"  Problems: {len(selected)}")
    print(f"  K values: {summary['K_values_tested']}")
    print(f"  Rows: {len(df)}")
    print(f"  Output: {TABLE_DIR / 'expanded_pilot_sample_complexity.csv'}")

    return summary


if __name__ == "__main__":
    main()
