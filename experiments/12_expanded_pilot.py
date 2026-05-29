"""
Expanded pilot sample-complexity experiment.

Part A: Better analysis of existing 48-sample data with representative models.
Part B: Collection pipeline for additional samples (48->128/256) if API keys work.
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

from config import MODELS, N_SAMPLES, RESULTS_DIR, DATA_DIR, NIM_API_KEYS, NIM_BASE_URL, RATE_LIMIT_PER_KEY
from src.feature_extraction import load_math500

MEASUREMENTS_DIR = RESULTS_DIR / "measurements"
OUT_DIR = RESULTS_DIR / "du_aligned"
TABLE_DIR = OUT_DIR / "tables"
FIGURE_DIR = OUT_DIR / "figures"
EXPANDED_MEASUREMENTS_DIR = OUT_DIR / "expanded_measurements"

TARGET_MODELS = ["3B", "70B", "Super49B", "Super120B", "MistralSmall119B"]
UNAVAILABLE_FOR_NEW_COLLECTION = {"Mixtral8x22B", "MiniMax", "KimiK2", "GLM5"}
SINGLE_COMPLETION_MODEL_KEYS = {"8B", "Qwen122B"}
MAX_BATCH_COMPLETIONS_BY_MODEL = {
    "Qwen397B": 4,
    "Super49B": 4,
}
REQUEST_TIMEOUT_SECONDS = 180.0
TARGET_N_PROBLEMS = 100
EXPANDED_TARGET_SAMPLES = 256

EXPANDED_K_VALUES = [8, 16, 24, 32, 48, 64, 96, 128, 192]
EXPANDED_N_VALUES = [2, 8, 16, 32, 48, 64, 96, 128]
N_SEEDS = 20


def _valid_cache_file(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        rec = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return False
    return "content" in rec


def _cache_file_present(path: Path) -> bool:
    """Fast preflight check; full JSON validation happens before measurement."""
    try:
        return path.exists()
    except OSError:
        return False


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


def existing_expanded_problem_subset(models: list[str], n_problems: int) -> list[int]:
    """Reuse a completed expanded-measurement subset to avoid selection drift."""
    common: set[int] | None = None
    for model_key in models:
        model_dir = EXPANDED_MEASUREMENTS_DIR / model_key
        if not model_dir.exists():
            return []
        indices: set[int] = set()
        for path in model_dir.glob("problem_*.json"):
            suffix = path.stem.split("_", 1)[1]
            if suffix.isdigit():
                indices.add(int(suffix))
        if not indices:
            return []
        common = indices if common is None else common & indices
    if common is None or len(common) < n_problems:
        return []
    return sorted(common)[:n_problems]


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


def _load_measurement_record(model_key: str, pidx: int) -> dict | None:
    expanded_path = EXPANDED_MEASUREMENTS_DIR / model_key / f"problem_{pidx}.json"
    baseline_path = MEASUREMENTS_DIR / model_key / f"problem_{pidx}.json"
    for path in [expanded_path, baseline_path]:
        if not path.exists():
            continue
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if len(rec.get("all_scores", [])) == len(rec.get("all_correct", [])):
            rec["model_key"] = model_key
            return rec
    return None


def run_expanded_analysis_existing(
    selected_problems: list[int],
    models: list[str],
) -> tuple[list[dict], dict]:
    import pandas as pd

    all_data = {}
    for model_key in models:
        records = []
        for pidx in selected_problems:
            rec = _load_measurement_record(model_key, pidx)
            if rec is not None:
                records.append(rec)
        all_data[model_key] = records

    rows = []
    sample_lengths = []
    sample_lengths_by_model: dict[str, list[int]] = defaultdict(list)
    for seed_id in range(N_SEEDS):
        for model_key, records in all_data.items():
            for rec in records:
                scores = np.asarray(rec["all_scores"], dtype=float)
                correct = np.asarray(rec["all_correct"], dtype=bool)
                n = len(scores)
                if seed_id == 0:
                    sample_lengths.append(n)
                    sample_lengths_by_model[model_key].append(n)
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
        "data_source": f"measurement_artifacts_max_{max(sample_lengths) if sample_lengths else 0}_samples",
        "max_samples_per_problem": max(sample_lengths) if sample_lengths else 0,
        "per_model_sample_coverage": {
            model: {
                "num_records": len(lengths),
                "max_samples": max(lengths) if lengths else 0,
                "min_samples": min(lengths) if lengths else 0,
                "records_at_max": sum(1 for n in lengths if n == max(lengths)) if lengths else 0,
            }
            for model, lengths in sorted(sample_lengths_by_model.items())
        },
        "expanded_measurements_dir": str(EXPANDED_MEASUREMENTS_DIR),
    }

    return agg_rows, summary


def write_cache_coverage(
    models: list[str],
    problem_indices: list[int],
    problems: list[dict],
    target_n: int,
) -> list[dict]:
    from src.nim_client import _cache_path
    import pandas as pd

    rows = []
    for model_key in models:
        nim_name = MODELS[model_key]
        for pidx in problem_indices:
            problem_text = problems[pidx]["problem"]
            cached = sum(1 for s_idx in range(target_n) if _cache_file_present(_cache_path(nim_name, problem_text, s_idx)))
            rows.append(
                {
                    "model": model_key,
                    "problem_idx": pidx,
                    "target_samples": target_n,
                    "cached_samples": cached,
                    "complete": cached >= target_n,
                    "scan_mode": "file_exists_fast",
                }
            )
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(TABLE_DIR / "expanded_pilot_cache_coverage.csv", index=False)
    return rows


def measure_selected_problems(
    models: list[str],
    problem_indices: list[int],
    problems: list[dict],
    n_samples: int,
) -> int:
    from src.nim_client import _cache_path
    from src.scorer import compute_problem_stats

    measured = 0
    for model_key in models:
        nim_name = MODELS[model_key]
        model_dir = EXPANDED_MEASUREMENTS_DIR / model_key
        model_dir.mkdir(parents=True, exist_ok=True)
        for pidx in problem_indices:
            prob = problems[pidx]
            responses = []
            missing = False
            for s_idx in range(n_samples):
                path = _cache_path(nim_name, prob["problem"], s_idx)
                if not _valid_cache_file(path):
                    missing = True
                    break
                try:
                    responses.append(json.loads(path.read_text(encoding="utf-8")))
                except (json.JSONDecodeError, OSError):
                    missing = True
                    break
            if missing:
                continue

            stats = compute_problem_stats(responses, prob["answer"])
            record = {
                "problem_idx": pidx,
                "model_key": model_key,
                "model_name": nim_name,
                "p": stats["p"],
                "kappa": stats["kappa"],
                "n_correct": stats["n_correct"],
                "n_incorrect": stats["n_incorrect"],
                "scores_correct": stats["scores_correct"],
                "scores_incorrect": stats["scores_incorrect"],
                "all_scores": stats["all_scores"],
                "all_correct": stats["all_correct"],
                "level": prob.get("level"),
                "type": prob.get("type"),
                "ground_truth": prob["answer"],
                "n_samples": n_samples,
                "source": "expanded_pilot_subset",
            }
            (model_dir / f"problem_{pidx}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
            measured += 1
    return measured


def check_api_connectivity() -> bool:
    try:
        from openai import OpenAI
        client = OpenAI(
            base_url=NIM_BASE_URL,
            api_key=NIM_API_KEYS[0],
            timeout=180.0,
            max_retries=0,
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
    workers: int | None = None,
    request_delay: float = 0.0,
    rate_limit_cooldown: float = 90.0,
    max_job_attempts: int = 8,
) -> int:
    from src.nim_client import _cache_path
    from openai import OpenAI
    from queue import Empty, Queue
    import threading
    import time

    if not NIM_API_KEYS:
        raise RuntimeError("No NIM API keys available.")

    jobs: Queue = Queue()
    for model_key in models:
        if model_key in UNAVAILABLE_FOR_NEW_COLLECTION:
            print(f"  Skipping {model_key}: provider endpoint is unavailable for new collection.")
            continue
        nim_name = MODELS[model_key]
        for pidx in problem_indices:
            problem_text = problems[pidx]["problem"]
            needed = []
            for s_idx in range(target_n):
                path = _cache_path(nim_name, problem_text, s_idx)
                if not _cache_file_present(path):
                    needed.append(s_idx)

            if not needed:
                continue

            max_batch = MAX_BATCH_COMPLETIONS_BY_MODEL.get(model_key, 16)
            batch_size = 1 if model_key in SINGLE_COMPLETION_MODEL_KEYS else min(max_batch, len(needed))
            for batch_start in range(0, len(needed), batch_size):
                batch_indices = needed[batch_start : batch_start + batch_size]
                jobs.put((model_key, nim_name, pidx, problem_text, batch_indices, 0))

    total_jobs = jobs.qsize()
    if total_jobs == 0:
        return 0

    counter = {"collected": 0, "jobs_done": 0, "jobs_failed": 0, "jobs_requeued": 0}
    lock = threading.Lock()
    min_delay = max(60.0 / max(float(RATE_LIMIT_PER_KEY), 1.0), request_delay)

    def _err_text(err: object) -> str:
        return str(err) if err is not None else ""

    def _is_auth_error(err: object) -> bool:
        text = _err_text(err)
        return "401" in text or "403" in text or "Authorization failed" in text

    def _is_gone_error(err: object) -> bool:
        text = _err_text(err)
        return "410" in text or "reached its end of life" in text or "Gone" in text

    def _is_retryable_error(err: object) -> bool:
        text = _err_text(err).lower()
        return any(
            marker in text
            for marker in [
                "429",
                "too many requests",
                "rate limit",
                "timed out",
                "timeout",
                "temporarily unavailable",
                "502",
                "503",
                "504",
            ]
        )

    def worker(key: str, worker_idx: int) -> None:
        client = OpenAI(
            base_url=NIM_BASE_URL,
            api_key=key,
            timeout=180.0,
            max_retries=0,
        )
        while True:
            try:
                item = jobs.get_nowait()
            except Empty:
                return
            if len(item) == 5:
                model_key, nim_name, pidx, problem_text, batch_indices = item
                job_attempt = 0
            else:
                model_key, nim_name, pidx, problem_text, batch_indices, job_attempt = item

            n_batch = len(batch_indices)
            last_error = None
            wrote = 0
            for attempt in range(4):
                try:
                    response = client.chat.completions.create(
                        model=nim_name,
                        messages=[
                            {
                                "role": "system",
                                "content": "Solve the math problem. Show your work step by step. Put your final answer in \\boxed{}.",
                            },
                            {"role": "user", "content": problem_text},
                        ],
                        temperature=0.7,
                        max_tokens=2048,
                        n=n_batch,
                        logprobs=True,
                        top_logprobs=5,
                        timeout=REQUEST_TIMEOUT_SECONDS,
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
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text(json.dumps(cache_data), encoding="utf-8")
                        wrote += 1
                    break
                except Exception as e:
                    last_error = e
                    if _is_auth_error(e) or _is_gone_error(e):
                        break
                    if _is_retryable_error(e):
                        time.sleep(max(rate_limit_cooldown, min(2 ** attempt * 3, 60)))
                    else:
                        time.sleep(min(2 ** attempt * 3, 60))

            with lock:
                if wrote == 0 and _is_auth_error(last_error):
                    jobs.put((model_key, nim_name, pidx, problem_text, batch_indices, job_attempt))
                    counter["jobs_failed"] += 1
                    print(
                        f"    Retiring key worker {worker_idx} after auth failure; "
                        f"requeued {model_key} problem {pidx} batch."
                    )
                    jobs.task_done()
                    return

                counter["jobs_done"] += 1
                counter["collected"] += wrote
                if wrote != n_batch:
                    missing_indices = batch_indices[wrote:]
                    if _is_retryable_error(last_error) and job_attempt + 1 < max_job_attempts:
                        jobs.put((model_key, nim_name, pidx, problem_text, missing_indices, job_attempt + 1))
                        counter["jobs_requeued"] += 1
                        print(
                            f"    Requeued batch {model_key} problem {pidx}: wrote {wrote}/{n_batch}; "
                            f"attempt {job_attempt + 1}/{max_job_attempts}; last error={last_error}"
                        )
                    else:
                        counter["jobs_failed"] += 1
                        print(f"    Incomplete batch {model_key} problem {pidx}: wrote {wrote}/{n_batch}; last error={last_error}")
                if counter["jobs_done"] % 10 == 0 or counter["jobs_done"] == total_jobs:
                    print(
                        f"    Progress: {counter['jobs_done']}/{total_jobs} batches, "
                        f"{counter['collected']} samples collected, {counter['jobs_requeued']} requeued, "
                        f"{counter['jobs_failed']} incomplete"
                    )

            jobs.task_done()
            time.sleep(min_delay)

    n_workers = min(workers or len(NIM_API_KEYS), len(NIM_API_KEYS), total_jobs)
    print(f"  Collecting {total_jobs} missing batches with {n_workers} key workers...")
    threads = [
        threading.Thread(target=worker, args=(key, idx), daemon=True)
        for idx, key in enumerate(NIM_API_KEYS[:n_workers])
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    return int(counter["collected"])


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
    parser.add_argument("--selection-models", nargs="*", default=None, help="Models used only to choose the stratified problem subset.")
    parser.add_argument("--n-problems", type=int, default=TARGET_N_PROBLEMS)
    parser.add_argument("--measure", action="store_true", help="Measure selected problems into expanded measurement artifacts.")
    parser.add_argument("--workers", type=int, default=None, help="Max key workers for collection.")
    parser.add_argument("--request-delay", type=float, default=0.0, help="Minimum seconds between collection requests per worker.")
    parser.add_argument("--rate-limit-cooldown", type=float, default=90.0, help="Sleep seconds after retryable collection errors such as 429s.")
    parser.add_argument("--max-job-attempts", type=int, default=8, help="Max queue-level attempts before leaving a collection batch incomplete.")
    parser.add_argument("--skip-coverage", action="store_true", help="Skip raw-cache coverage scan for analysis-only refreshes.")
    parser.add_argument("--collect-only", action="store_true", help="Collect/cache samples and skip measurement/analysis artifact refresh.")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    available_models = [m for m in args.models if m in MODELS]
    if not available_models:
        print("No target models found in MODELS catalogue.")
        return
    selection_models = [m for m in (args.selection_models or args.models) if m in MODELS]
    if not selection_models:
        selection_models = available_models

    selected = existing_expanded_problem_subset(selection_models, args.n_problems)
    if selected:
        print(f"Reusing {len(selected)} existing expanded-measurement problems.")
    else:
        print(f"Selecting {args.n_problems} stratified problems...")
        selected = select_stratified_problems(selection_models, args.n_problems)
    print(f"Selected {len(selected)} problems.")

    collection_ran = False
    problems = load_math500(DATA_DIR)
    coverage_rows = []
    if not args.skip_coverage:
        coverage_rows = write_cache_coverage(available_models, selected, problems, args.target_samples)
        complete_before = sum(1 for r in coverage_rows if r["complete"])
        print(f"Cache coverage before collection: {complete_before}/{len(coverage_rows)} complete model/problem pairs.")
    if args.collect:
        print("\nChecking API connectivity...")
        if check_api_connectivity():
            print("API keys work. Collecting additional samples...")
            collected = collect_additional_samples(
                available_models, selected, problems,
                target_n=args.target_samples,
                workers=args.workers,
                request_delay=args.request_delay,
                rate_limit_cooldown=args.rate_limit_cooldown,
                max_job_attempts=args.max_job_attempts,
            )
            print(f"Collected {collected} additional samples.")
            collection_ran = collected > 0
            coverage_rows = write_cache_coverage(available_models, selected, problems, args.target_samples)
            complete_after = sum(1 for r in coverage_rows if r["complete"])
            print(f"Cache coverage after collection: {complete_after}/{len(coverage_rows)} complete model/problem pairs.")
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

    if args.collect_only:
        print("Collect-only mode; skipping measurement and analysis refresh.")
        return {}

    if args.measure:
        measured = measure_selected_problems(available_models, selected, problems, args.target_samples)
        print(f"Measured {measured} complete expanded model/problem records at n={args.target_samples}.")

    print("\nRunning expanded analysis on existing 48-sample data...")
    agg_rows, summary = run_expanded_analysis_existing(selected, available_models)

    if collection_ran:
        summary["data_source"] = f"existing_48_plus_collected_up_to_{args.target_samples}"

    df = pd.DataFrame(agg_rows)
    df.to_csv(TABLE_DIR / "expanded_pilot_sample_complexity.csv", index=False)
    make_expanded_pilot_figure(agg_rows)
    (OUT_DIR / "expanded_pilot_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nExpanded pilot results:")
    print(f"  Models: {available_models}")
    print(f"  Problems: {len(selected)}")
    print(f"  K values: {summary['K_values_tested']}")
    print(f"  Rows: {len(df)}")
    print(f"  Output: {TABLE_DIR / 'expanded_pilot_sample_complexity.csv'}")

    return summary


if __name__ == "__main__":
    main()
