"""
Maxed-out campaign orchestrator for the LLM response-pool evaluation project.

This script does not assume the 47M-response campaign should be launched by
accident. It provides locked manifests, resumable MATH collection/measurement,
held-out pilot analysis, and lightweight status reports for the three planned
expansions:

1. held-out forecasting / sample complexity,
2. real verifier / judge-score scaling,
3. cross-benchmark generalization.

Typical use:

    python experiments/17_maxed_out_campaign.py prepare
    python experiments/17_maxed_out_campaign.py status

Smoke-safe local check:

    python experiments/17_maxed_out_campaign.py prepare --smoke --force
    python experiments/17_maxed_out_campaign.py status --smoke
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Queue
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import DATA_DIR, MODELS, NIM_API_KEYS, NIM_BASE_URL, RATE_LIMIT_PER_KEY, RESULTS_DIR  # noqa: E402
from src.nim_client import _cache_path  # noqa: E402
from src.scorer import compute_problem_stats  # noqa: E402


OUT_DIR = RESULTS_DIR / "maxed_out"
MANIFEST_PATH = OUT_DIR / "manifest.json"
SMOKE_MANIFEST_PATH = OUT_DIR / "manifest.smoke.json"

HELDOUT_DIR = OUT_DIR / "heldout_forecasting"
HELDOUT_MEASUREMENTS_DIR = HELDOUT_DIR / "measurements"
HELDOUT_TABLE_DIR = HELDOUT_DIR / "tables"
COLLECTION_STATUS_DIR = HELDOUT_DIR / "collection_status"

VERIFIER_DIR = OUT_DIR / "real_verifier"
CROSS_DIR = OUT_DIR / "cross_benchmark"
EXPANDED_PROXY_DIR = OUT_DIR / "expanded_pilot_proxy"

DEFAULT_K_VALUES = [8, 16, 32, 64, 128, 256, 512, 1024]
DEFAULT_N_VALUES = [1, 2, 4, 8, 16, 32, 48, 64, 128, 256]
DEFAULT_CROSS_BENCHMARKS = [
    "math500",
    "gpqa_diamond",
    "ifeval",
    "livebench_selected",
    "livecodebench",
    "humaneval_mbpp",
]
SUPPORTED_EXISTING_CROSS_RUNNER_BENCHMARKS = {
    "math500",
    "gpqa_diamond",
    "ifeval",
    "livebench_selected",
    "livecodebench",
    "humaneval_mbpp",
}
DEFAULT_JUDGE_LABELS = ["judge_a", "judge_b", "judge_c"]
IDEAL_GATES = {
    "exact_law_mae": 0.001,
    "cross_benchmark_law_mae": 0.005,
    "heldout_k128_mae": 0.03,
    "heldout_k256_mae": 0.02,
    "heldout_k512_mae": 0.015,
    "verifier_delta_over_logprob": 0.03,
    "adaptive_delta_over_uniform": 0.0,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def run_git(args: list[str]) -> str:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
    except Exception:
        return "UNKNOWN"
    if proc.returncode != 0:
        return "UNKNOWN"
    return proc.stdout.strip()


def stable_seed(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16) % (2**32)


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_math500(limit: int | None = None) -> list[dict[str, Any]]:
    path = DATA_DIR / "math500.jsonl"
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "problem" in rec and "answer" in rec:
                rows.append(rec)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def parse_indices(value: str | None, upper: int | None = None) -> list[int] | None:
    if value is None or value.strip() == "":
        return None
    out: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start, end = int(start_s), int(end_s)
            if end < start:
                raise ValueError(f"Bad index range: {part}")
            out.update(range(start, end + 1))
        else:
            out.add(int(part))
    indices = sorted(out)
    if upper is not None:
        indices = [idx for idx in indices if 0 <= idx < upper]
    return indices


def discover_measured_models() -> list[str]:
    root = RESULTS_DIR / "measurements"
    if not root.exists():
        return []
    return sorted(path.name for path in root.iterdir() if path.is_dir())


def model_panel() -> dict[str, Any]:
    measured = discover_measured_models()
    active = sorted(MODELS)
    return {
        "active_live_models": active,
        "historically_measured_models": measured,
        "inactive_historical_models": sorted(set(measured) - set(active)),
        "new_active_models_without_baseline_measurements": sorted(set(active) - set(measured)),
    }


def manifest_path(smoke: bool) -> Path:
    return SMOKE_MANIFEST_PATH if smoke else MANIFEST_PATH


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    panel = model_panel()
    measured = panel["historically_measured_models"]
    active = panel["active_live_models"]

    if args.models:
        target_models = [m for m in args.models if m in set(measured) | set(active)]
    elif args.model_source == "active":
        target_models = active
    elif args.model_source == "measured":
        target_models = measured
    else:
        target_models = sorted(set(measured) | set(active))

    if args.smoke:
        target_models = target_models[: min(2, len(target_models))]
        math_problems = min(args.math_problems, 3)
        heldout_samples = min(args.heldout_samples, 8)
        split = {"pilot": 4, "calibration": 2, "evaluation": 2}
        k_values = [1, 2, 4]
        n_values = [1, 2, 4]
        judge_samples = min(args.judge_samples, 4)
        cross_tasks = min(args.cross_tasks, 3)
        cross_samples = min(args.cross_samples, 4)
        cross_benchmarks = [
            b for b in args.cross_benchmarks
            if b in SUPPORTED_EXISTING_CROSS_RUNNER_BENCHMARKS
        ][:2]
    else:
        math_problems = args.math_problems
        heldout_samples = args.heldout_samples
        split = {
            "pilot": args.pilot_samples,
            "calibration": args.calibration_samples,
            "evaluation": args.evaluation_samples,
        }
        k_values = args.k_values
        n_values = args.n_values
        judge_samples = args.judge_samples
        cross_tasks = args.cross_tasks
        cross_samples = args.cross_samples
        cross_benchmarks = args.cross_benchmarks

    split_total = split["pilot"] + split["calibration"] + split["evaluation"]
    if split_total != heldout_samples:
        raise ValueError(
            f"Held-out split totals {split_total}, but heldout_samples={heldout_samples}."
        )

    live_collection_models = [m for m in target_models if m in active]
    inactive_targets = [m for m in target_models if m not in active]

    heldout_responses = len(target_models) * math_problems * heldout_samples
    verifier_responses = len(target_models) * math_problems * judge_samples
    verifier_judgments = verifier_responses * len(args.judge_labels)
    verifier_live_responses = len(live_collection_models) * math_problems * judge_samples
    verifier_live_judgments = verifier_live_responses * len(args.judge_labels)
    cross_responses = len(live_collection_models) * len(cross_benchmarks) * cross_tasks * cross_samples
    supported_cross_benchmarks = [
        b for b in cross_benchmarks
        if b in SUPPORTED_EXISTING_CROSS_RUNNER_BENCHMARKS
    ]
    supported_cross_responses = (
        len(live_collection_models)
        * len(supported_cross_benchmarks)
        * cross_tasks
        * cross_samples
    )

    manifest = {
        "schema_version": 1,
        "campaign": "maxed_out_inference_value_theorem",
        "smoke": bool(args.smoke),
        "created_utc": utc_now(),
        "repo": {
            "path": str(PROJECT_ROOT),
            "commit": run_git(["rev-parse", "HEAD"]),
            "branch": run_git(["branch", "--show-current"]),
            "status_short": run_git(["status", "--short"]),
        },
        "model_panel": panel,
        "target_models": target_models,
        "live_collection_models": live_collection_models,
        "inactive_target_models": inactive_targets,
        "policy": {
            "do_not_claim_inactive_models_as_new_live_runs": True,
            "oracle_scores_are_diagnostic_upper_bounds_only": True,
            "livecodebench_code_scope_default": "public",
            "all_full_runs_must_use_this_manifest": True,
        },
        "heldout_forecasting": {
            "math_problems": math_problems,
            "samples_per_problem": heldout_samples,
            "split": split,
            "K_values": k_values,
            "N_values": n_values,
            "estimators": [
                "raw_plugin",
                "pooled",
                "hierarchical_shrinkage",
                "tail_smoothed_moment",
                "oracle_full_distribution",
            ],
            "target_response_count": heldout_responses,
            "measurement_dir": str(HELDOUT_MEASUREMENTS_DIR),
        },
        "real_verifier": {
            "math_problems": math_problems,
            "responses_per_problem": judge_samples,
            "judge_labels": args.judge_labels,
            "V_values": args.v_values,
            "N_values": [n for n in n_values if n <= judge_samples],
            "score_functions": [
                "mean_logprob",
                "total_logprob",
                "length_normalized_logprob",
                "learned_verifier",
                "judge_a",
                "judge_b",
                "judge_c",
                "judge_ensemble",
                "logprob_plus_judge_ensemble",
                "oracle_correctness_upper_bound",
            ],
            "target_response_count": verifier_responses,
            "target_judgment_count": verifier_judgments,
            "live_collection_response_count": verifier_live_responses,
            "live_collection_judgment_count": verifier_live_judgments,
            "manual_audit_target": args.manual_audit_target,
            "output_dir": str(VERIFIER_DIR),
        },
        "cross_benchmark": {
            "benchmarks": cross_benchmarks,
            "models": live_collection_models,
            "tasks_per_benchmark": cross_tasks,
            "samples_per_task": cross_samples,
            "N_values": [n for n in n_values if n <= cross_samples],
            "target_response_count": cross_responses,
            "supported_existing_runner_benchmarks": supported_cross_benchmarks,
            "supported_existing_runner_response_count": supported_cross_responses,
            "output_dir": str(CROSS_DIR),
        },
        "commands": {
            "prepare": "python experiments/17_maxed_out_campaign.py prepare",
            "status": "python experiments/17_maxed_out_campaign.py status",
            "collect_math_example": (
                "python experiments/17_maxed_out_campaign.py collect-math "
                "--model 3B --problem-indices 0-9 --target-samples 4096"
            ),
            "measure_math_example": (
                "python experiments/17_maxed_out_campaign.py measure-math "
                "--model 3B --problem-indices 0-9 --target-samples 4096"
            ),
            "analyze_heldout_example": (
                "python experiments/17_maxed_out_campaign.py analyze-heldout --models 3B"
            ),
            "cross_benchmark_existing_runner": (
                "python experiments/16_cross_benchmark.py --model-set all "
                "--n-tasks 500 --n-samples 128 --collect --measure --analyze "
                "--allow-unsafe-code-exec --code-test-scope public"
            ),
        },
    }
    return manifest


def write_command_queue(manifest: dict[str, Any], smoke: bool) -> Path:
    path = OUT_DIR / ("command_queue.smoke.jsonl" if smoke else "command_queue.jsonl")
    held = manifest["heldout_forecasting"]
    verifier = manifest["real_verifier"]
    cross = manifest["cross_benchmark"]
    rows = []
    for model in manifest["live_collection_models"]:
        rows.append(
            {
                "phase": "heldout_collect_math",
                "model": model,
                "command": (
                    f"python experiments/17_maxed_out_campaign.py collect-math --model {model} "
                    f"--n-problems {held['math_problems']} --target-samples {held['samples_per_problem']}"
                ),
            }
        )
        rows.append(
            {
                "phase": "heldout_measure_math",
                "model": model,
                "command": (
                    f"python experiments/17_maxed_out_campaign.py measure-math --model {model} "
                    f"--n-problems {held['math_problems']} --target-samples {held['samples_per_problem']}"
                ),
            }
        )
        rows.append(
            {
                "phase": "heldout_analyze",
                "model": model,
                "command": (
                    f"python experiments/17_maxed_out_campaign.py analyze-heldout --models {model}"
                    + (" --smoke" if smoke else "")
                ),
            }
        )

    rows.append(
        {
                "phase": "real_verifier_seed_manifest",
                "command": (
                    "python experiments/11_model_judge.py --live "
                f"--models {' '.join(manifest['live_collection_models'])} "
                f"--n-problems {verifier['math_problems']} "
                f"--max-samples {verifier['responses_per_problem']} "
                "--workers 12 --request-delay 3 --rate-limit-cooldown 90 "
                "--max-task-attempts 16 --live-batch-size 8"
            ),
        }
    )
    supported_cross = [
        b for b in cross["benchmarks"]
        if b in SUPPORTED_EXISTING_CROSS_RUNNER_BENCHMARKS
    ]
    if supported_cross:
        rows.append(
            {
                "phase": "cross_benchmark_collect_measure_analyze",
                "command": (
                    "python experiments/16_cross_benchmark.py "
                    f"--benchmarks {' '.join(supported_cross)} --model-set all "
                    f"--n-tasks {cross['tasks_per_benchmark']} "
                    f"--n-samples {cross['samples_per_task']} --batch-size 1 --workers 192 "
                    "--collect --measure --analyze --allow-unsafe-code-exec "
                    "--code-test-scope public --measure-workers 12"
                ),
            }
        )
    unsupported_cross = [
        b for b in cross["benchmarks"]
        if b not in SUPPORTED_EXISTING_CROSS_RUNNER_BENCHMARKS
    ]
    if unsupported_cross:
        rows.append(
            {
                "phase": "cross_benchmark_loader_gap",
                "benchmarks": unsupported_cross,
                "command": "Add benchmark loaders before running these benchmark families.",
            }
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def cmd_prepare(args: argparse.Namespace) -> None:
    path = manifest_path(args.smoke)
    if path.exists() and not args.force:
        raise SystemExit(f"Manifest already exists: {path}. Use --force to overwrite.")
    manifest = build_manifest(args)
    write_json(path, manifest)
    queue_path = write_command_queue(manifest, args.smoke)
    print(f"Wrote manifest: {path}")
    print(f"Wrote command queue: {queue_path}")
    print(f"Target held-out responses: {manifest['heldout_forecasting']['target_response_count']:,}")
    print(f"Target verifier judgments: {manifest['real_verifier']['target_judgment_count']:,}")
    print(f"Target cross-benchmark responses: {manifest['cross_benchmark']['target_response_count']:,}")


def _valid_cache_file(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        rec = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    return "content" in rec


def _write_cache_file(path: Path, rec: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _extract_logprobs(choice: Any) -> list[dict[str, float]] | None:
    lp = getattr(choice, "logprobs", None)
    if lp is None or getattr(lp, "content", None) is None:
        return None
    return [{"token": t.token, "logprob": t.logprob} for t in lp.content]


def _missing_sample_indices(model_name: str, problem_text: str, target_samples: int) -> list[int]:
    return [
        sample_idx
        for sample_idx in range(target_samples)
        if not _valid_cache_file(_cache_path(model_name, problem_text, sample_idx))
    ]


def cmd_collect_math(args: argparse.Namespace) -> None:
    if args.model not in MODELS:
        raise SystemExit(f"Unknown active model {args.model}. Active models: {sorted(MODELS)}")
    if args.dry_run:
        print("Dry run: no API calls will be made.")
    if not args.dry_run and not NIM_API_KEYS:
        raise SystemExit("No NIM API keys available. Set NIM_API_KEYS or MIRROR .env first.")

    from openai import OpenAI

    problems = load_math500(args.n_problems)
    indices = parse_indices(args.problem_indices, upper=len(problems))
    if indices is None:
        indices = list(range(len(problems)))

    model_name = MODELS[args.model]
    jobs: Queue[tuple[int, str, list[int], int]] = Queue()
    cached = 0
    for pidx in indices:
        missing = _missing_sample_indices(model_name, problems[pidx]["problem"], args.target_samples)
        cached += args.target_samples - len(missing)
        for start in range(0, len(missing), args.batch_size):
            jobs.put((pidx, problems[pidx]["problem"], missing[start : start + args.batch_size], 1))

    total_jobs = jobs.qsize()
    total_samples = len(indices) * args.target_samples
    missing_samples = total_samples - cached
    status_suffix = ".dry_run" if args.dry_run else ""
    status_path = COLLECTION_STATUS_DIR / f"{args.model}_collection_status{status_suffix}.json"
    status = {
        "model": args.model,
        "model_name": model_name,
        "target_samples": args.target_samples,
        "problem_count": len(indices),
        "sample_slots": total_samples,
        "cached_before": cached,
        "missing_before": missing_samples,
        "jobs_before": total_jobs,
        "dry_run": bool(args.dry_run),
        "started_utc": utc_now(),
    }
    write_json(status_path, status)

    if total_jobs == 0 or args.dry_run:
        status.update({"completed_batches": 0, "failed_batches": 0, "collected_samples": 0, "finished_utc": utc_now()})
        write_json(status_path, status)
        print(json.dumps(status, indent=2))
        return

    lock = threading.Lock()
    counters = {
        "completed_batches": 0,
        "failed_batches": 0,
        "requeued_batches": 0,
        "retired_auth_keys": 0,
        "collected_samples": 0,
    }
    failures: list[dict[str, Any]] = []
    min_delay = max(60.0 / max(float(RATE_LIMIT_PER_KEY), 1.0), args.request_delay)
    worker_count = min(args.workers or len(NIM_API_KEYS), len(NIM_API_KEYS), max(total_jobs, 1))

    def is_retryable_error(exc: Exception) -> bool:
        text = str(exc).lower()
        return any(
            marker in text
            for marker in [
                "429",
                "too many requests",
                "rate limit",
                "timeout",
                "timed out",
                "temporarily unavailable",
                "degraded",
                "internal server error",
                "inference connection error",
                "500",
                "502",
                "503",
                "504",
            ]
        )

    def is_auth_error(exc: Exception) -> bool:
        text = str(exc).lower()
        return "401" in text or "403" in text or "authorization failed" in text or "forbidden" in text

    def worker(key: str) -> None:
        client = OpenAI(base_url=NIM_BASE_URL, api_key=key, timeout=args.timeout, max_retries=0)
        while True:
            try:
                pidx, problem_text, sample_indices, attempt = jobs.get(timeout=3)
            except Empty:
                return

            wrote = 0
            last_error: Exception | None = None
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": "Solve the math problem. Show your work step by step. Put your final answer in \\boxed{}.",
                        },
                        {"role": "user", "content": problem_text},
                    ],
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    n=len(sample_indices),
                    logprobs=True,
                    top_logprobs=5,
                    timeout=args.timeout,
                )
                if len(response.choices) != len(sample_indices):
                    raise RuntimeError(
                        f"Model returned {len(response.choices)} choices for {len(sample_indices)} requested samples."
                    )
                for sample_idx, choice in zip(sample_indices, response.choices):
                    rec = {
                        "content": choice.message.content or "",
                        "logprobs": _extract_logprobs(choice),
                        "model": model_name,
                        "problem": problem_text,
                        "sample_idx": sample_idx,
                        "campaign": "maxed_out",
                    }
                    path = _cache_path(model_name, problem_text, sample_idx)
                    _write_cache_file(path, rec)
                    wrote += 1
            except Exception as exc:
                last_error = exc

            retire_worker = False
            with lock:
                if wrote == len(sample_indices):
                    counters["completed_batches"] += 1
                    counters["collected_samples"] += wrote
                else:
                    remaining = sample_indices[wrote:]
                    if last_error is not None and is_auth_error(last_error) and attempt < args.max_job_attempts:
                        jobs.put((pidx, problem_text, remaining, attempt + 1))
                        counters["requeued_batches"] += 1
                        counters["retired_auth_keys"] = counters.get("retired_auth_keys", 0) + 1
                        retire_worker = True
                    elif last_error is not None and is_retryable_error(last_error) and attempt < args.max_job_attempts:
                        jobs.put((pidx, problem_text, remaining, attempt + 1))
                        counters["requeued_batches"] += 1
                    else:
                        counters["failed_batches"] += 1
                        failures.append(
                            {
                                "problem_idx": pidx,
                                "sample_indices": remaining,
                                "attempt": attempt,
                                "error": str(last_error)[:500] if last_error else "incomplete_response",
                            }
                        )
                done = counters["completed_batches"] + counters["failed_batches"]
                if done % args.progress_every == 0 or done == total_jobs:
                    status.update(counters)
                    status["failures"] = failures[-20:]
                    status["updated_utc"] = utc_now()
                    write_json(status_path, status)
                    print(
                        f"{args.model}: batches done={done}/{total_jobs} "
                        f"collected={counters['collected_samples']} "
                        f"requeued={counters['requeued_batches']} failed={counters['failed_batches']}",
                        flush=True,
                    )

            jobs.task_done()
            if retire_worker:
                return
            time.sleep(min_delay)

    threads = [threading.Thread(target=worker, args=(NIM_API_KEYS[i],), daemon=True) for i in range(worker_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    status.update(counters)
    status["failures"] = failures
    status["finished_utc"] = utc_now()
    write_json(status_path, status)
    print(f"Wrote collection status: {status_path}")


def cmd_measure_math(args: argparse.Namespace) -> None:
    if args.model not in MODELS:
        raise SystemExit(f"Unknown active model {args.model}. Active models: {sorted(MODELS)}")

    problems = load_math500(args.n_problems)
    indices = parse_indices(args.problem_indices, upper=len(problems))
    if indices is None:
        indices = list(range(len(problems)))

    model_name = MODELS[args.model]
    out_dir = HELDOUT_MEASUREMENTS_DIR / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    measured = 0
    skipped_missing = 0
    skipped_bad = 0
    for i, pidx in enumerate(indices, 1):
        prob = problems[pidx]
        responses = []
        missing = False
        for sample_idx in range(args.target_samples):
            path = _cache_path(model_name, prob["problem"], sample_idx)
            if not _valid_cache_file(path):
                missing = True
                break
            try:
                responses.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                missing = True
                break
        if missing:
            skipped_missing += 1
            continue

        try:
            stats = compute_problem_stats(responses, prob["answer"])
        except Exception:
            skipped_bad += 1
            continue

        record = {
            "problem_idx": pidx,
            "model_key": args.model,
            "model_name": model_name,
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
            "n_samples": args.target_samples,
            "source": "maxed_out_math",
            "measured_utc": utc_now(),
        }
        write_json(out_dir / f"problem_{pidx}.json", record)
        measured += 1
        if measured % args.progress_every == 0:
            print(f"{args.model}: measured {measured} complete records ({i}/{len(indices)} scanned)")

    summary = {
        "model": args.model,
        "target_samples": args.target_samples,
        "problem_indices": indices,
        "measured": measured,
        "skipped_missing": skipped_missing,
        "skipped_bad": skipped_bad,
        "finished_utc": utc_now(),
    }
    write_json(out_dir / "measurement_summary.json", summary)
    print(json.dumps(summary, indent=2))


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

    out: dict[int, float] = {}
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


def difficulty_bin(p: float) -> str:
    if p <= 0:
        return "no_correct"
    if p >= 1:
        return "all_correct"
    if p < 0.25:
        return "hard"
    if p < 0.75:
        return "medium"
    return "easy"


def tail_smoothed_curve(scores: np.ndarray, correct: np.ndarray, n_values: list[int]) -> dict[int, float]:
    """Small monotone smoothing over score ranks for finite-pilot stability."""
    scores = np.asarray(scores, dtype=float)
    correct = np.asarray(correct, dtype=float)
    n = len(scores)
    if n == 0:
        return {int(N): float("nan") for N in n_values}
    order = np.argsort(scores, kind="mergesort")
    sorted_correct = correct[order]
    window = max(3, int(math.sqrt(n)))
    padded = np.pad(sorted_correct, (window // 2, window - 1 - window // 2), mode="edge")
    kernel = np.ones(window, dtype=float) / window
    smoothed_correct = np.convolve(padded, kernel, mode="valid")
    out = {}
    ranks = (np.arange(1, n + 1, dtype=float) / n)
    lower = (np.arange(0, n, dtype=float) / n)
    for N in n_values:
        masses = np.power(ranks, N) - np.power(lower, N)
        out[int(N)] = float(np.sum(smoothed_correct * masses))
    return out


def estimator_curves_from_slices(
    pilot_scores: np.ndarray,
    pilot_correct: np.ndarray,
    tune_scores: np.ndarray,
    tune_correct: np.ndarray,
    n_values: list[int],
) -> dict[str, dict[int, float]]:
    curves: dict[str, dict[int, float]] = {}
    if len(pilot_scores) > 0:
        curves["raw_plugin"] = exact_curve(pilot_scores, pilot_correct, n_values)
        curves["tail_smoothed_pilot"] = tail_smoothed_curve(pilot_scores, pilot_correct, n_values)
    if len(tune_scores) > 0:
        curves["tune_plugin"] = exact_curve(tune_scores, tune_correct, n_values)
        curves["tail_smoothed_tune"] = tail_smoothed_curve(tune_scores, tune_correct, n_values)
    if len(pilot_scores) > 0 and len(tune_scores) > 0:
        pooled_scores = np.concatenate([pilot_scores, tune_scores])
        pooled_correct = np.concatenate([pilot_correct, tune_correct])
        raw = curves["raw_plugin"]
        tune = curves["tune_plugin"]
        curves["pooled_pilot_tune"] = exact_curve(pooled_scores, pooled_correct, n_values)
        for alpha in [8, 16, 32, 64, 128]:
            weight = len(pilot_scores) / (len(pilot_scores) + alpha)
            curves[f"shrink_raw_to_tune_alpha{alpha}"] = {
                n: float(weight * raw[n] + (1.0 - weight) * tune[n])
                for n in n_values
            }
    return curves


def cmd_analyze_heldout(args: argparse.Namespace) -> None:
    path = manifest_path(args.smoke)
    manifest = load_json(path, {})
    if not manifest:
        raise SystemExit(f"Manifest not found: {path}. Run prepare first.")

    held = manifest["heldout_forecasting"]
    k_values = [int(k) for k in held["K_values"]]
    n_values = [int(n) for n in held["N_values"]]
    split = held["split"]
    total_needed = int(split["pilot"]) + int(split["calibration"]) + int(split["evaluation"])

    models = args.models or manifest["target_models"]
    rows = []
    calibration_rows = []
    pooled_rows = []

    for model in models:
        model_dir = HELDOUT_MEASUREMENTS_DIR / model
        records = []
        for rec_path in sorted(model_dir.glob("problem_*.json")):
            rec = load_json(rec_path, {})
            if not rec:
                continue
            scores = np.asarray(rec.get("all_scores", []), dtype=float)
            correct = np.asarray(rec.get("all_correct", []), dtype=bool)
            if len(scores) >= total_needed and len(scores) == len(correct):
                records.append((rec, scores, correct))
        if not records:
            print(f"{model}: no complete maxed-out measurement records found.")
            continue

        for K in k_values:
            if K > split["pilot"]:
                continue
            pooled_pilot_scores = []
            pooled_pilot_correct = []
            pooled_eval_scores = []
            pooled_eval_correct = []
            calibration_curves: list[dict[int, float]] = []
            per_problem_payload = []

            for rec, scores, correct in records:
                pidx = int(rec["problem_idx"])
                rng = np.random.default_rng(stable_seed("maxed_heldout", model, pidx))
                perm = rng.permutation(len(scores))
                pilot_idx = perm[: split["pilot"]]
                cal_idx = perm[split["pilot"] : split["pilot"] + split["calibration"]]
                eval_idx = perm[
                    split["pilot"] + split["calibration"] :
                    split["pilot"] + split["calibration"] + split["evaluation"]
                ]
                k_idx = pilot_idx[:K]
                usable_n = [n for n in n_values if n <= len(eval_idx)]
                if not usable_n:
                    continue
                cal_curve = exact_curve(scores[cal_idx], correct[cal_idx], usable_n)
                eval_curve = exact_curve(scores[eval_idx], correct[eval_idx], usable_n)
                raw_curve = exact_curve(scores[k_idx], correct[k_idx], usable_n)
                tail_curve = tail_smoothed_curve(scores[k_idx], correct[k_idx], usable_n)
                oracle_curve = exact_curve(scores, correct, usable_n)
                calibration_curves.append(cal_curve)
                pooled_pilot_scores.extend(scores[k_idx].tolist())
                pooled_pilot_correct.extend(correct[k_idx].tolist())
                pooled_eval_scores.extend(scores[eval_idx].tolist())
                pooled_eval_correct.extend(correct[eval_idx].tolist())
                per_problem_payload.append(
                    {
                        "problem_idx": pidx,
                        "p_eval": float(np.mean(correct[eval_idx])),
                        "difficulty_bin": difficulty_bin(float(np.mean(correct[eval_idx]))),
                        "cal_curve": cal_curve,
                        "eval_curve": eval_curve,
                        "raw_curve": raw_curve,
                        "tail_curve": tail_curve,
                        "oracle_curve": oracle_curve,
                    }
                )

            if not per_problem_payload:
                continue

            calibration_prior = {
                n: float(np.mean([curve[n] for curve in calibration_curves if n in curve]))
                for n in n_values
                if any(n in curve for curve in calibration_curves)
            }
            pooled_curve = exact_curve(
                np.asarray(pooled_pilot_scores, dtype=float),
                np.asarray(pooled_pilot_correct, dtype=bool),
                [n for n in n_values if n <= len(pooled_eval_scores)],
            )
            pooled_eval_curve = exact_curve(
                np.asarray(pooled_eval_scores, dtype=float),
                np.asarray(pooled_eval_correct, dtype=bool),
                [n for n in n_values if n <= len(pooled_eval_scores)],
            )
            for n, pred in pooled_curve.items():
                pooled_rows.append(
                    {
                        "model": model,
                        "K": K,
                        "N": n,
                        "estimator": "pooled",
                        "predicted": pred,
                        "actual": pooled_eval_curve[n],
                        "abs_error": abs(pred - pooled_eval_curve[n]),
                        "num_problems": len(per_problem_payload),
                    }
                )

            for item in per_problem_payload:
                weight = K / (K + 64.0)
                shrinkage_curve = {
                    n: float(weight * item["raw_curve"][n] + (1.0 - weight) * calibration_prior.get(n, item["raw_curve"][n]))
                    for n in item["eval_curve"]
                }
                for n, actual in item["eval_curve"].items():
                    candidates = {
                        "raw_plugin": item["raw_curve"][n],
                        "hierarchical_shrinkage": shrinkage_curve[n],
                        "tail_smoothed_moment": item["tail_curve"][n],
                        "oracle_full_distribution": item["oracle_curve"][n],
                    }
                    for estimator, pred in candidates.items():
                        cal_actual = item["cal_curve"].get(n)
                        if cal_actual is not None:
                            calibration_rows.append(
                                {
                                    "model": model,
                                    "problem_idx": item["problem_idx"],
                                    "K": K,
                                    "N": n,
                                    "estimator": estimator,
                                    "predicted": pred,
                                    "actual": cal_actual,
                                    "abs_error": abs(pred - cal_actual),
                                    "difficulty_bin": item["difficulty_bin"],
                                }
                            )
                        rows.append(
                            {
                                "model": model,
                                "problem_idx": item["problem_idx"],
                                "K": K,
                                "N": n,
                                "estimator": estimator,
                                "predicted": pred,
                                "actual": actual,
                                "abs_error": abs(pred - actual),
                                "difficulty_bin": item["difficulty_bin"],
                            }
                        )

    if not rows and not pooled_rows:
        raise SystemExit("No held-out rows generated. Measure complete maxed-out records first.")

    table_dir = HELDOUT_DIR / ("smoke_tables" if args.smoke else "tables")
    table_dir.mkdir(parents=True, exist_ok=True)
    detail_path = table_dir / "heldout_sample_complexity_detail.csv"
    calibration_detail_path = table_dir / "heldout_calibration_detail.csv"
    calibration_summary_path = table_dir / "heldout_calibration_summary.csv"
    locked_summary_path = table_dir / "heldout_locked_estimator_summary.csv"
    summary_path = table_dir / "heldout_sample_complexity_summary.csv"
    pooled_path = table_dir / "heldout_pooled_summary.csv"

    write_csv(detail_path, rows)
    grouped = aggregate_rows(rows, ["model", "estimator", "K", "N"])
    write_csv(summary_path, grouped)
    if calibration_rows:
        write_csv(calibration_detail_path, calibration_rows)
        calibration_grouped = aggregate_rows(calibration_rows, ["model", "estimator", "K", "N"])
        write_csv(calibration_summary_path, calibration_grouped)
        locked_rows = build_locked_estimator_summary(calibration_grouped, rows)
        write_csv(locked_summary_path, locked_rows)
    if pooled_rows:
        write_csv(pooled_path, pooled_rows)

    print(f"Wrote detail rows: {detail_path}")
    print(f"Wrote summary rows: {summary_path}")
    if calibration_rows:
        print(f"Wrote calibration summary rows: {calibration_summary_path}")
        print(f"Wrote locked-estimator rows: {locked_summary_path}")
    if pooled_rows:
        print(f"Wrote pooled rows: {pooled_path}")


def load_expanded_proxy_records(
    models: list[str] | None = None,
    min_samples: int = 256,
) -> list[tuple[str, int, np.ndarray, np.ndarray]]:
    summary = load_json(RESULTS_DIR / "du_aligned" / "expanded_pilot_summary.json", {})
    target_models = models or summary.get("models", [])
    root = RESULTS_DIR / "du_aligned" / "expanded_measurements"
    records = []
    for model in target_models:
        model_dir = root / model
        if not model_dir.exists():
            continue
        for rec_path in sorted(model_dir.glob("problem_*.json")):
            rec = load_json(rec_path, {})
            scores = np.asarray(rec.get("all_scores", []), dtype=float)
            correct = np.asarray(rec.get("all_correct", []), dtype=bool)
            if len(scores) == len(correct) and len(scores) >= min_samples:
                records.append((model, int(rec["problem_idx"]), scores, correct))
    return records


def cmd_analyze_expanded_proxy(args: argparse.Namespace) -> None:
    records = load_expanded_proxy_records(args.models, min_samples=args.min_samples)
    if not records:
        raise SystemExit("No expanded-pilot measurement records found.")

    k_values = args.k_values
    n_values = args.n_values
    tune_size = args.tune_size
    select_size = args.select_size
    rows = []
    selected_rows = []

    for seed in range(args.seeds):
        for model, problem_idx, scores, correct in records:
            n_total = len(scores)
            rng = np.random.default_rng(stable_seed("expanded_proxy_nested", seed, model, problem_idx))
            perm = rng.permutation(n_total)
            for K in k_values:
                if K + tune_size + select_size + min(n_values) > n_total:
                    continue
                pilot_idx = perm[:K]
                tune_idx = perm[K : K + tune_size]
                select_idx = perm[K + tune_size : K + tune_size + select_size]
                eval_idx = perm[K + tune_size + select_size :]
                usable_n = [
                    n for n in n_values
                    if n <= len(tune_idx) and n <= len(select_idx) and n <= len(eval_idx)
                ]
                if not usable_n:
                    continue

                curves = estimator_curves_from_slices(
                    scores[pilot_idx],
                    correct[pilot_idx],
                    scores[tune_idx],
                    correct[tune_idx],
                    usable_n,
                )
                select_curve = exact_curve(scores[select_idx], correct[select_idx], usable_n)
                eval_curve = exact_curve(scores[eval_idx], correct[eval_idx], usable_n)
                oracle_curve = exact_curve(scores, correct, usable_n)
                curves["oracle_full_distribution_diagnostic"] = oracle_curve

                for N in usable_n:
                    select_errors = {
                        estimator: abs(curve[N] - select_curve[N])
                        for estimator, curve in curves.items()
                        if not estimator.endswith("_diagnostic")
                    }
                    locked_estimator = min(select_errors, key=select_errors.get)
                    locked_pred = curves[locked_estimator][N]
                    selected_rows.append(
                        {
                            "model": model,
                            "problem_idx": problem_idx,
                            "seed": seed,
                            "K": K,
                            "N": N,
                            "selected_estimator": locked_estimator,
                            "selection_abs_error": select_errors[locked_estimator],
                            "eval_predicted": locked_pred,
                            "eval_actual": eval_curve[N],
                            "eval_abs_error": abs(locked_pred - eval_curve[N]),
                            "tune_size": tune_size,
                            "select_size": select_size,
                            "eval_size": len(eval_idx),
                        }
                    )
                    for estimator, curve in curves.items():
                        rows.append(
                            {
                                "model": model,
                                "problem_idx": problem_idx,
                                "seed": seed,
                                "K": K,
                                "N": N,
                                "estimator": estimator,
                                "select_abs_error": abs(curve[N] - select_curve[N]),
                                "eval_abs_error": abs(curve[N] - eval_curve[N]),
                                "diagnostic": estimator.endswith("_diagnostic"),
                                "tune_size": tune_size,
                                "select_size": select_size,
                                "eval_size": len(eval_idx),
                            }
                        )

    if not selected_rows:
        raise SystemExit("Expanded proxy analysis produced no rows; reduce K/tune/select sizes.")

    EXPANDED_PROXY_DIR.mkdir(parents=True, exist_ok=True)
    detail_path = EXPANDED_PROXY_DIR / "expanded_proxy_estimator_detail.csv"
    locked_detail_path = EXPANDED_PROXY_DIR / "expanded_proxy_locked_detail.csv"
    locked_summary_path = EXPANDED_PROXY_DIR / "expanded_proxy_locked_summary.csv"
    estimator_summary_path = EXPANDED_PROXY_DIR / "expanded_proxy_estimator_summary.csv"
    write_csv(detail_path, rows)
    write_csv(locked_detail_path, selected_rows)
    write_csv(
        estimator_summary_path,
        aggregate_rows(
            [
                {
                    "model": row["model"],
                    "estimator": row["estimator"],
                    "K": row["K"],
                    "N": row["N"],
                    "abs_error": row["eval_abs_error"],
                }
                for row in rows
            ],
            ["model", "estimator", "K", "N"],
        ),
    )
    locked_summary = aggregate_rows(
        [
            {
                "model": row["model"],
                "K": row["K"],
                "N": row["N"],
                "abs_error": row["eval_abs_error"],
            }
            for row in selected_rows
        ],
        ["model", "K", "N"],
    )
    all_rows = []
    grouped_all: dict[tuple[int, int], list[float]] = defaultdict(list)
    for row in selected_rows:
        grouped_all[(int(row["K"]), int(row["N"]))].append(float(row["eval_abs_error"]))
    for (K, N), vals in sorted(grouped_all.items()):
        arr = np.asarray(vals, dtype=float)
        all_rows.append(
            {
                "model": "ALL",
                "K": K,
                "N": N,
                "MAE": float(np.mean(arr)),
                "median_AE": float(np.median(arr)),
                "max_AE": float(np.max(arr)),
                "num_rows": int(len(arr)),
            }
        )
    write_csv(locked_summary_path, locked_summary + all_rows)

    summary = {
        "created_utc": utc_now(),
        "records": len(records),
        "models": sorted({r[0] for r in records}),
        "seeds": args.seeds,
        "K_values": k_values,
        "N_values": n_values,
        "tune_size": tune_size,
        "select_size": select_size,
        "selection_policy": "choose estimator with lowest selector-calibration error; evaluate once on independent held-out slice",
        "locked_summary": str(locked_summary_path),
    }
    write_json(EXPANDED_PROXY_DIR / "expanded_proxy_summary.json", summary)
    print(f"Wrote expanded proxy detail: {detail_path}")
    print(f"Wrote expanded proxy locked summary: {locked_summary_path}")
    print(json.dumps(summary, indent=2))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write(",".join(columns) + "\n")
        for row in rows:
            values = []
            for col in columns:
                value = row.get(col, "")
                if isinstance(value, str):
                    value = '"' + value.replace('"', '""') + '"'
                values.append(str(value))
            f.write(",".join(values) + "\n")


def build_locked_estimator_summary(
    calibration_summary_rows: list[dict[str, Any]],
    evaluation_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected: dict[tuple[Any, int, int], str] = {}
    for row in calibration_summary_rows:
        model = row["model"]
        k = int(row["K"])
        n = int(row["N"])
        estimator = row["estimator"]
        mae = float(row["MAE"])
        key = (model, k, n)
        current = selected.get(key)
        if current is None:
            selected[key] = estimator
            continue
        current_mae = next(
            float(r["MAE"])
            for r in calibration_summary_rows
            if r["model"] == model and int(r["K"]) == k and int(r["N"]) == n and r["estimator"] == current
        )
        if mae < current_mae:
            selected[key] = estimator

    errors: dict[tuple[Any, int, int, str], list[float]] = defaultdict(list)
    for row in evaluation_rows:
        model = row["model"]
        k = int(row["K"])
        n = int(row["N"])
        estimator = row["estimator"]
        locked = selected.get((model, k, n))
        if estimator != locked:
            continue
        key = (model, k, n, estimator)
        errors[key].append(float(row["abs_error"]))

    out = []
    by_all: dict[tuple[int, int], list[float]] = defaultdict(list)
    for (model, k, n, estimator), vals in sorted(errors.items()):
        arr = np.asarray(vals, dtype=float)
        out.append(
            {
                "model": model,
                "K": k,
                "N": n,
                "selected_estimator": estimator,
                "MAE": float(np.mean(arr)),
                "median_AE": float(np.median(arr)),
                "max_AE": float(np.max(arr)),
                "num_rows": int(len(arr)),
                "selection_source": "calibration",
            }
        )
        by_all[(k, n)].extend(float(v) for v in vals)

    for (k, n), vals in sorted(by_all.items()):
        arr = np.asarray(vals, dtype=float)
        out.append(
            {
                "model": "ALL",
                "K": k,
                "N": n,
                "selected_estimator": "mixed_by_model_calibration_choice",
                "MAE": float(np.mean(arr)),
                "median_AE": float(np.median(arr)),
                "max_AE": float(np.max(arr)),
                "num_rows": int(len(arr)),
                "selection_source": "calibration",
            }
        )
    return out


def aggregate_rows(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rows:
        key = tuple(row[k] for k in keys)
        groups[key].append(float(row["abs_error"]))
    out = []
    for key, values in sorted(groups.items()):
        arr = np.asarray(values, dtype=float)
        rec = {k: v for k, v in zip(keys, key)}
        rec.update(
            {
                "MAE": float(np.mean(arr)),
                "median_AE": float(np.median(arr)),
                "max_AE": float(np.max(arr)),
                "num_rows": int(len(arr)),
            }
        )
        out.append(rec)
    return out


def count_measurement_record_files(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in path.glob("problem_*.json"))


def count_measurement_records_at_target(path: Path, target_samples: int) -> int:
    if not path.exists():
        return 0
    count = 0
    for rec_path in path.glob("problem_*.json"):
        rec = load_json(rec_path, {})
        if int(rec.get("n_samples", 0) or 0) >= target_samples:
            count += 1
    return count


def max_measurement_samples(path: Path) -> int:
    if not path.exists():
        return 0
    max_samples = 0
    for rec_path in path.glob("problem_*.json"):
        rec = load_json(rec_path, {})
        try:
            max_samples = max(max_samples, int(rec.get("n_samples", 0) or 0))
        except (TypeError, ValueError):
            continue
    return max_samples


def existing_artifact_summary() -> dict[str, Any]:
    paper_claims = load_json(RESULTS_DIR / "du_aligned" / "du_experiment_results.json", {})
    live_judge = load_json(RESULTS_DIR / "du_aligned" / "live_judge_score_summary.json", {})
    expanded_pilot = load_json(RESULTS_DIR / "du_aligned" / "expanded_pilot_summary.json", {})
    cross = load_json(RESULTS_DIR / "benchmarks" / "cross_benchmark_summary.json", {})
    return {
        "du_aligned_status": "present" if paper_claims else "missing",
        "live_judge_status": live_judge.get("status", "missing"),
        "live_judge_pairs": live_judge.get("num_model_problem_pairs", 0),
        "expanded_pilot_max_samples": expanded_pilot.get("max_samples_per_problem"),
        "expanded_pilot_models": expanded_pilot.get("models", []),
        "cross_benchmarks": [row.get("benchmark") for row in cross.get("benchmarks", [])],
    }


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(val):
        return None
    return val


def pass_gate(
    name: str,
    status: str,
    value: Any,
    threshold: Any,
    evidence: str,
    requirement: str,
    action: str,
    category: str = "claim",
    claim_blocking: bool = True,
) -> dict[str, Any]:
    return {
        "name": name,
        "category": category,
        "status": status,
        "value": value,
        "threshold": threshold,
        "evidence": evidence,
        "requirement": requirement,
        "action": action,
        "claim_blocking": claim_blocking,
    }


def summarize_expanded_pilot_proxy() -> dict[str, Any]:
    calibrated_table = EXPANDED_PROXY_DIR / "expanded_proxy_locked_summary.csv"
    calibrated_rows = read_csv_rows(calibrated_table)
    if calibrated_rows:
        out = {"path": str(calibrated_table), "available": True, "by_K_N": {}, "selection": "nested_calibration_locked"}
        for row in calibrated_rows:
            if row.get("model") != "ALL":
                continue
            k = safe_float(row.get("K"))
            n = safe_float(row.get("N"))
            mae = safe_float(row.get("MAE"))
            if k is None or n is None or mae is None:
                continue
            out["by_K_N"][f"K{int(k)}_N{int(n)}"] = mae
        if out["by_K_N"]:
            return out

    table = RESULTS_DIR / "du_aligned" / "tables" / "expanded_pilot_sample_complexity.csv"
    rows = read_csv_rows(table)
    if not rows:
        return {"path": str(table), "available": False}
    summary: dict[tuple[int, int], list[float]] = defaultdict(list)
    for row in rows:
        k = safe_float(row.get("K"))
        n = safe_float(row.get("N"))
        mae = safe_float(row.get("heldout_MAE_mean"))
        if k is None or n is None or mae is None:
            continue
        summary[(int(k), int(n))].append(mae)
    out = {"path": str(table), "available": True, "by_K_N": {}, "selection": "legacy_raw_random_split"}
    for (k, n), vals in sorted(summary.items()):
        out["by_K_N"][f"K{k}_N{n}"] = float(np.mean(vals))
    return out


def summarize_maxed_heldout(smoke: bool) -> dict[str, Any]:
    table_dir = HELDOUT_DIR / ("smoke_tables" if smoke else "tables")
    locked_table = table_dir / "heldout_locked_estimator_summary.csv"
    locked_rows = read_csv_rows(locked_table)
    if locked_rows:
        best: dict[tuple[int, int], dict[str, Any]] = {}
        for row in locked_rows:
            if row.get("model") != "ALL":
                continue
            k = safe_float(row.get("K"))
            n = safe_float(row.get("N"))
            mae = safe_float(row.get("MAE"))
            estimator = row.get("selected_estimator", "")
            if k is None or n is None or mae is None:
                continue
            best[(int(k), int(n))] = {"MAE": mae, "estimator": estimator, "selection": "calibration_locked"}
        return {
            "path": str(locked_table),
            "available": bool(best),
            "best_by_K_N": {f"K{k}_N{n}": val for (k, n), val in sorted(best.items())},
        }

    table = table_dir / "heldout_sample_complexity_summary.csv"
    rows = read_csv_rows(table)
    if not rows:
        return {"path": str(table), "available": False}
    best: dict[tuple[int, int], dict[str, Any]] = {}
    for row in rows:
        k = safe_float(row.get("K"))
        n = safe_float(row.get("N"))
        mae = safe_float(row.get("MAE"))
        estimator = row.get("estimator", "")
        if k is None or n is None or mae is None:
            continue
        key = (int(k), int(n))
        prev = best.get(key)
        if prev is None or mae < prev["MAE"]:
            best[key] = {"MAE": mae, "estimator": estimator, "selection": "evaluation_best_fallback"}
    return {
        "path": str(table),
        "available": True,
        "best_by_K_N": {f"K{k}_N{n}": val for (k, n), val in sorted(best.items())},
    }


def heldout_coverage_audit(manifest: dict[str, Any]) -> dict[str, Any]:
    held = manifest.get("heldout_forecasting", {}) if manifest else {}
    target_samples = int(held.get("samples_per_problem", 4096) or 4096)
    math_problems = int(held.get("math_problems", 500) or 500)
    target_models = list(manifest.get("target_models", [])) if manifest else []
    live_models = set(manifest.get("live_collection_models", [])) if manifest else set()
    by_model = {}
    complete_records = 0
    any_records = 0
    for model in target_models:
        meas_dir = HELDOUT_MEASUREMENTS_DIR / model
        at_target = count_measurement_records_at_target(meas_dir, target_samples)
        any_count = count_measurement_record_files(meas_dir)
        max_samples = max_measurement_samples(meas_dir)
        complete_records += at_target
        any_records += any_count
        issues = []
        if model not in live_models:
            issues.append("inactive_live_endpoint")
        if at_target < math_problems:
            issues.append(f"records_at_target<{math_problems}")
        by_model[model] = {
            "live": model in live_models,
            "records_any_depth": any_count,
            "records_at_target": at_target,
            "max_samples_per_record": max_samples,
            "issues": issues,
        }
    required_records = len(target_models) * math_problems
    incomplete = {model: row for model, row in by_model.items() if row["issues"]}
    return {
        "required": {
            "models": len(target_models),
            "math_problems": math_problems,
            "samples_per_problem": target_samples,
            "records": required_records,
            "responses": required_records * target_samples,
        },
        "observed": {
            "records_any_depth": any_records,
            "records_at_target": complete_records,
            "max_samples_any_record": max((row["max_samples_per_record"] for row in by_model.values()), default=0),
        },
        "incomplete": incomplete,
    }


def live_judge_delta_summary() -> dict[str, Any]:
    summary = load_json(RESULTS_DIR / "du_aligned" / "live_judge_score_summary.json", {})
    if not summary:
        return {"available": False}
    return {
        "available": True,
        "delta": summary.get("n48_improvement_over_meanlogprob"),
        "pairs": summary.get("num_model_problem_pairs"),
        "models": summary.get("models", []),
        "path": str(RESULTS_DIR / "du_aligned" / "live_judge_score_summary.json"),
    }


def adaptive_delta_summary() -> dict[str, Any]:
    summary = load_json(RESULTS_DIR / "du_aligned" / "adaptive_allocation_summary.json", {})
    if not summary:
        return {"available": False}
    rows = summary.get("ranking_at_reference_budget", [])
    uniform = next((r for r in rows if r.get("policy") == "uniform"), None)
    moment = next((r for r in rows if r.get("policy") == "moment_law"), None)
    if not uniform or not moment:
        return {"available": True, "delta": None, "path": str(RESULTS_DIR / "du_aligned" / "adaptive_allocation_summary.json")}
    return {
        "available": True,
        "delta": float(moment["accuracy"]) - float(uniform["accuracy"]),
        "path": str(RESULTS_DIR / "du_aligned" / "adaptive_allocation_summary.json"),
    }


def cross_benchmark_gate_summary() -> dict[str, Any]:
    summary = load_json(RESULTS_DIR / "benchmarks" / "cross_benchmark_summary.json", {})
    rows = summary.get("benchmarks", []) if summary else []
    if not rows:
        return {"available": False, "benchmarks": [], "missing_benchmarks": list(DEFAULT_CROSS_BENCHMARKS)}
    benchmarks = [row.get("benchmark") for row in rows]
    missing = [benchmark for benchmark in DEFAULT_CROSS_BENCHMARKS if benchmark not in benchmarks]
    return {
        "available": True,
        "benchmarks": benchmarks,
        "rows": rows,
        "missing_benchmarks": missing,
        "num_benchmarks": len(rows),
        "all_law_mae": [
            safe_float(row.get("mean_exact_law_mae")) for row in rows
        ],
        "all_coverage": [
            safe_float(row.get("grading_coverage_rate")) for row in rows
        ],
        "all_nondegenerate": [
            int(row.get("nondegenerate_records", 0) or 0) for row in rows
        ],
        "path": str(RESULTS_DIR / "benchmarks" / "cross_benchmark_summary.json"),
    }


def cross_benchmark_scale_audit(cross: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    cross_manifest = manifest.get("cross_benchmark", {}) if manifest else {}
    target_task_limit = int(cross_manifest.get("tasks_per_benchmark", 500) or 500)
    target_samples = int(cross_manifest.get("samples_per_task", 128) or 128)
    target_models = len(cross_manifest.get("models", [])) or len(MODELS)
    rows_by_benchmark = {
        str(row.get("benchmark")): row
        for row in cross.get("rows", [])
        if row.get("benchmark")
    }
    incomplete: dict[str, dict[str, Any]] = {}
    complete = []
    for benchmark in DEFAULT_CROSS_BENCHMARKS:
        row = rows_by_benchmark.get(benchmark)
        if not row:
            incomplete[benchmark] = {"issues": ["missing_family"]}
            continue

        issues = []
        requested_limit = safe_float(row.get("requested_task_limit"))
        requested_tasks = safe_float(row.get("requested_tasks"))
        observed_tasks = safe_float(row.get("observed_task_count"))
        observed_models = safe_float(row.get("observed_model_count"))
        requested_samples = safe_float(row.get("requested_samples_per_task"))
        max_samples = safe_float(row.get("max_samples_per_record"))
        max_n = safe_float(row.get("max_N_evaluated"))
        coverage = safe_float(row.get("grading_coverage_rate"))
        expected_records = safe_float(row.get("expected_records"))
        measurement_records = safe_float(row.get("measurement_records"))

        if requested_limit is None or requested_limit < target_task_limit:
            issues.append(f"requested_task_limit<{target_task_limit}")
        if requested_tasks is None or requested_tasks <= 0:
            issues.append("requested_tasks_missing")
        if observed_tasks is None or (requested_tasks is not None and observed_tasks < requested_tasks):
            issues.append("not_all_requested_tasks_measured")
        if observed_models is None or observed_models < target_models:
            issues.append(f"observed_model_count<{target_models}")
        if requested_samples is None or requested_samples < target_samples:
            issues.append(f"requested_samples_per_task<{target_samples}")
        if max_samples is None or max_samples < target_samples:
            issues.append(f"max_samples_per_record<{target_samples}")
        if max_n is None or max_n < target_samples:
            issues.append(f"max_N_evaluated<{target_samples}")
        if coverage is None or coverage < 1.0:
            issues.append("grading_coverage<1.0")
        if expected_records is None or measurement_records is None or measurement_records < expected_records:
            issues.append("measurement_records<expected_records")

        if issues:
            incomplete[benchmark] = {
                "issues": issues,
                "requested_task_limit": requested_limit,
                "requested_tasks": requested_tasks,
                "observed_task_count": observed_tasks,
                "observed_model_count": observed_models,
                "requested_samples_per_task": requested_samples,
                "max_samples_per_record": max_samples,
                "max_N_evaluated": max_n,
                "grading_coverage_rate": coverage,
            }
        else:
            complete.append(benchmark)

    return {
        "required": {
            "benchmarks": list(DEFAULT_CROSS_BENCHMARKS),
            "task_limit": target_task_limit,
            "samples_per_task": target_samples,
            "model_count": target_models,
        },
        "complete": complete,
        "incomplete": incomplete,
    }


def build_claim_gate_report(smoke: bool = False) -> dict[str, Any]:
    manifest = load_json(manifest_path(smoke), {})
    du = load_json(RESULTS_DIR / "du_aligned" / "du_experiment_results.json", {})
    gates: list[dict[str, Any]] = []

    audit = du.get("audit", {}) if du else {}
    overall_mae = audit.get("overall_mae")
    if overall_mae is None:
        gates.append(
            pass_gate(
                "core_exact_law_mae",
                "MISSING",
                None,
                f"<= {IDEAL_GATES['exact_law_mae']}",
                str(RESULTS_DIR / "du_aligned" / "du_experiment_results.json"),
                "Full-estimate exact-law validation must be near numerical precision.",
                "Run or restore the Du-aligned audit bundle before making the core exact-law claim.",
            )
        )
    else:
        status = "PASS" if float(overall_mae) <= IDEAL_GATES["exact_law_mae"] else "FAIL"
        gates.append(
            pass_gate(
                "core_exact_law_mae",
                status,
                float(overall_mae),
                f"<= {IDEAL_GATES['exact_law_mae']}",
                str(RESULTS_DIR / "du_aligned" / "du_experiment_results.json"),
                "Full-estimate exact-law validation must be near numerical precision.",
                "If this fails, inspect tie handling, score alignment, and measurement/simulation parity; do not change labels or omit failures.",
            )
        )

    moment = du.get("moment_hierarchy", {}) if du else {}
    moment_maes = []
    for val in moment.get("by_n", {}).values():
        mae = safe_float(val.get("moment_mae"))
        if mae is not None:
            moment_maes.append(mae)
    max_moment_mae = max(moment_maes) if moment_maes else None
    gates.append(
        pass_gate(
            "moment_hierarchy_exactness",
            "PASS" if max_moment_mae is not None and max_moment_mae <= 1e-9 else "MISSING" if max_moment_mae is None else "FAIL",
            max_moment_mae,
            "<= 1e-9",
            str(RESULTS_DIR / "du_aligned" / "du_experiment_results.json"),
            "Moment-law prediction should be exact up to floating-point error when using the full empirical distribution.",
            "If this fails, fix the moment/rank formula before scaling any new experiments.",
        )
    )

    expanded_proxy = summarize_expanded_pilot_proxy()
    for k, threshold in [(128, IDEAL_GATES["heldout_k128_mae"]), (256, IDEAL_GATES["heldout_k256_mae"])]:
        key = f"K{k}_N8"
        val = expanded_proxy.get("by_K_N", {}).get(key)
        if val is None:
            status = "INFO"
            action = (
                "This proxy pool cannot certify the gate at this K; run the 4096-sample locked split "
                "and select estimators only on calibration data."
            )
        else:
            status = "PASS" if val <= threshold else "WARN"
            action = (
                "If this diagnostic proxy fails, do not overclaim tiny-pilot sufficiency; the next legitimate "
                "step is the full 4096-sample locked split, with estimator selection on calibration data only "
                "and final reporting on the independent evaluation pool."
            )
        gates.append(
            pass_gate(
                f"expanded_pilot_proxy_{key}",
                status,
                val,
                f"<= {threshold}",
                expanded_proxy.get("path"),
                "Existing 256-sample proxy is diagnostic pre-flight evidence for the pilot-forecasting target, not final proof.",
                action,
                category="diagnostic",
                claim_blocking=False,
            )
        )

    maxed_heldout = summarize_maxed_heldout(smoke)
    heldout_coverage = heldout_coverage_audit(manifest)
    gates.append(
        pass_gate(
            "maxed_heldout_coverage",
            "PASS" if manifest and not heldout_coverage["incomplete"] else "MISSING",
            heldout_coverage,
            "all target model/problem records measured at manifest sample depth",
            str(HELDOUT_MEASUREMENTS_DIR),
            "The maximum held-out forecasting claim requires complete manifest-scale coverage, not one deep record.",
            "Continue resumable collection until every target model/problem has 4096 measured samples, or explicitly scope the claim to the completed live subset.",
        )
    )
    for k, threshold in [
        (128, IDEAL_GATES["heldout_k128_mae"]),
        (256, IDEAL_GATES["heldout_k256_mae"]),
        (512, IDEAL_GATES["heldout_k512_mae"]),
    ]:
        best = maxed_heldout.get("best_by_K_N", {}).get(f"K{k}_N8")
        val = best.get("MAE") if isinstance(best, dict) else None
        if val is None:
            status = "MISSING"
        else:
            status = "PASS" if float(val) <= threshold else "FAIL"
        gates.append(
            pass_gate(
                f"maxed_heldout_K{k}_N8",
                status,
                val,
                f"<= {threshold}",
                maxed_heldout.get("path"),
                "Maxed-out held-out forecasting must hit the paper's sample-complexity gate on independent evaluation.",
                "If this fails, improve estimators on pilot/calibration only, add samples, or scope the claim to the observed sample-complexity curve.",
            )
        )

    live_judge = live_judge_delta_summary()
    delta = safe_float(live_judge.get("delta"))
    gates.append(
        pass_gate(
            "live_judge_delta_over_logprob",
            "PASS" if delta is not None and delta >= IDEAL_GATES["verifier_delta_over_logprob"] else "MISSING" if delta is None else "FAIL",
            delta,
            f">= {IDEAL_GATES['verifier_delta_over_logprob']}",
            live_judge.get("path"),
            "A real judge/verifier should beat mean logprob by at least +0.03 at high N.",
            "If this fails, add independent judges, calibrate judge ensembles on calibration data, and evaluate once on held-out pairs.",
        )
    )

    if manifest:
        expected_live_pairs = (
            len(manifest.get("live_collection_models", []))
            * int(manifest.get("real_verifier", {}).get("math_problems", 0) or 0)
        )
        observed_pairs = int(live_judge.get("pairs", 0) or 0)
        gates.append(
            pass_gate(
                "maxed_live_judge_coverage",
                "PASS" if expected_live_pairs and observed_pairs >= expected_live_pairs else "MISSING",
                observed_pairs,
                f">= {expected_live_pairs}",
                live_judge.get("path"),
                "Maxed-out live-judge claim needs complete model/problem coverage for the live panel.",
                "Run the live judge command queue to completion; keep existing 135-pair result scoped as a subset until then.",
            )
        )

    cross = cross_benchmark_gate_summary()
    maes = [m for m in cross.get("all_law_mae", []) if m is not None]
    coverages = [c for c in cross.get("all_coverage", []) if c is not None]
    cross_mae_pass = bool(maes) and all(m <= IDEAL_GATES["cross_benchmark_law_mae"] for m in maes)
    cross_cov_pass = bool(coverages) and all(c >= 1.0 for c in coverages)
    gates.append(
        pass_gate(
            "cross_benchmark_exact_law",
            "PASS" if cross_mae_pass and cross_cov_pass else "MISSING" if not cross.get("available") else "FAIL",
            {"mae": maes, "coverage": coverages},
            f"MAE <= {IDEAL_GATES['cross_benchmark_law_mae']} and coverage=1.0",
            cross.get("path"),
            "Every completed benchmark should hit the exact-law MAE and grading-coverage gate.",
            "If this fails, fix grader/score alignment before adding more benchmarks.",
        )
    )
    gates.append(
        pass_gate(
            "cross_benchmark_family_count",
            "PASS" if int(cross.get("num_benchmarks", 0) or 0) >= 6 else "MISSING",
            {
                "completed": cross.get("num_benchmarks", 0),
                "missing": cross.get("missing_benchmarks", DEFAULT_CROSS_BENCHMARKS),
            },
            ">= 6",
            cross.get("path"),
            "The maximum generalization claim needs at least six benchmark families.",
            "Run and measure the two missing families, MATH500 and MBPP-backed humaneval_mbpp, before claiming six-family generality.",
        )
    )
    cross_scale = cross_benchmark_scale_audit(cross, manifest)
    gates.append(
        pass_gate(
            "maxed_cross_benchmark_scale",
            "PASS" if cross.get("available") and not cross_scale["incomplete"] else "MISSING",
            cross_scale,
            "6 families at manifest task limit, sample depth, model count, and 100% measured coverage",
            cross.get("path"),
            "The maximum cross-benchmark claim requires the manifest-scale run, not smoke-scale family presence.",
            "Run the full cross-benchmark command queue with 500 requested tasks, 128 samples/task, and the full live model panel; then regenerate analysis.",
        )
    )

    adaptive = adaptive_delta_summary()
    adaptive_delta = safe_float(adaptive.get("delta"))
    gates.append(
        pass_gate(
            "adaptive_allocation_delta",
            "PASS" if adaptive_delta is not None and adaptive_delta > IDEAL_GATES["adaptive_delta_over_uniform"] else "MISSING" if adaptive_delta is None else "FAIL",
            adaptive_delta,
            "> 0",
            adaptive.get("path"),
            "Moment-law allocation should beat uniform under fixed budget.",
            "If this fails, tune allocation policies on calibration seeds only and preserve oracle/uniform baselines.",
        )
    )

    counts = defaultdict(int)
    claim_counts = defaultdict(int)
    diagnostic_counts = defaultdict(int)
    for gate in gates:
        counts[gate["status"]] += 1
        if gate.get("claim_blocking", True):
            claim_counts[gate["status"]] += 1
        else:
            diagnostic_counts[gate["status"]] += 1

    report = {
        "created_utc": utc_now(),
        "smoke": smoke,
        "manifest": str(manifest_path(smoke)),
        "policy": {
            "no_result_fabrication": True,
            "no_label_or_ground_truth_changes_to_pass_gates": True,
            "estimator_or_policy_tuning_must_use_pilot_or_calibration_only": True,
            "independent_evaluation_gates_are_final": True,
        },
        "ideal_gates": IDEAL_GATES,
        "summary": dict(counts),
        "claim_summary": dict(claim_counts),
        "diagnostic_summary": dict(diagnostic_counts),
        "all_passed": claim_counts.get("FAIL", 0) == 0 and claim_counts.get("MISSING", 0) == 0,
        "gates": gates,
    }
    return report


def write_claim_gate_report(report: dict[str, Any]) -> tuple[Path, Path]:
    suffix = ".smoke" if report.get("smoke") else ""
    json_path = OUT_DIR / f"claim_gate_report{suffix}.json"
    md_path = OUT_DIR / f"claim_gate_report{suffix}.md"
    write_json(json_path, report)

    lines = [
        "# Ideal Claim Gate Report",
        "",
        f"- Created UTC: `{report['created_utc']}`",
        f"- Smoke mode: `{report['smoke']}`",
        f"- All claim gates passed: `{report['all_passed']}`",
        f"- Summary: `{report['summary']}`",
        f"- Claim summary: `{report.get('claim_summary', {})}`",
        f"- Diagnostic summary: `{report.get('diagnostic_summary', {})}`",
        "",
        "## Integrity Policy",
        "",
        "- Do not fabricate responses, scores, labels, or confidence intervals.",
        "- Do not change ground truth or omit failing slices to pass a gate.",
        "- Tune estimators and allocation policies only on pilot/calibration data.",
        "- Treat independent evaluation gates as final evidence for paper claims.",
        "",
        "## Gates",
        "",
        "| Gate | Category | Claim Blocking | Status | Value | Threshold | Evidence | Required Action |",
        "| --- | --- | ---: | --- | ---: | --- | --- | --- |",
    ]
    for gate in report["gates"]:
        value = gate["value"]
        if isinstance(value, (dict, list)):
            value_text = json.dumps(value)
        else:
            value_text = str(value)
        lines.append(
            f"| {gate['name']} | {gate.get('category', 'claim')} | {gate.get('claim_blocking', True)} | "
            f"{gate['status']} | {value_text} | {gate['threshold']} | "
            f"`{gate['evidence']}` | {gate['action']} |"
        )
    lines.extend(
        [
            "",
            "## Claim Rule",
            "",
            "Only claim-blocking gates marked `PASS` can support maximum-strength paper wording. "
            "Any claim-blocking `FAIL` or `MISSING` gate must either be fixed with more evidence or scoped as a limitation. "
            "Diagnostic rows are pre-flight checks: disclose them, but do not treat them as final independent-evaluation evidence.",
            "",
        ]
    )
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def cmd_gate_report(args: argparse.Namespace) -> None:
    report = build_claim_gate_report(smoke=args.smoke)
    json_path, md_path = write_claim_gate_report(report)
    print(f"Wrote gate JSON: {json_path}")
    print(f"Wrote gate report: {md_path}")
    print(f"Gate summary: {report['summary']}")
    if args.strict and not report["all_passed"]:
        raise SystemExit(2)


def cmd_status(args: argparse.Namespace) -> None:
    path = manifest_path(args.smoke)
    manifest = load_json(path, {})
    if not manifest:
        raise SystemExit(f"Manifest not found: {path}. Run prepare first.")

    held = manifest["heldout_forecasting"]
    split = held.get("split", {})
    pilot_depth = int(split.get("pilot", 1024) or 1024)
    calibration_depth = pilot_depth + int(split.get("calibration", 1024) or 1024)
    target_depth = int(held.get("samples_per_problem", 0) or 0)
    model_rows = []
    for model in manifest["target_models"]:
        meas_dir = HELDOUT_MEASUREMENTS_DIR / model
        summary = load_json(meas_dir / "measurement_summary.json", {})
        any_records = count_measurement_record_files(meas_dir)
        target_records = count_measurement_records_at_target(
            meas_dir,
            target_depth,
        )
        model_rows.append(
            {
                "model": model,
                "live": model in manifest["live_collection_models"],
                "measurement_records_any_n": any_records,
                "records_at_pilot_depth": count_measurement_records_at_target(meas_dir, pilot_depth),
                "records_at_calibration_depth": count_measurement_records_at_target(meas_dir, calibration_depth),
                "max_samples_per_record": max_measurement_samples(meas_dir),
                "measurement_records_at_manifest_n": target_records,
                "last_measured": summary.get("finished_utc", "N/A"),
            }
        )

    table_dir = HELDOUT_DIR / ("smoke_tables" if args.smoke else "tables")
    detail_csv = table_dir / "heldout_sample_complexity_detail.csv"
    summary_csv = table_dir / "heldout_sample_complexity_summary.csv"
    queue_path = OUT_DIR / ("command_queue.smoke.jsonl" if args.smoke else "command_queue.jsonl")
    next_slice: tuple[str, int] | None = None
    for model in manifest["target_models"]:
        if model not in MODELS:
            continue
        meas_dir = HELDOUT_MEASUREMENTS_DIR / model
        for pidx in range(int(held.get("math_problems", 0) or 0)):
            rec = load_json(meas_dir / f"problem_{pidx}.json", {})
            if int(rec.get("n_samples", 0) or 0) < target_depth:
                next_slice = (model, pidx)
                break
        if next_slice:
            break

    text_lines = [
        "# Maxed-Out Campaign Status",
        "",
        f"- Manifest: `{path}`",
        f"- Smoke mode: `{manifest.get('smoke')}`",
        f"- Created UTC: `{manifest.get('created_utc')}`",
        f"- Repo commit: `{manifest.get('repo', {}).get('commit')}`",
        f"- Target models: {len(manifest.get('target_models', []))}",
        f"- Live collection models: {len(manifest.get('live_collection_models', []))}",
        f"- Inactive target models: {manifest.get('inactive_target_models', [])}",
        "",
        "## Planned Scale",
        "",
        f"- Held-out MATH responses: {held.get('target_response_count', 0):,}",
        f"- Held-out split: {held.get('split')}",
        f"- Real-verifier judgments, all target models: {manifest['real_verifier'].get('target_judgment_count', 0):,}",
        f"- Real-verifier judgments, live-runnable panel: {manifest['real_verifier'].get('live_collection_judgment_count', 0):,}",
        f"- Cross-benchmark responses, desired families: {manifest['cross_benchmark'].get('target_response_count', 0):,}",
        f"- Cross-benchmark responses, existing runner support: {manifest['cross_benchmark'].get('supported_existing_runner_response_count', 0):,}",
        "",
        "## Current Maxed-Out Artifacts",
        "",
        f"- Command queue: `{queue_path}`",
        f"- Held-out detail table exists: `{detail_csv.exists()}`",
        f"- Held-out summary table exists: `{summary_csv.exists()}`",
        "",
        f"| Model | Live endpoint | Any measurement records | Records >= {pilot_depth} | Records >= {calibration_depth} | Max samples/record | Records at manifest n | Last measured |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in model_rows:
        text_lines.append(
            f"| {row['model']} | {row['live']} | {row['measurement_records_any_n']} | "
            f"{row['records_at_pilot_depth']} | {row['records_at_calibration_depth']} | "
            f"{row['max_samples_per_record']} | {row['measurement_records_at_manifest_n']} | {row['last_measured']} |"
        )

    existing = existing_artifact_summary()
    text_lines.extend(
        [
            "",
            "## Existing Closure Evidence",
            "",
            f"- Du-aligned bundle: {existing['du_aligned_status']}",
            f"- Existing expanded pilot max samples/problem: {existing['expanded_pilot_max_samples']}",
            f"- Existing expanded pilot models: {existing['expanded_pilot_models']}",
            f"- Existing live judge status: {existing['live_judge_status']} ({existing['live_judge_pairs']} pairs)",
            f"- Existing cross benchmarks: {existing['cross_benchmarks']}",
            "",
            "## Next Commands",
            "",
        ]
    )
    if next_slice:
        next_model, next_pidx = next_slice
        text_lines.extend(
            [
                "Continue with the next real 4096-sample held-out slice:",
                "",
                "```bash",
                (
                    "python experiments/17_maxed_out_campaign.py collect-math "
                    f"--model {next_model} --n-problems {next_pidx + 1} --problem-indices {next_pidx} "
                    "--target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 32"
                ),
                (
                    "python experiments/17_maxed_out_campaign.py measure-math "
                    f"--model {next_model} --n-problems {next_pidx + 1} --problem-indices {next_pidx} "
                    "--target-samples 4096"
                ),
                f"python experiments/17_maxed_out_campaign.py analyze-heldout --models {next_model}",
                "```",
            ]
        )
    else:
        text_lines.append("All active held-out slices in the manifest are measured at target depth.")

    status_path = OUT_DIR / ("campaign_status.smoke.md" if args.smoke else "campaign_status.md")
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text("\n".join(text_lines) + "\n", encoding="utf-8")
    print(f"Wrote status: {status_path}")
    print("\n".join(text_lines[:18]))


def add_shared_prepare_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--smoke", action="store_true", help="Use tiny local-safe manifest values.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing manifest.")
    parser.add_argument("--model-source", choices=["active", "measured", "union"], default="measured")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--math-problems", type=int, default=500)
    parser.add_argument("--heldout-samples", type=int, default=4096)
    parser.add_argument("--pilot-samples", type=int, default=1024)
    parser.add_argument("--calibration-samples", type=int, default=1024)
    parser.add_argument("--evaluation-samples", type=int, default=2048)
    parser.add_argument("--k-values", nargs="*", type=int, default=DEFAULT_K_VALUES)
    parser.add_argument("--n-values", nargs="*", type=int, default=DEFAULT_N_VALUES)
    parser.add_argument("--judge-samples", type=int, default=256)
    parser.add_argument("--judge-labels", nargs="*", default=DEFAULT_JUDGE_LABELS)
    parser.add_argument("--v-values", nargs="*", type=int, default=[1, 2, 3, 5])
    parser.add_argument("--manual-audit-target", type=int, default=5000)
    parser.add_argument("--cross-benchmarks", nargs="*", default=DEFAULT_CROSS_BENCHMARKS)
    parser.add_argument("--cross-tasks", type=int, default=500)
    parser.add_argument("--cross-samples", type=int, default=128)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Maxed-out campaign orchestrator.")
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="Write locked manifest and command queue.")
    add_shared_prepare_args(prepare)
    prepare.set_defaults(func=cmd_prepare)

    status = sub.add_parser("status", help="Write campaign status Markdown.")
    status.add_argument("--smoke", action="store_true")
    status.set_defaults(func=cmd_status)

    gate_report = sub.add_parser("gate-report", help="Evaluate ideal paper-claim metric gates.")
    gate_report.add_argument("--smoke", action="store_true")
    gate_report.add_argument("--strict", action="store_true", help="Exit nonzero unless every gate passes.")
    gate_report.set_defaults(func=cmd_gate_report)

    collect = sub.add_parser("collect-math", help="Collect MATH samples into the shared raw cache.")
    collect.add_argument("--model", required=True)
    collect.add_argument("--n-problems", type=int, default=500)
    collect.add_argument("--problem-indices", default=None, help="Comma/range list, e.g. 0,2,5-9")
    collect.add_argument("--target-samples", type=int, default=4096)
    collect.add_argument("--batch-size", type=int, default=4)
    collect.add_argument("--workers", type=int, default=1)
    collect.add_argument("--request-delay", type=float, default=0.0)
    collect.add_argument("--max-job-attempts", type=int, default=8)
    collect.add_argument("--temperature", type=float, default=0.7)
    collect.add_argument("--max-tokens", type=int, default=2048)
    collect.add_argument("--timeout", type=float, default=180.0)
    collect.add_argument("--progress-every", type=int, default=10)
    collect.add_argument("--dry-run", action="store_true")
    collect.set_defaults(func=cmd_collect_math)

    measure = sub.add_parser("measure-math", help="Measure complete MATH raw-cache records.")
    measure.add_argument("--model", required=True)
    measure.add_argument("--n-problems", type=int, default=500)
    measure.add_argument("--problem-indices", default=None)
    measure.add_argument("--target-samples", type=int, default=4096)
    measure.add_argument("--progress-every", type=int, default=25)
    measure.set_defaults(func=cmd_measure_math)

    analyze = sub.add_parser("analyze-heldout", help="Analyze held-out sample complexity from measurements.")
    analyze.add_argument("--models", nargs="*", default=None)
    analyze.add_argument("--smoke", action="store_true")
    analyze.set_defaults(func=cmd_analyze_heldout)

    expanded_proxy = sub.add_parser(
        "analyze-expanded-proxy",
        help="Run nested calibration/evaluation analysis on existing 256-sample expanded-pilot records.",
    )
    expanded_proxy.add_argument("--models", nargs="*", default=None)
    expanded_proxy.add_argument("--seeds", type=int, default=20)
    expanded_proxy.add_argument("--k-values", nargs="*", type=int, default=[8, 16, 32, 48, 64, 96, 128, 160, 192])
    expanded_proxy.add_argument("--n-values", nargs="*", type=int, default=[2, 8, 16, 32, 48, 64])
    expanded_proxy.add_argument("--tune-size", type=int, default=32)
    expanded_proxy.add_argument("--select-size", type=int, default=32)
    expanded_proxy.add_argument("--min-samples", type=int, default=256)
    expanded_proxy.set_defaults(func=cmd_analyze_expanded_proxy)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
