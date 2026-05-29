"""
Structured judge-score diagnostic experiment.

Reads raw cached responses and produces structured judgment scores.
Supports checkpointing: writes each judgment to JSONL immediately.

If a live judge_fn is supplied, each response is formatted as a judge prompt.
The default path is a deterministic no-reference heuristic plus a
reference-based diagnostic; those defaults are not live model-judge calls.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import MODELS, N_SAMPLES, RESULTS_DIR, DATA_DIR, NIM_API_KEYS, NIM_BASE_URL, get_next_key
from src.nim_client import _cache_path
from src.utils import extract_boxed_answer
from src.model_judge import (
    select_judgment_subset,
    load_checkpoint,
    format_no_reference_prompt,
    format_reference_prompt,
    parse_judgment,
    JUDGE_RUBRIC,
)
from src.feature_extraction import load_math500

OUT_DIR = RESULTS_DIR / "du_aligned" / "model_judge"
JUDGED_PATH = OUT_DIR / "judged_responses.jsonl"
LIVE_JUDGED_PATH = OUT_DIR / "judged_responses_live.jsonl"
MANIFEST_PATH = OUT_DIR / "judge_subset_manifest.json"
LIVE_MANIFEST_PATH = OUT_DIR / "judge_subset_manifest_live.json"
RUBRIC_PATH = OUT_DIR / "judge_rubric.md"

TARGET_MODELS = ["3B", "8B", "70B"]
N_JUDGE_PROBLEMS = 50
NO_REFERENCE_MODE = "heuristic_no_reference"
REFERENCE_MODE = "reference_based_diagnostic"
LIVE_NO_REFERENCE_MODE = "live_no_reference"
LIVE_REFERENCE_MODE = "live_reference_diagnostic"
LIVE_JUDGE_TIMEOUT_SECONDS = 90.0
LIVE_JUDGE_REQUEST_DELAY_SECONDS = 8.0
LIVE_JUDGE_RATE_LIMIT_COOLDOWN_SECONDS = 90.0
LIVE_JUDGE_MAX_TASK_ATTEMPTS = 8


def build_work_items(
    problems: list[dict],
    target_models: list[str] | None = None,
    n_judge_problems: int | None = None,
) -> tuple[list[dict], dict]:
    measurements_dir = RESULTS_DIR / "measurements"
    items, manifest = select_judgment_subset(
        models=MODELS,
        problems=problems,
        measurements_dir=measurements_dir,
        target_models=target_models or TARGET_MODELS,
        n_problems=n_judge_problems or N_JUDGE_PROBLEMS,
    )
    return items, manifest


def run_judgments(
    problems: list[dict],
    items: list[dict],
    judge_fn=None,
    batch_size: int = 50,
    output_path: Path = JUDGED_PATH,
    judge_modes: list[tuple[str, str]] | None = None,
    max_samples: int = N_SAMPLES,
) -> int:
    if judge_modes is None:
        judge_modes = [
            (NO_REFERENCE_MODE, "no_reference"),
            (REFERENCE_MODE, "reference"),
        ]

    done = load_checkpoint(output_path)
    n_samples = min(max_samples, N_SAMPLES)
    total = len(items) * n_samples * len(judge_modes)
    completed = 0
    skipped = 0

    with open(output_path, "a", encoding="utf-8") as f:
        for item_idx, item in enumerate(items):
            model_key = item["model_key"]
            nim_name = item["nim_name"]
            problem_idx = item["problem_idx"]
            problem_text = item["problem_text"]
            ground_truth = item["ground_truth"]

            for sample_idx in range(n_samples):
                for judge_mode, prompt_kind in judge_modes:
                    key = f"{model_key}:{problem_idx}:{sample_idx}:{judge_mode}"
                    if key in done:
                        skipped += 1
                        continue

                    cache_path = _cache_path(nim_name, problem_text, sample_idx)
                    if not cache_path.exists():
                        continue

                    try:
                        raw = json.loads(cache_path.read_text(encoding="utf-8"))
                    except (json.JSONDecodeError, OSError):
                        continue

                    content = raw.get("content", "")
                    extracted = extract_boxed_answer(content)

                    if prompt_kind == "no_reference":
                        prompt = format_no_reference_prompt(problem_text, content, extracted)
                    else:
                        prompt = format_reference_prompt(problem_text, content, extracted, ground_truth)

                    logprobs_list = raw.get("logprobs") or []
                    mean_lp = None
                    if logprobs_list:
                        mean_lp = sum(t["logprob"] for t in logprobs_list) / len(logprobs_list)

                    if judge_fn is not None:
                        judgment = judge_fn(prompt)
                    else:
                        judgment = _default_judge(
                            content, extracted, ground_truth, judge_mode,
                            mean_logprob=mean_lp, logprobs_list=logprobs_list,
                        )

                    if judgment is None:
                        judgment = {
                            "judge_score": 0.5,
                            "judge_label": "uncertain",
                            "reason_short": "judgment parse failed",
                            "format_valid": extracted is not None,
                            "solution_validity": 0.5,
                            "final_answer_confidence": 0.5,
                        }

                    record = {
                        "model": model_key,
                        "problem_idx": problem_idx,
                        "sample_idx": sample_idx,
                        "judge_mode": judge_mode,
                        **judgment,
                    }

                    f.write(json.dumps(record) + "\n")
                    completed += 1
                    del raw

                    if completed % batch_size == 0:
                        f.flush()
                        pct = (completed + skipped) / total * 100
                        print(f"  Progress: {completed} judged, {skipped} skipped ({pct:.1f}%)")

        f.flush()

    return completed


def _build_calibrated_judge(measurements_dir: Path, target_models: list[str]):
    """Build a calibrated no-reference judge using logistic regression on training problems."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    import re as _re

    X_train = []
    y_train = []

    for model_key in target_models:
        model_dir = measurements_dir / model_key
        nim_name = MODELS.get(model_key, "")

        for pidx in range(500):
            if pidx % 5 == 0:
                continue
            meas_path = model_dir / f"problem_{pidx}.json"
            if not meas_path.exists():
                continue
            try:
                meas = json.loads(meas_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue

            all_scores = meas.get("all_scores", [])
            all_correct = meas.get("all_correct", [])
            for sidx in range(min(len(all_scores), len(all_correct))):
                score = all_scores[sidx]
                if not np.isfinite(score):
                    continue
                X_train.append([score])
                y_train.append(int(bool(all_correct[sidx])))

    if len(X_train) < 10:
        return None, None

    X_train = np.array(X_train, dtype=float)
    y_train = np.array(y_train, dtype=float)

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_train)
    clf = LogisticRegression(C=1.0, max_iter=1000)
    clf.fit(X_s, y_train)

    return clf, scaler


_calibrated_judge_cache = {}


def _default_judge(
    content: str,
    extracted_answer: str | None,
    ground_truth: str,
    judge_mode: str,
    mean_logprob: float | None = None,
    logprobs_list: list | None = None,
) -> dict:
    import re as _re

    has_boxed = extracted_answer is not None
    has_reasoning = len(content) > 100
    content_lower = content.lower()

    n_tokens = len(logprobs_list) if logprobs_list else 0
    if logprobs_list and mean_logprob is None:
        mean_logprob = sum(t["logprob"] for t in logprobs_list) / len(logprobs_list)

    has_therefore = "therefore" in content_lower
    has_equations = content.count("$$") >= 2 or "\\[" in content or "\\begin{" in content
    n_latex = sum(content.count(m) for m in ["\\frac", "\\sqrt", "\\sum", "\\int"])
    has_steps = content.count("\n") > 3 or content.count("\\\\") > 2
    has_contradiction = ("but" in content_lower and "incorrect" in content_lower) or \
                       ("however" in content_lower and "mistake" in content_lower)

    if judge_mode == REFERENCE_MODE:
        from src.utils import check_correctness
        is_correct, _ = check_correctness(content, ground_truth)
        if is_correct:
            base = 0.85
            if has_boxed:
                base += 0.10
            if has_reasoning and has_steps:
                base += 0.05
            score = min(1.0, base)
            label = "likely_correct"
        else:
            base = 0.15
            if has_boxed and has_reasoning:
                base = 0.25
            if has_contradiction:
                base = max(base - 0.1, 0.0)
            score = base
            label = "likely_incorrect"
    else:
        features = []
        feature_score = 0.0

        if has_boxed:
            feature_score += 0.15
        if has_reasoning:
            feature_score += 0.10
        if has_steps:
            feature_score += 0.05
        if has_equations:
            feature_score += 0.05
        if has_therefore:
            feature_score += 0.03
        if n_latex > 3:
            feature_score += 0.02
        if has_contradiction:
            feature_score -= 0.10
        if len(content) < 50:
            feature_score -= 0.15
        if extracted_answer and len(extracted_answer) > 200:
            feature_score -= 0.05

        logprob_signal = 0.5
        if mean_logprob is not None and np.isfinite(mean_logprob):
            lp_normalized = max(0.0, min(1.0, (mean_logprob + 5.0) / 5.0))
            logprob_signal = lp_normalized

        score = 0.4 * logprob_signal + 0.6 * (0.5 + feature_score)
        score = max(0.0, min(1.0, score))

        if score >= 0.65:
            label = "likely_correct"
        elif score >= 0.4:
            label = "uncertain"
        else:
            label = "likely_incorrect"

    reason_parts = []
    if has_boxed:
        reason_parts.append("boxed")
    else:
        reason_parts.append("no_box")
    if has_reasoning:
        reason_parts.append("reasoning")
    if has_equations:
        reason_parts.append("equations")
    reason_parts.append(f"len={len(content)}")

    return {
        "judge_score": round(score, 4),
        "judge_label": label,
        "reason_short": ", ".join(reason_parts)[:100],
        "format_valid": has_boxed,
        "solution_validity": round(score, 4),
        "final_answer_confidence": round(score, 4),
    }


def build_live_judge_fn(judge_model: str, max_retries: int = 4):
    if not NIM_API_KEYS:
        raise RuntimeError("No NIM keys available. Set NIM_API_KEYS or provide C:\\Users\\wangz\\MIRROR\\.env.")

    from openai import OpenAI

    def _judge(prompt: str) -> dict | None:
        last_error = None
        for attempt in range(max_retries):
            key = get_next_key()
            client = OpenAI(base_url=NIM_BASE_URL, api_key=key)
            try:
                response = client.chat.completions.create(
                    model=judge_model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a careful mathematical solution judge. Return only the requested JSON object.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                    max_tokens=512,
                )
                text = response.choices[0].message.content or ""
                parsed = parse_judgment(text)
                if parsed is not None:
                    parsed["judge_source"] = f"live_nim:{judge_model}"
                    return parsed
                last_error = "parse_failed"
            except Exception as exc:
                last_error = str(exc)[:160]
            time.sleep(min(2 ** attempt, 20))
        return {
            "judge_score": 0.5,
            "judge_label": "uncertain",
            "reason_short": f"live judge failed: {last_error}",
            "format_valid": False,
            "solution_validity": 0.5,
            "final_answer_confidence": 0.5,
            "judge_source": f"live_nim:{judge_model}:failed",
        }

    return _judge


def format_no_reference_batch_prompt(problem_text: str, samples: list[dict]) -> str:
    blocks = []
    for sample in samples:
        extracted = sample.get("extracted")
        answer_line = (
            f"Extracted final answer: {extracted}"
            if extracted
            else "No clearly extracted final answer found."
        )
        blocks.append(
            f"""### sample_idx: {sample['sample_idx']}
{answer_line}

Candidate response:
{sample['content']}"""
        )

    return f"""You are a math solution quality judge. Evaluate each candidate response to the same math problem independently.
Do NOT assume you know the correct answer. Judge based on the mathematical reasoning quality.
Do NOT compare candidates to each other.

## Math Problem
{problem_text}

## Candidate Responses
{chr(10).join(blocks)}

## Instructions
Score each response on a scale of 0.0 to 1.0:
- 1.0: Solution appears mathematically correct with clear, valid reasoning leading to a supported final answer.
- 0.7-0.9: Mostly correct reasoning with minor issues; final answer likely correct.
- 0.4-0.6: Partially correct reasoning but uncertain; answer may or may not be right.
- 0.1-0.3: Significant errors in reasoning; answer likely wrong.
- 0.0: Clearly incorrect, incoherent, or no usable final answer.

Penalize: unsupported final answers, contradictions between reasoning and answer, missing steps.

Respond with ONLY a JSON array (no markdown fences). Include one object per sample_idx:
[
  {{"sample_idx": <int>, "score": <float 0.0-1.0>, "label": "<likely_correct|uncertain|likely_incorrect>", "reason": "<max 18 words>", "format_valid": <true|false>, "solution_validity": <float 0.0-1.0>, "answer_confidence": <float 0.0-1.0>}}
]"""


def format_reference_batch_prompt(problem_text: str, ground_truth: str, samples: list[dict]) -> str:
    blocks = []
    for sample in samples:
        extracted = sample.get("extracted")
        answer_line = (
            f"Extracted final answer: {extracted}"
            if extracted
            else "No clearly extracted final answer found."
        )
        blocks.append(
            f"""### sample_idx: {sample['sample_idx']}
{answer_line}

Candidate response:
{sample['content']}"""
        )

    return f"""You are a math solution quality judge. Evaluate each candidate response to the same math problem independently.
You are given the correct answer for reference.
Do NOT compare candidates to each other.

## Math Problem
{problem_text}

## Correct Answer
{ground_truth}

## Candidate Responses
{chr(10).join(blocks)}

## Instructions
Score each response on a scale of 0.0 to 1.0:
- 1.0: Final answer matches the correct answer and reasoning is valid.
- 0.7-0.9: Final answer is correct (possibly in different form) with mostly sound reasoning.
- 0.4-0.6: Answer is close but not equivalent, or correct answer with flawed reasoning.
- 0.1-0.3: Answer is wrong but reasoning shows partial understanding.
- 0.0: Completely wrong answer with no valid reasoning.

Respond with ONLY a JSON array (no markdown fences). Include one object per sample_idx:
[
  {{"sample_idx": <int>, "score": <float 0.0-1.0>, "label": "<likely_correct|uncertain|likely_incorrect>", "reason": "<max 18 words>", "format_valid": <true|false>, "solution_validity": <float 0.0-1.0>, "answer_confidence": <float 0.0-1.0>}}
]"""


def parse_batch_judgments(text: str, expected_sample_indices: list[int]) -> dict[int, dict]:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [line for line in lines if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()

    candidates = [text]
    start = text.find("[")
    end = text.rfind("]")
    if 0 <= start < end:
        candidates.append(text[start : end + 1])

    import re

    def _sample_idx(value, fallback: int | None = None) -> int | None:
        if value is None:
            return fallback
        try:
            return int(value)
        except (TypeError, ValueError):
            match = re.search(r"\d+", str(value))
            if match:
                return int(match.group(0))
        return fallback

    def _convert(obj: dict) -> dict | None:
        try:
            score = float(obj.get("score", obj.get("judge_score")))
        except (TypeError, ValueError):
            return None
        score = max(0.0, min(1.0, score))
        return {
            "judge_score": score,
            "judge_label": str(obj.get("label", obj.get("judge_label", "uncertain"))),
            "reason_short": str(obj.get("reason", obj.get("reason_short", "")))[:100],
            "format_valid": bool(obj.get("format_valid", True)),
            "solution_validity": float(obj.get("solution_validity", score)),
            "final_answer_confidence": float(obj.get("answer_confidence", obj.get("final_answer_confidence", score))),
        }

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue

        if isinstance(parsed, dict) and isinstance(parsed.get("judgments"), list):
            parsed = parsed["judgments"]

        results: dict[int, dict] = {}
        if isinstance(parsed, list):
            for idx, obj in enumerate(parsed):
                if not isinstance(obj, dict):
                    continue
                sample_idx = _sample_idx(
                    obj.get("sample_idx", obj.get("sample_id", obj.get("id"))),
                    expected_sample_indices[idx] if idx < len(expected_sample_indices) else None,
                )
                if sample_idx not in expected_sample_indices:
                    continue
                converted = _convert(obj)
                if converted is not None:
                    results[sample_idx] = converted
        elif isinstance(parsed, dict):
            for key, obj in parsed.items():
                if not isinstance(obj, dict):
                    continue
                sample_idx = _sample_idx(obj.get("sample_idx", key))
                if sample_idx not in expected_sample_indices:
                    continue
                converted = _convert(obj)
                if converted is not None:
                    results[sample_idx] = converted

        if results:
            return results

    return {}


def run_live_judgments_batch_parallel(
    problems: list[dict],
    items: list[dict],
    judge_model: str,
    output_path: Path = LIVE_JUDGED_PATH,
    judge_modes: list[tuple[str, str]] | None = None,
    max_samples: int = N_SAMPLES,
    workers: int | None = None,
    samples_per_call: int = 4,
    progress_batch_size: int = 50,
    request_delay: float = LIVE_JUDGE_REQUEST_DELAY_SECONDS,
    rate_limit_cooldown: float = LIVE_JUDGE_RATE_LIMIT_COOLDOWN_SECONDS,
    max_task_attempts: int = LIVE_JUDGE_MAX_TASK_ATTEMPTS,
) -> int:
    if not NIM_API_KEYS:
        raise RuntimeError("No NIM keys available. Set NIM_API_KEYS or provide C:\\Users\\wangz\\MIRROR\\.env.")
    if judge_modes is None:
        judge_modes = [(LIVE_NO_REFERENCE_MODE, "no_reference")]

    from openai import OpenAI
    from queue import Empty, Queue
    import threading
    from config import RATE_LIMIT_PER_KEY

    done = load_checkpoint(output_path)
    n_samples = min(max_samples, N_SAMPLES)
    samples_per_call = max(1, samples_per_call)
    tasks: Queue = Queue()
    pending_groups: list[dict] = []
    skipped = 0
    done_by_pair: dict[tuple[str, int, str], int] = {}
    for key in done:
        try:
            model, pidx, _sample_idx, judge_mode = key.split(":", 3)
            pair_key = (model, int(pidx), judge_mode)
            done_by_pair[pair_key] = done_by_pair.get(pair_key, 0) + 1
        except ValueError:
            continue

    for item in items:
        model_key = item["model_key"]
        nim_name = item["nim_name"]
        problem_idx = item["problem_idx"]
        problem_text = item["problem_text"]
        ground_truth = item["ground_truth"]

        loaded_samples: list[dict] = []
        for sample_idx in range(n_samples):
            cache_path = _cache_path(nim_name, problem_text, sample_idx)
            if not cache_path.exists():
                continue
            try:
                raw = json.loads(cache_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            content = raw.get("content", "")
            loaded_samples.append(
                {
                    "sample_idx": sample_idx,
                    "content": content,
                    "extracted": extract_boxed_answer(content),
                }
            )

        for judge_mode, prompt_kind in judge_modes:
            missing_samples = []
            for sample in loaded_samples:
                key = f"{model_key}:{problem_idx}:{sample['sample_idx']}:{judge_mode}"
                if key in done:
                    skipped += 1
                else:
                    missing_samples.append(sample)

            pair_key = (model_key, problem_idx, judge_mode)
            for start in range(0, len(missing_samples), samples_per_call):
                pending_groups.append(
                    {
                        "model": model_key,
                        "problem_idx": problem_idx,
                        "judge_mode": judge_mode,
                        "prompt_kind": prompt_kind,
                        "problem_text": problem_text,
                        "ground_truth": ground_truth,
                        "samples": missing_samples[start : start + samples_per_call],
                        "attempts": 0,
                        "done_in_pair": done_by_pair.get(pair_key, 0),
                    }
                )

    pending_groups.sort(
        key=lambda task: (
            0 if task["done_in_pair"] > 0 else 1,
            -int(task["done_in_pair"]),
            task["model"],
            int(task["problem_idx"]),
            int(task["samples"][0]["sample_idx"]) if task["samples"] else 0,
        )
    )
    for task in pending_groups:
        tasks.put(task)

    total_records = sum(len(group["samples"]) for group in pending_groups)
    total_groups = len(pending_groups)
    if total_records == 0:
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    counter = {"completed": 0, "failed": 0, "requeued": 0, "calls": 0}
    min_delay = max(60.0 / max(float(RATE_LIMIT_PER_KEY), 1.0), request_delay)
    n_workers = min(workers or len(NIM_API_KEYS), len(NIM_API_KEYS), total_groups)

    def is_retryable_error(err_text: str | None) -> bool:
        if not err_text:
            return False
        lowered = err_text.lower()
        return any(
            marker in lowered
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

    def build_prompt(task: dict) -> str:
        if task["prompt_kind"] == "no_reference":
            return format_no_reference_batch_prompt(task["problem_text"], task["samples"])
        return format_reference_batch_prompt(task["problem_text"], task["ground_truth"], task["samples"])

    def worker(key: str) -> None:
        client = OpenAI(
            base_url=NIM_BASE_URL,
            api_key=key,
            timeout=LIVE_JUDGE_TIMEOUT_SECONDS,
            max_retries=0,
        )
        while True:
            try:
                task = tasks.get_nowait()
            except Empty:
                return

            last_error = None
            parsed: dict[int, dict] = {}
            sample_indices = [int(sample["sample_idx"]) for sample in task["samples"]]
            prompt = build_prompt(task)
            for attempt in range(3):
                try:
                    response = client.chat.completions.create(
                        model=judge_model,
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a careful mathematical solution judge. Return only the requested JSON array.",
                            },
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.0,
                        max_tokens=min(4096, max(512, 256 * len(sample_indices) + 128)),
                        timeout=LIVE_JUDGE_TIMEOUT_SECONDS,
                    )
                    text = response.choices[0].message.content or ""
                    parsed = parse_batch_judgments(text, sample_indices)
                    if parsed:
                        break
                    last_error = "parse_failed"
                except Exception as exc:
                    last_error = str(exc)[:160]
                if is_retryable_error(last_error):
                    time.sleep(max(rate_limit_cooldown, min(2 ** attempt * 3, 60)))
                else:
                    time.sleep(min(2 ** attempt * 3, 60))

            parsed_samples = set(parsed)
            missing_samples = [
                sample for sample in task["samples"]
                if int(sample["sample_idx"]) not in parsed_samples
            ]

            with lock:
                if parsed:
                    with open(output_path, "a", encoding="utf-8") as f:
                        for sample in task["samples"]:
                            sample_idx = int(sample["sample_idx"])
                            if sample_idx not in parsed:
                                continue
                            judgment = parsed[sample_idx]
                            judgment["judge_source"] = f"live_nim_batch:{judge_model}"
                            record = {
                                "model": task["model"],
                                "problem_idx": task["problem_idx"],
                                "sample_idx": sample_idx,
                                "judge_mode": task["judge_mode"],
                                **judgment,
                            }
                            f.write(json.dumps(record) + "\n")
                            counter["completed"] += 1
                        f.flush()
                    counter["calls"] += 1
                    if counter["completed"] % progress_batch_size == 0 or counter["completed"] == total_records:
                        print(
                            f"  Live batch judge progress: {counter['completed']}/{total_records} new records, "
                            f"{skipped} skipped, {counter['calls']} calls, "
                            f"{counter['requeued']} requeued groups, {counter['failed']} failed-unwritten"
                        )

                if missing_samples:
                    task["attempts"] = int(task.get("attempts", 0)) + 1
                    if task["attempts"] < max_task_attempts:
                        task["samples"] = missing_samples
                        task["done_in_pair"] = len(parsed_samples)
                        tasks.put(task)
                        counter["requeued"] += 1
                    else:
                        counter["failed"] += len(missing_samples)
                        if counter["failed"] % progress_batch_size == 0:
                            print(f"  Live batch judge failures without checkpoint: {counter['failed']} latest={last_error}")

            tasks.task_done()
            time.sleep(min_delay)

    print(
        f"Running {total_records} live judgments in {total_groups} batched calls "
        f"with {n_workers} key workers ({skipped} skipped, batch_size={samples_per_call})."
    )
    threads = [threading.Thread(target=worker, args=(NIM_API_KEYS[i],), daemon=True) for i in range(n_workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return int(counter["completed"])


def run_live_judgments_parallel(
    problems: list[dict],
    items: list[dict],
    judge_model: str,
    output_path: Path = LIVE_JUDGED_PATH,
    judge_modes: list[tuple[str, str]] | None = None,
    max_samples: int = N_SAMPLES,
    workers: int | None = None,
    batch_size: int = 50,
    request_delay: float = LIVE_JUDGE_REQUEST_DELAY_SECONDS,
    rate_limit_cooldown: float = LIVE_JUDGE_RATE_LIMIT_COOLDOWN_SECONDS,
    max_task_attempts: int = LIVE_JUDGE_MAX_TASK_ATTEMPTS,
) -> int:
    if not NIM_API_KEYS:
        raise RuntimeError("No NIM keys available. Set NIM_API_KEYS or provide C:\\Users\\wangz\\MIRROR\\.env.")
    if judge_modes is None:
        judge_modes = [(LIVE_NO_REFERENCE_MODE, "no_reference")]

    from openai import OpenAI
    from queue import Empty, Queue
    import threading
    from config import RATE_LIMIT_PER_KEY

    done = load_checkpoint(output_path)
    n_samples = min(max_samples, N_SAMPLES)
    tasks: Queue = Queue()
    pending_tasks: list[dict] = []
    skipped = 0
    done_by_pair: dict[tuple[str, int, str], int] = {}
    for key in done:
        try:
            model, pidx, _sample_idx, judge_mode = key.split(":", 3)
            pair_key = (model, int(pidx), judge_mode)
            done_by_pair[pair_key] = done_by_pair.get(pair_key, 0) + 1
        except ValueError:
            continue

    for item in items:
        model_key = item["model_key"]
        nim_name = item["nim_name"]
        problem_idx = item["problem_idx"]
        problem_text = item["problem_text"]
        ground_truth = item["ground_truth"]

        for sample_idx in range(n_samples):
            cache_path = _cache_path(nim_name, problem_text, sample_idx)
            if not cache_path.exists():
                continue
            try:
                raw = json.loads(cache_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue

            content = raw.get("content", "")
            extracted = extract_boxed_answer(content)
            for judge_mode, prompt_kind in judge_modes:
                key = f"{model_key}:{problem_idx}:{sample_idx}:{judge_mode}"
                if key in done:
                    skipped += 1
                    continue
                if prompt_kind == "no_reference":
                    prompt = format_no_reference_prompt(problem_text, content, extracted)
                else:
                    prompt = format_reference_prompt(problem_text, content, extracted, ground_truth)
                pending_tasks.append(
                    {
                        "model": model_key,
                        "problem_idx": problem_idx,
                        "sample_idx": sample_idx,
                        "judge_mode": judge_mode,
                        "prompt": prompt,
                        "attempts": 0,
                        "done_in_pair": done_by_pair.get((model_key, problem_idx, judge_mode), 0),
                    }
                )

    pending_tasks.sort(
        key=lambda task: (
            0 if task["done_in_pair"] > 0 else 1,
            -int(task["done_in_pair"]),
            task["model"],
            int(task["problem_idx"]),
            int(task["sample_idx"]),
        )
    )
    for task in pending_tasks:
        tasks.put(task)

    total_tasks = tasks.qsize()
    if total_tasks == 0:
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    counter = {"completed": 0, "failed": 0, "requeued": 0}
    min_delay = max(60.0 / max(float(RATE_LIMIT_PER_KEY), 1.0), request_delay)
    n_workers = min(workers or len(NIM_API_KEYS), len(NIM_API_KEYS), total_tasks)

    def is_retryable_error(err_text: str | None) -> bool:
        if not err_text:
            return False
        lowered = err_text.lower()
        return any(
            marker in lowered
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

    def worker(key: str) -> None:
        client = OpenAI(
            base_url=NIM_BASE_URL,
            api_key=key,
            timeout=LIVE_JUDGE_TIMEOUT_SECONDS,
            max_retries=0,
        )
        while True:
            try:
                task = tasks.get_nowait()
            except Empty:
                return

            last_error = None
            judgment = None
            for attempt in range(4):
                try:
                    response = client.chat.completions.create(
                        model=judge_model,
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a careful mathematical solution judge. Return only the requested JSON object.",
                            },
                            {"role": "user", "content": task["prompt"]},
                        ],
                        temperature=0.0,
                        max_tokens=512,
                        timeout=LIVE_JUDGE_TIMEOUT_SECONDS,
                    )
                    text = response.choices[0].message.content or ""
                    judgment = parse_judgment(text)
                    if judgment is not None:
                        judgment["judge_source"] = f"live_nim:{judge_model}"
                        break
                    last_error = "parse_failed"
                except Exception as exc:
                    last_error = str(exc)[:160]
                if is_retryable_error(last_error):
                    time.sleep(max(rate_limit_cooldown, min(2 ** attempt * 3, 60)))
                else:
                    time.sleep(min(2 ** attempt * 3, 60))

            if judgment is None:
                with lock:
                    task["attempts"] = int(task.get("attempts", 0)) + 1
                    if is_retryable_error(last_error) and task["attempts"] < max_task_attempts:
                        tasks.put(task)
                        counter["requeued"] += 1
                        if counter["requeued"] % batch_size == 0:
                            print(
                                f"  Live judge retry queue: {counter['requeued']} requeued, "
                                f"latest={last_error}"
                            )
                    else:
                        counter["failed"] += 1
                    if counter["failed"] % batch_size == 0 and counter["failed"]:
                        print(f"  Live judge failures without checkpoint: {counter['failed']} latest={last_error}")
                tasks.task_done()
                time.sleep(min_delay)
                continue

            with lock:
                record = {
                    "model": task["model"],
                    "problem_idx": task["problem_idx"],
                    "sample_idx": task["sample_idx"],
                    "judge_mode": task["judge_mode"],
                    **judgment,
                }
                with open(output_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")
                    f.flush()
                counter["completed"] += 1
                if counter["completed"] % batch_size == 0 or counter["completed"] == total_tasks:
                    print(
                        f"  Live judge progress: {counter['completed']}/{total_tasks} new, "
                        f"{skipped} skipped, {counter['requeued']} requeued, "
                        f"{counter['failed']} failed-unwritten"
                    )

            tasks.task_done()
            time.sleep(min_delay)

    print(f"Running {total_tasks} live judgments with {n_workers} key workers ({skipped} skipped).")
    threads = [threading.Thread(target=worker, args=(NIM_API_KEYS[i],), daemon=True) for i in range(n_workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return int(counter["completed"])


def load_judge_scores(judged_path: Path, judge_mode: str = NO_REFERENCE_MODE) -> dict[str, dict[int, np.ndarray]]:
    scores: dict[str, dict[int, list[tuple[int, float]]]] = {}
    if not judged_path.exists():
        return {}
    with open(judged_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("judge_mode") != judge_mode:
                continue
            model = rec["model"]
            pidx = int(rec["problem_idx"])
            sidx = int(rec["sample_idx"])
            score = float(rec.get("judge_score", 0.5))
            if model not in scores:
                scores[model] = {}
            if pidx not in scores[model]:
                scores[model][pidx] = []
            scores[model][pidx].append((sidx, score))

    result: dict[str, dict[int, np.ndarray]] = {}
    for model, prob_dict in scores.items():
        result[model] = {}
        for pidx, pairs in prob_dict.items():
            pairs.sort(key=lambda x: x[0])
            result[model][pidx] = np.array([s for _, s in pairs], dtype=float)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run structured judge-score diagnostics.")
    parser.add_argument("--models", nargs="*", default=TARGET_MODELS)
    parser.add_argument("--n-problems", type=int, default=N_JUDGE_PROBLEMS)
    parser.add_argument("--live", action="store_true", help="Use a live NIM LLM judge instead of the heuristic fallback.")
    parser.add_argument("--judge-model", default=MODELS.get("70B", "meta/llama-3.3-70b-instruct"))
    parser.add_argument("--include-reference", action="store_true", help="Also run reference-aware live diagnostic judgments.")
    parser.add_argument("--max-samples", type=int, default=N_SAMPLES, help="Limit samples per selected problem for smoke tests.")
    parser.add_argument("--workers", type=int, default=None, help="Max live judge key workers.")
    parser.add_argument("--request-delay", type=float, default=LIVE_JUDGE_REQUEST_DELAY_SECONDS, help="Minimum seconds between live judge calls per worker.")
    parser.add_argument("--rate-limit-cooldown", type=float, default=LIVE_JUDGE_RATE_LIMIT_COOLDOWN_SECONDS, help="Sleep seconds after retryable live judge errors such as 429s.")
    parser.add_argument("--max-task-attempts", type=int, default=LIVE_JUDGE_MAX_TASK_ATTEMPTS, help="Max queue-level attempts before leaving a live judge task unwritten.")
    parser.add_argument("--live-batch-size", type=int, default=1, help="Responses to judge per live API call. Values above 1 use the batched live-judge runner.")
    args = parser.parse_args()

    target_models = args.models
    n_judge_problems = args.n_problems

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading MATH500 problems...")
    problems = load_math500(DATA_DIR)

    print("Selecting judgment subset...")
    items, manifest = build_work_items(problems, target_models, n_judge_problems)
    if args.live:
        judge_modes = [(LIVE_NO_REFERENCE_MODE, "no_reference")]
        if args.include_reference:
            judge_modes.append((LIVE_REFERENCE_MODE, "reference"))
        judged_path = LIVE_JUDGED_PATH
        manifest_path = LIVE_MANIFEST_PATH
        judge_fn = build_live_judge_fn(args.judge_model)
        manifest["judge_kind"] = "live_nim"
        manifest["judge_model"] = args.judge_model
    else:
        judge_modes = [
            (NO_REFERENCE_MODE, "no_reference"),
            (REFERENCE_MODE, "reference"),
        ]
        judged_path = JUDGED_PATH
        manifest_path = MANIFEST_PATH
        judge_fn = None
        manifest["judge_kind"] = "deterministic_fallback_heuristic"

    manifest["judge_modes"] = [mode for mode, _ in judge_modes]
    n_samples = min(args.max_samples, N_SAMPLES)
    manifest["samples_per_problem"] = n_samples
    manifest["total_judgments_with_modes"] = len(items) * n_samples * len(judge_modes)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if args.live:
        manifest_snapshot_path = manifest_path.with_name(
            f"{manifest_path.stem}_n{manifest['n_problems']}_s{n_samples}.json"
        )
        manifest_snapshot_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    RUBRIC_PATH.write_text(JUDGE_RUBRIC, encoding="utf-8")
    print(f"Selected {manifest['total_judgments_with_modes']} total judgments "
          f"({len(target_models)} models x {manifest['n_problems']} problems x {n_samples} samples x {len(judge_modes)} modes)")

    mode_desc = "live NIM judge" if args.live else "heuristic judge"
    print(f"Running judgments ({mode_desc})...")
    if args.live:
        if args.live_batch_size > 1:
            completed = run_live_judgments_batch_parallel(
                problems,
                items,
                judge_model=args.judge_model,
                output_path=judged_path,
                judge_modes=judge_modes,
                max_samples=args.max_samples,
                workers=args.workers,
                samples_per_call=args.live_batch_size,
                request_delay=args.request_delay,
                rate_limit_cooldown=args.rate_limit_cooldown,
                max_task_attempts=args.max_task_attempts,
            )
        else:
            completed = run_live_judgments_parallel(
                problems,
                items,
                judge_model=args.judge_model,
                output_path=judged_path,
                judge_modes=judge_modes,
                max_samples=args.max_samples,
                workers=args.workers,
                request_delay=args.request_delay,
                rate_limit_cooldown=args.rate_limit_cooldown,
                max_task_attempts=args.max_task_attempts,
            )
    else:
        completed = run_judgments(
            problems,
            items,
            judge_fn=judge_fn,
            output_path=judged_path,
            judge_modes=judge_modes,
            max_samples=args.max_samples,
        )
    print(f"Completed {completed} new judgments. Output: {judged_path}")


if __name__ == "__main__":
    main()
