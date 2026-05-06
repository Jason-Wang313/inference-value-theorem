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
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import MODELS, N_SAMPLES, RESULTS_DIR, DATA_DIR
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
MANIFEST_PATH = OUT_DIR / "judge_subset_manifest.json"
RUBRIC_PATH = OUT_DIR / "judge_rubric.md"

TARGET_MODELS = ["3B", "8B", "70B"]
N_JUDGE_PROBLEMS = 50
NO_REFERENCE_MODE = "heuristic_no_reference"
REFERENCE_MODE = "reference_based_diagnostic"


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
) -> int:
    done = load_checkpoint(JUDGED_PATH)
    total = len(items) * N_SAMPLES * 2
    completed = 0
    skipped = 0

    with open(JUDGED_PATH, "a", encoding="utf-8") as f:
        for item_idx, item in enumerate(items):
            model_key = item["model_key"]
            nim_name = item["nim_name"]
            problem_idx = item["problem_idx"]
            problem_text = item["problem_text"]
            ground_truth = item["ground_truth"]

            for sample_idx in range(N_SAMPLES):
                for judge_mode in [NO_REFERENCE_MODE, REFERENCE_MODE]:
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

                    if judge_mode == NO_REFERENCE_MODE:
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
    args = parser.parse_args()

    target_models = args.models
    n_judge_problems = args.n_problems

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading MATH500 problems...")
    problems = load_math500(DATA_DIR)

    print("Selecting judgment subset...")
    items, manifest = build_work_items(problems, target_models, n_judge_problems)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    RUBRIC_PATH.write_text(JUDGE_RUBRIC, encoding="utf-8")
    print(f"Selected {manifest['total_judgments']} total judgments "
          f"({len(target_models)} models x {manifest['n_problems']} problems x {N_SAMPLES} samples x 2 modes)")

    print("Running judgments (using heuristic judge as default)...")
    completed = run_judgments(problems, items, judge_fn=None)
    print(f"Completed {completed} new judgments. Output: {JUDGED_PATH}")


if __name__ == "__main__":
    main()
