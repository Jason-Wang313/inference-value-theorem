"""
Structured judge-score prompt utilities.

Selects a stratified subset of responses and provides utilities for
structured judgment. The actual judging is done by the experiment runner
(11_model_judge.py), either through a supplied model judge or through the
runner's deterministic heuristic diagnostic.
"""

from __future__ import annotations

import json
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.nim_client import _cache_path
from src.utils import extract_boxed_answer


def select_judgment_subset(
    models: dict[str, str],
    problems: list[dict],
    measurements_dir: Path,
    target_models: list[str],
    n_problems: int = 50,
    seed: int = 42,
) -> tuple[list[dict], dict]:
    rng = np.random.default_rng(seed)
    test_indices = [i for i in range(len(problems)) if i % 5 == 0]

    bins: dict[str, list[int]] = {"hard": [], "medium": [], "easy": []}
    for model_key in target_models:
        model_dir = measurements_dir / model_key
        for pidx in test_indices:
            meas_path = model_dir / f"problem_{pidx}.json"
            if not meas_path.exists():
                continue
            try:
                meas = json.loads(meas_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            p = meas.get("p", 0.5)
            if p < 0.25:
                if pidx not in bins["hard"]:
                    bins["hard"].append(pidx)
            elif p < 0.75:
                if pidx not in bins["medium"]:
                    bins["medium"].append(pidx)
            else:
                if pidx not in bins["easy"]:
                    bins["easy"].append(pidx)

    per_bin = n_problems // 3
    remainder = n_problems - 3 * per_bin
    selected = []
    for i, (bin_name, indices) in enumerate(bins.items()):
        n = per_bin + (1 if i < remainder else 0)
        n = min(n, len(indices))
        chosen = rng.choice(indices, size=n, replace=False).tolist()
        selected.extend(chosen)

    selected = sorted(set(selected))[:n_problems]

    items = []
    for model_key in target_models:
        nim_name = models[model_key]
        for pidx in selected:
            prob = problems[pidx]
            items.append({
                "model_key": model_key,
                "nim_name": nim_name,
                "problem_idx": pidx,
                "problem_text": prob["problem"],
                "ground_truth": prob["answer"],
                "level": prob.get("level", 0),
            })

    manifest = {
        "target_models": target_models,
        "n_problems": len(selected),
        "problem_indices": selected,
        "difficulty_bins": {k: len(v) for k, v in bins.items()},
        "selected_per_bin": {
            "hard": len([p for p in selected if p in bins["hard"]]),
            "medium": len([p for p in selected if p in bins["medium"]]),
            "easy": len([p for p in selected if p in bins["easy"]]),
        },
        "total_judgments": len(items) * 48,
    }

    return items, manifest


def load_checkpoint(output_path: Path) -> set[str]:
    done = set()
    if not output_path.exists():
        return done
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                key = f"{rec['model']}:{rec['problem_idx']}:{rec['sample_idx']}:{rec['judge_mode']}"
                done.add(key)
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def format_no_reference_prompt(problem_text: str, response_text: str, extracted_answer: str | None) -> str:
    answer_line = f"Extracted final answer: {extracted_answer}" if extracted_answer else "No clearly extracted final answer found."
    return f"""You are a math solution quality judge. Evaluate the following candidate response to a math problem.
Do NOT assume you know the correct answer. Judge based on the mathematical reasoning quality.

## Math Problem
{problem_text}

## Candidate Response
{response_text}

## {answer_line}

## Instructions
Score this response on a scale of 0.0 to 1.0:
- 1.0: Solution appears mathematically correct with clear, valid reasoning leading to a supported final answer.
- 0.7-0.9: Mostly correct reasoning with minor issues; final answer likely correct.
- 0.4-0.6: Partially correct reasoning but uncertain; answer may or may not be right.
- 0.1-0.3: Significant errors in reasoning; answer likely wrong.
- 0.0: Clearly incorrect, incoherent, or no usable final answer.

Penalize: unsupported final answers, contradictions between reasoning and answer, missing steps.

Respond with ONLY a JSON object (no markdown fences):
{{"score": <float 0.0-1.0>, "label": "<likely_correct|uncertain|likely_incorrect>", "reason": "<max 25 words>", "format_valid": <true|false>, "solution_validity": <float 0.0-1.0>, "answer_confidence": <float 0.0-1.0>}}"""


def format_reference_prompt(problem_text: str, response_text: str, extracted_answer: str | None, ground_truth: str) -> str:
    answer_line = f"Extracted final answer: {extracted_answer}" if extracted_answer else "No clearly extracted final answer found."
    return f"""You are a math solution quality judge. Evaluate the following candidate response to a math problem.
You are given the correct answer for reference.

## Math Problem
{problem_text}

## Correct Answer
{ground_truth}

## Candidate Response
{response_text}

## {answer_line}

## Instructions
Score this response on a scale of 0.0 to 1.0:
- 1.0: Final answer matches the correct answer and reasoning is valid.
- 0.7-0.9: Final answer is correct (possibly in different form) with mostly sound reasoning.
- 0.4-0.6: Answer is close but not equivalent, or correct answer with flawed reasoning.
- 0.1-0.3: Answer is wrong but reasoning shows partial understanding.
- 0.0: Completely wrong answer with no valid reasoning.

Respond with ONLY a JSON object (no markdown fences):
{{"score": <float 0.0-1.0>, "label": "<likely_correct|uncertain|likely_incorrect>", "reason": "<max 25 words>", "format_valid": <true|false>, "solution_validity": <float 0.0-1.0>, "answer_confidence": <float 0.0-1.0>}}"""


def parse_judgment(text: str) -> dict | None:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        obj = json.loads(text)
        if "score" in obj:
            return {
                "judge_score": float(obj["score"]),
                "judge_label": str(obj.get("label", "uncertain")),
                "reason_short": str(obj.get("reason", ""))[:100],
                "format_valid": bool(obj.get("format_valid", True)),
                "solution_validity": float(obj.get("solution_validity", obj["score"])),
                "final_answer_confidence": float(obj.get("answer_confidence", obj["score"])),
            }
    except (json.JSONDecodeError, ValueError, KeyError):
        pass

    import re
    m = re.search(r'"score"\s*:\s*([\d.]+)', text)
    if m:
        score = float(m.group(1))
        score = max(0.0, min(1.0, score))
        label_m = re.search(r'"label"\s*:\s*"([^"]+)"', text)
        label = label_m.group(1) if label_m else "uncertain"
        reason_m = re.search(r'"reason"\s*:\s*"([^"]*)"', text)
        reason = reason_m.group(1)[:100] if reason_m else ""
        return {
            "judge_score": score,
            "judge_label": label,
            "reason_short": reason,
            "format_valid": True,
            "solution_validity": score,
            "final_answer_confidence": score,
        }

    return None


JUDGE_RUBRIC = """# Structured Judge-Score Rubric

## Purpose
Judge math problem responses on quality and correctness without seeing the ground-truth answer (no-reference mode) or with reference (reference-based mode).

## No-Reference Scoring (0.0 to 1.0)
- **1.0**: Solution appears mathematically correct; final answer clearly supported by valid reasoning.
- **0.7-0.9**: Mostly correct reasoning with minor issues; final answer likely correct.
- **0.4-0.6**: Plausible but uncertain or incomplete reasoning.
- **0.1-0.3**: Significant mathematical errors; answer likely wrong.
- **0.0**: Clearly incorrect, incoherent, or no usable final answer.

## Penalties
- Unsupported final answers (answer appears without derivation)
- Contradictions between reasoning steps and final answer
- Missing critical steps in the derivation
- Format issues (no boxed answer, unclear final answer)

## Reference-Based Scoring (0.0 to 1.0)
Same scale but with knowledge of the correct answer:
- 1.0: Correct answer + valid reasoning
- 0.0: Wrong answer + invalid reasoning

Note: the checked-in judge-score artifact was generated by the deterministic fallback heuristic unless a live judge function is supplied.
"""
