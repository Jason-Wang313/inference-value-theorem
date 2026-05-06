"""
Raw-cache feature extraction with parallel I/O.

Uses a thread pool for file reads to overcome per-file I/O overhead on Windows.
Peak RAM: ~500MB (one model's raw files loaded in parallel batches).
"""

from __future__ import annotations

import csv
import json
import math
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.nim_client import _cache_path, compute_score
from src.utils import extract_boxed_answer

FEATURE_COLS = [
    "model_key",
    "problem_idx",
    "sample_idx",
    "mean_logprob",
    "total_logprob",
    "response_length_tokens",
    "response_length_chars",
    "sqrt_length_normalized_logprob",
    "length_penalized_logprob",
    "boxed_answer_present",
    "extracted_answer_present",
    "answer_format_valid",
    "final_answer_length",
    "contains_therefore",
    "equation_count",
    "latex_marker_count",
    "correct",
]

NUMERIC_FEATURE_COLS = [
    "mean_logprob",
    "total_logprob",
    "response_length_tokens",
    "response_length_chars",
    "sqrt_length_normalized_logprob",
    "length_penalized_logprob",
    "boxed_answer_present",
    "extracted_answer_present",
    "answer_format_valid",
    "final_answer_length",
    "contains_therefore",
    "equation_count",
    "latex_marker_count",
    "correct",
]

LEARNABLE_FEATURE_COLS = [
    "mean_logprob",
    "total_logprob",
    "response_length_tokens",
    "response_length_chars",
    "sqrt_length_normalized_logprob",
    "boxed_answer_present",
    "extracted_answer_present",
    "final_answer_length",
    "contains_therefore",
    "equation_count",
    "latex_marker_count",
]

MISSING_MEAN_LOGPROB = -100.0
MISSING_TOTAL_LOGPROB = -1_000_000.0


def _count_equations(text: str) -> int:
    count = 0
    count += text.count("$$") // 2
    count += len(re.findall(r"\\\\?\[", text))
    count += len(re.findall(r"\\begin\{equation", text))
    count += len(re.findall(r"\\begin\{align", text))
    return count


def _count_latex_markers(text: str) -> int:
    count = 0
    for marker in (r"\frac", r"\sqrt", r"\sum", r"\int", r"\prod", r"\lim"):
        count += text.count(marker)
    return count


def _fast_extract_answer(content: str) -> str | None:
    boxed = extract_boxed_answer(content)
    if boxed is not None:
        return boxed
    m = re.search(r"(?:answer|result)\s+is\s*:?\s*(.+?)(?:\.|$)", content, re.IGNORECASE)
    if m:
        return m.group(1).strip()[:200]
    return None


def _process_one_file(args: tuple) -> dict | None:
    cache_path, model_key, p_idx, s_idx, correct_label = args
    try:
        raw_text = cache_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    try:
        raw = json.loads(raw_text)
    except json.JSONDecodeError:
        return None

    content = raw.get("content", "")
    logprobs = raw.get("logprobs") or []

    if logprobs:
        total_lp = sum(t["logprob"] for t in logprobs)
        n_tokens = len(logprobs)
        mean_lp = total_lp / n_tokens
        sqrt_norm = total_lp / math.sqrt(n_tokens)
        length_pen = mean_lp - 0.01 * n_tokens
    else:
        total_lp = MISSING_TOTAL_LOGPROB
        n_tokens = 0
        mean_lp = MISSING_MEAN_LOGPROB
        sqrt_norm = MISSING_TOTAL_LOGPROB
        length_pen = MISSING_MEAN_LOGPROB

    boxed = extract_boxed_answer(content)
    extracted = _fast_extract_answer(content)

    return {
        "model_key": model_key,
        "problem_idx": p_idx,
        "sample_idx": s_idx,
        "mean_logprob": mean_lp,
        "total_logprob": total_lp,
        "response_length_tokens": n_tokens,
        "response_length_chars": len(content),
        "sqrt_length_normalized_logprob": sqrt_norm,
        "length_penalized_logprob": length_pen,
        "boxed_answer_present": int(boxed is not None),
        "extracted_answer_present": int(extracted is not None),
        "answer_format_valid": int(boxed is not None),
        "final_answer_length": len(extracted) if extracted else 0,
        "contains_therefore": int("therefore" in content.lower()),
        "equation_count": _count_equations(content),
        "latex_marker_count": _count_latex_markers(content),
        "correct": correct_label,
    }


def load_math500(data_dir: Path, n_problems: int = 500) -> list[dict]:
    path = data_dir / "math500.jsonl"
    problems = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            problems.append(json.loads(line))
            if len(problems) >= n_problems:
                break
    return problems


def _build_cache_index(raw_dir: Path) -> set[str]:
    """Pre-index all filenames in the raw cache directory (one OS call)."""
    try:
        return set(f.name for f in raw_dir.iterdir() if f.suffix == ".json")
    except OSError:
        return set()


def extract_all_features(
    models: dict[str, str],
    problems: list[dict],
    n_samples: int,
    measurements_dir: Path,
    output_csv_path: Path,
    progress: bool = True,
    max_workers: int = 8,
    append: bool = False,
) -> int:
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    raw_dir = Path(__file__).resolve().parent.parent / "data" / "raw"
    if progress:
        print("  Pre-indexing raw cache directory...", end="", flush=True)
    cache_index = _build_cache_index(raw_dir)
    if progress:
        print(f" {len(cache_index)} files found.")

    model_keys = sorted(models.keys())
    total_models = len(model_keys)
    row_count = 0

    mode = "a" if append and output_csv_path.exists() else "w"
    with open(output_csv_path, mode, newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FEATURE_COLS)
        if mode == "w":
            writer.writeheader()

        for m_idx, model_key in enumerate(model_keys):
            nim_name = models[model_key]
            model_meas_dir = measurements_dir / model_key

            if progress:
                print(f"  [{m_idx + 1}/{total_models}] {model_key} ...", end="", flush=True)

            measurement_cache: dict[int, list] = {}
            for p_idx in range(len(problems)):
                meas_path = model_meas_dir / f"problem_{p_idx}.json"
                try:
                    meas = json.loads(meas_path.read_text(encoding="utf-8"))
                    measurement_cache[p_idx] = meas.get("all_correct", [])
                except (json.JSONDecodeError, OSError):
                    pass

            work_items = []
            for p_idx, prob in enumerate(problems):
                problem_text = prob["problem"]
                all_correct = measurement_cache.get(p_idx, [])

                for s_idx in range(n_samples):
                    cache_path = _cache_path(nim_name, problem_text, s_idx)
                    if cache_path.name not in cache_index:
                        continue
                    correct_label = int(bool(all_correct[s_idx])) if s_idx < len(all_correct) else 0
                    work_items.append((cache_path, model_key, p_idx, s_idx, correct_label))

            if progress:
                print(f" {len(work_items)} files to process...", end="", flush=True)

            model_rows = 0
            batch_size = 500
            for batch_start in range(0, len(work_items), batch_size):
                batch = work_items[batch_start : batch_start + batch_size]
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    results = list(executor.map(_process_one_file, batch))
                for result in results:
                    if result is not None:
                        writer.writerow(result)
                        model_rows += 1

            row_count += model_rows
            if progress:
                print(f" {model_rows} rows")

            del measurement_cache
            del work_items
            csvfile.flush()

    return row_count
