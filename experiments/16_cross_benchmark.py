"""
Cross-benchmark theorem runner for LiveCodeBench, GPQA Diamond, IFEval, and
selected LiveBench subsets.

The runner normalizes each benchmark into measurement records with:
  all_scores: mean logprob score for each response
  all_correct: benchmark-specific correctness labels

Then it applies the same exact finite best-of-N law used by the MATH pipeline.
"""

from __future__ import annotations

import argparse
import ast
import base64
import csv
import hashlib
import importlib
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from typing import Any

import numpy as np
import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from openai import OpenAI
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import MODELS, NIM_API_KEYS, NIM_BASE_URL, RESULTS_DIR  # noqa: E402
from src.nim_client import compute_score  # noqa: E402
from src.theorem import compute_f_theoretical  # noqa: E402
from src.utils import check_correctness  # noqa: E402


OUT_DIR = RESULTS_DIR / "benchmarks"
DEFAULT_SMOKE_MODELS = ["3B", "70B"]
PILOT_MODELS = ["3B", "70B", "Super49B", "Super120B", "MistralSmall119B"]
DEFAULT_N_VALUES = [1, 2, 4, 8, 16, 32, 48]
GOOGLE_RESEARCH_BASE = (
    "https://raw.githubusercontent.com/google-research/google-research/"
    "master/instruction_following_eval/"
)
IFEVAL_FILES = ["instructions.py", "instructions_registry.py", "instructions_util.py"]


@dataclass
class BenchmarkTask:
    benchmark: str
    task_id: str
    prompt: str
    grader_type: str
    answer: Any
    metadata: dict[str, Any]


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def safe_name(text: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")
    return clean[:96] or stable_hash(str(text))


def raw_path(task: BenchmarkTask, model_key: str, sample_idx: int) -> Path:
    return (
        OUT_DIR
        / task.benchmark
        / "raw"
        / model_key
        / safe_name(task.task_id)
        / f"sample_{sample_idx}.json"
    )


def measurement_path(benchmark: str, model_key: str, task_id: str) -> Path:
    return OUT_DIR / benchmark / "measurements" / model_key / f"{safe_name(task_id)}.json"


def build_messages(task: BenchmarkTask) -> list[dict[str, str]]:
    if task.benchmark == "livecodebench":
        system = (
            "You are an expert competitive programmer. Return only the final Python 3 program. "
            "Use stdin/stdout unless the prompt explicitly provides starter code."
        )
    elif task.benchmark in {"mbpp", "humaneval_mbpp"}:
        system = (
            "You are an expert Python programmer. Return only the final Python code. "
            "Define the requested function and avoid prose outside the code."
        )
    elif task.benchmark == "math500":
        system = (
            "Solve the math problem. Show your work step by step. Put your final answer in \\boxed{}."
        )
    elif task.benchmark == "gpqa_diamond":
        system = (
            "Answer the multiple-choice science question. Think briefly, then end with "
            "'Final answer: X' where X is A, B, C, or D."
        )
    elif task.grader_type == "ifeval":
        system = "Follow the user's instruction exactly. Do not add commentary about the constraints."
    else:
        system = (
            "Answer the task. Put the final answer at the end using the format "
            "'Final answer: <answer>'."
        )
    return [{"role": "system", "content": system}, {"role": "user", "content": task.prompt}]


def extract_logprobs(choice) -> list[dict[str, float]] | None:
    lp = getattr(choice, "logprobs", None)
    if lp is None or getattr(lp, "content", None) is None:
        return None
    return [{"token": t.token, "logprob": t.logprob} for t in lp.content]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_math500_benchmark(limit: int | None) -> list[BenchmarkTask]:
    path = PROJECT_ROOT / "data" / "math500.jsonl"
    tasks = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "problem" not in row or "answer" not in row:
                continue
            tasks.append(
                BenchmarkTask(
                    benchmark="math500",
                    task_id=str(row.get("unique_id", row.get("id", i))),
                    prompt=str(row["problem"]),
                    grader_type="math_boxed",
                    answer=str(row["answer"]),
                    metadata={
                        "source": "local:data/math500.jsonl",
                        "level": row.get("level"),
                        "type": row.get("type"),
                    },
                )
            )
            if limit and len(tasks) >= limit:
                break
    return tasks


def load_livecodebench(limit: int | None) -> list[BenchmarkTask]:
    path = Path(
        hf_hub_download(
            repo_id="livecodebench/code_generation",
            repo_type="dataset",
            filename="test.jsonl",
        )
    )
    tasks = []
    for row in load_jsonl(path):
        prompt = (
            "Solve this programming problem in Python 3.\n\n"
            f"{row['question_content']}\n\n"
            "Return only the complete program, preferably inside one ```python code block."
        )
        if row.get("starter_code"):
            prompt += f"\n\nStarter code:\n```python\n{row['starter_code']}\n```"
        tasks.append(
            BenchmarkTask(
                benchmark="livecodebench",
                task_id=str(row["question_id"]),
                prompt=prompt,
                grader_type="python_public_tests",
                answer={
                    "public_test_cases": row.get("public_test_cases", "[]"),
                    "private_test_cases": row.get("private_test_cases", ""),
                    "starter_code": row.get("starter_code", ""),
                },
                metadata={
                    "source": "livecodebench/code_generation:test.jsonl",
                    "title": row.get("question_title"),
                    "platform": row.get("platform"),
                    "difficulty": row.get("difficulty"),
                },
            )
        )
    tasks.sort(key=lambda t: (str(t.metadata.get("difficulty") or ""), t.task_id))
    return tasks[:limit] if limit else tasks


def load_mbpp(limit: int | None, benchmark_name: str = "mbpp") -> list[BenchmarkTask]:
    last_error = None
    dataset = None
    source = None
    for spec in [
        ("google-research-datasets/mbpp", "sanitized", "test"),
        ("google-research-datasets/mbpp", None, "test"),
        ("mbpp", "sanitized", "test"),
        ("mbpp", None, "test"),
    ]:
        repo_id, config_name, split = spec
        try:
            if config_name:
                loaded = load_dataset(repo_id, config_name)
            else:
                loaded = load_dataset(repo_id)
            dataset = loaded[split] if split in loaded else loaded[list(loaded.keys())[0]]
            source = f"{repo_id}:{config_name or 'default'}:{split}"
            break
        except Exception as exc:
            last_error = exc
            continue
    if dataset is None:
        raise RuntimeError(f"Could not load MBPP dataset; last error={last_error}")

    tasks = []
    for i, row in enumerate(dataset):
        tests = row.get("test_list") or row.get("tests") or []
        if isinstance(tests, str):
            try:
                tests = json.loads(tests)
            except json.JSONDecodeError:
                tests = [line.strip() for line in tests.splitlines() if line.strip()]
        prompt = (
            "Write a Python function that satisfies the following task.\n\n"
            f"{row.get('text', row.get('prompt', ''))}\n\n"
            "Return only the Python code. The code will be checked against unit tests."
        )
        tasks.append(
            BenchmarkTask(
                benchmark=benchmark_name,
                task_id=str(row.get("task_id", row.get("id", i))),
                prompt=prompt,
                grader_type="mbpp_tests",
                answer={
                    "tests": tests,
                    "test_setup_code": row.get("test_setup_code", ""),
                    "canonical_code": row.get("code", ""),
                },
                metadata={"source": source},
            )
        )
        if limit and len(tasks) >= limit:
            break
    return tasks


def load_gpqa_diamond(limit: int | None) -> list[BenchmarkTask]:
    try:
        dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond")["train"]
        source = "Idavidrein/gpqa:gpqa_diamond"
    except Exception:
        dataset = load_dataset("dongboklee/GPQA-diamond")["train"]
        source = "dongboklee/GPQA-diamond"
    tasks = []
    for i, row in enumerate(dataset):
        answer = str(row.get("answer") or row.get("Correct Answer") or "").strip().upper()
        question = str(row.get("question") or row.get("Question") or "")
        prompt = (
            f"{question}\n\n"
            "Choose exactly one option. End with 'Final answer: X' where X is A, B, C, or D."
        )
        tasks.append(
            BenchmarkTask(
                benchmark="gpqa_diamond",
                task_id=str(row.get("q_id", row.get("Record ID", i))),
                prompt=prompt,
                grader_type="multiple_choice_letter",
                answer=answer,
                metadata={"source": source},
            )
        )
    return tasks[:limit] if limit else tasks


def load_ifeval(limit: int | None) -> list[BenchmarkTask]:
    dataset = load_dataset("google/IFEval")["train"]
    tasks = []
    for row in dataset:
        tasks.append(
            BenchmarkTask(
                benchmark="ifeval",
                task_id=str(row["key"]),
                prompt=str(row["prompt"]),
                grader_type="ifeval",
                answer={
                    "instruction_id_list": row["instruction_id_list"],
                    "kwargs": row["kwargs"],
                },
                metadata={"source": "google/IFEval"},
            )
        )
    return tasks[:limit] if limit else tasks


def load_livebench_selected(limit: int | None) -> list[BenchmarkTask]:
    per_subset_limit = None
    if limit:
        per_subset_limit = max(1, int(np.ceil(limit / 3)))
    tasks: list[BenchmarkTask] = []
    for dataset_name, grader_type in [
        ("livebench/reasoning", "normalized_exact"),
        ("livebench/data_analysis", "json_or_exact"),
        ("livebench/instruction_following", "ifeval"),
    ]:
        dataset = load_dataset(dataset_name)["test"]
        count = 0
        for row in dataset:
            if per_subset_limit and count >= per_subset_limit:
                break
            turns = row.get("turns") or []
            prompt = "\n\n".join(str(t) for t in turns)
            if grader_type == "ifeval":
                answer = {
                    "instruction_id_list": row["instruction_id_list"],
                    "kwargs": row["kwargs"],
                }
            else:
                answer = row.get("ground_truth", "")
                prompt += "\n\nEnd your response with 'Final answer: <answer>'."
            tasks.append(
                BenchmarkTask(
                    benchmark="livebench_selected",
                    task_id=f"{row.get('task', 'task')}:{row.get('question_id')}",
                    prompt=prompt,
                    grader_type=grader_type,
                    answer=answer,
                    metadata={
                        "source": dataset_name,
                        "category": row.get("category"),
                        "task": row.get("task"),
                    },
                )
            )
            count += 1
    return tasks[:limit] if limit else tasks


def load_tasks(benchmark: str, limit: int | None) -> list[BenchmarkTask]:
    if benchmark == "math500":
        return load_math500_benchmark(limit)
    if benchmark == "livecodebench":
        return load_livecodebench(limit)
    if benchmark == "mbpp":
        return load_mbpp(limit, benchmark_name="mbpp")
    if benchmark == "humaneval_mbpp":
        return load_mbpp(limit, benchmark_name="humaneval_mbpp")
    if benchmark == "gpqa_diamond":
        return load_gpqa_diamond(limit)
    if benchmark == "ifeval":
        return load_ifeval(limit)
    if benchmark == "livebench_selected":
        return load_livebench_selected(limit)
    raise ValueError(f"Unknown benchmark: {benchmark}")


def extract_code(text: str) -> str:
    matches = re.findall(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    return text.strip()


def parse_test_cases(raw: str) -> list[dict[str, Any]]:
    if not raw:
        return []
    try:
        obj = json.loads(raw)
        if isinstance(obj, list):
            return obj
    except json.JSONDecodeError:
        pass
    try:
        decoded = zlib.decompress(base64.b64decode(raw)).decode("utf-8")
        obj = json.loads(decoded)
        if isinstance(obj, list):
            return obj
    except Exception:
        return []
    return []


def parse_literal_lines(text: str) -> list[Any]:
    values = []
    for line in str(text).splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            values.append(ast.literal_eval(line))
        except Exception:
            try:
                values.append(json.loads(line))
            except Exception:
                values.append(line)
    return values


def normalized_value(value: Any) -> Any:
    if isinstance(value, str):
        vals = parse_literal_lines(value)
        if len(vals) == 1:
            return vals[0]
        if len(vals) > 1:
            return vals
        return normalize_text(value)
    return value


def values_equal(got: Any, expected: Any) -> bool:
    got_norm = normalized_value(got)
    expected_norm = normalized_value(expected)
    if got_norm == expected_norm:
        return True
    return normalize_text(str(got_norm)) == normalize_text(str(expected_norm))


def method_name_from_starter(starter_code: str, code: str) -> str | None:
    for source in [starter_code, code]:
        match = re.search(r"def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", source or "")
        if match:
            return match.group(1)
    return None


def grade_functional_tests(code: str, tests: list[dict[str, Any]], starter_code: str) -> bool | None:
    method_name = method_name_from_starter(starter_code, code)
    if not method_name or "class Solution" not in code:
        return False
    harness = f"""
import ast
import json
import sys
from typing import *

{code}

TESTS = {json.dumps(tests)}
METHOD = {json.dumps(method_name)}

def parse_args(raw):
    values = []
    for line in str(raw).splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            values.append(ast.literal_eval(line))
        except Exception:
            try:
                values.append(json.loads(line))
            except Exception:
                values.append(line)
    return values

solution = Solution()
outputs = []
for test in TESTS:
    args = parse_args(test.get("input", ""))
    result = getattr(solution, METHOD)(*args)
    outputs.append(result)
print(json.dumps(outputs, ensure_ascii=False, sort_keys=True))
"""
    with tempfile.TemporaryDirectory() as td:
        script = Path(td) / "functional_runner.py"
        script.write_text(harness, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, str(script)],
                text=True,
                capture_output=True,
                timeout=max(3, min(15, len(tests) // 20 + 3)),
            )
        except subprocess.TimeoutExpired:
            return False
        if proc.returncode != 0:
            return False
        try:
            outputs = json.loads(proc.stdout)
        except json.JSONDecodeError:
            return False
        if len(outputs) != len(tests):
            return False
        return all(values_equal(got, test.get("output", "")) for got, test in zip(outputs, tests))


def grade_python_tests(
    response: str,
    answer: dict[str, Any],
    allow_exec: bool,
    code_test_scope: str,
) -> bool | None:
    if not allow_exec:
        return None
    tests = []
    if code_test_scope in {"public", "all"}:
        tests.extend(parse_test_cases(str(answer.get("public_test_cases", ""))))
    if code_test_scope in {"private", "all"}:
        tests.extend(parse_test_cases(str(answer.get("private_test_cases", ""))))
    stdin_tests = [t for t in tests if t.get("testtype") == "stdin"]
    functional_tests = [t for t in tests if t.get("testtype") == "functional"]
    code = extract_code(response)
    if functional_tests and not stdin_tests:
        return grade_functional_tests(code, functional_tests, str(answer.get("starter_code", "")))
    if not stdin_tests:
        return None
    with tempfile.TemporaryDirectory() as td:
        script = Path(td) / "solution.py"
        script.write_text(code, encoding="utf-8")
        for test in stdin_tests:
            try:
                proc = subprocess.run(
                    [sys.executable, str(script)],
                    input=str(test.get("input", "")),
                    text=True,
                    capture_output=True,
                    timeout=3,
                )
            except subprocess.TimeoutExpired:
                return False
            if proc.returncode != 0:
                return False
            got = proc.stdout.strip().replace("\r\n", "\n")
            expected = str(test.get("output", "")).strip().replace("\r\n", "\n")
            if got != expected:
                return False
    if functional_tests:
        return grade_functional_tests(code, functional_tests, str(answer.get("starter_code", "")))
    return True


def grade_mbpp_tests(response: str, answer: dict[str, Any], allow_exec: bool) -> bool | None:
    if not allow_exec:
        return None
    tests = answer.get("tests") or []
    if isinstance(tests, str):
        tests = [line.strip() for line in tests.splitlines() if line.strip()]
    tests = [str(test).strip() for test in tests if str(test).strip()]
    if not tests:
        return None
    code = extract_code(response)
    setup = str(answer.get("test_setup_code", "") or "")
    harness = "\n\n".join([setup, code, *tests])
    with tempfile.TemporaryDirectory() as td:
        script = Path(td) / "mbpp_runner.py"
        script.write_text(harness, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, str(script)],
                text=True,
                capture_output=True,
                timeout=max(3, min(20, len(tests) + 3)),
            )
        except subprocess.TimeoutExpired:
            return False
        return proc.returncode == 0


def grade_math_boxed(response: str, answer: str) -> bool:
    correct, _ = check_correctness(response, answer)
    return bool(correct)


def extract_answer_text(response: str) -> str:
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", response)
    if boxed:
        return boxed[-1].strip()
    final = re.findall(r"final answer\s*:\s*(.+)", response, flags=re.IGNORECASE)
    if final:
        return final[-1].strip()
    answer = re.findall(r"answer(?: is)?\s*[:\-]?\s*(.+)", response, flags=re.IGNORECASE)
    if answer:
        return answer[-1].strip()
    return response.strip().splitlines()[-1].strip() if response.strip() else ""


def grade_multiple_choice(response: str, answer: str) -> bool:
    text = extract_answer_text(response).upper()
    match = re.search(r"\b([A-D])\b", text)
    if not match:
        letters = re.findall(r"\b([A-D])\b", response.upper())
        pred = letters[-1] if letters else ""
    else:
        pred = match.group(1)
    return pred == str(answer).strip().upper()


def normalize_text(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"^final answer\s*:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    return text.strip().strip("`'\". ").lower()


def grade_normalized_exact(response: str, answer: str) -> bool:
    return normalize_text(extract_answer_text(response)) == normalize_text(answer)


def try_json(text: str) -> Any:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except Exception:
        return None


def grade_json_or_exact(response: str, answer: str) -> bool:
    pred_text = extract_answer_text(response)
    pred_json = try_json(pred_text)
    answer_json = try_json(str(answer))
    if pred_json is not None and answer_json is not None:
        return pred_json == answer_json
    return normalize_text(pred_text) == normalize_text(str(answer))


def ensure_ifeval_official() -> Any | None:
    package_dir = OUT_DIR / "_external" / "instruction_following_eval"
    package_dir.mkdir(parents=True, exist_ok=True)
    init_path = package_dir / "__init__.py"
    init_path.write_text("", encoding="utf-8")
    for filename in IFEVAL_FILES:
        path = package_dir / filename
        if not path.exists():
            url = GOOGLE_RESEARCH_BASE + filename
            path.write_text(urllib.request.urlopen(url, timeout=30).read().decode("utf-8"), encoding="utf-8")
    parent = str(package_dir.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    try:
        import nltk

        try:
            nltk.data.find("tokenizers/punkt/english.pickle")
        except LookupError:
            nltk.download("punkt", quiet=True)
        try:
            nltk.data.find("tokenizers/punkt_tab/english/")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)
        return importlib.import_module("instruction_following_eval.instructions_registry")
    except Exception as exc:
        print(f"WARNING: official IFEval checker unavailable: {exc}")
        return None


_IFEVAL_REGISTRY = None


def grade_ifeval(response: str, answer: dict[str, Any]) -> bool | None:
    global _IFEVAL_REGISTRY
    if _IFEVAL_REGISTRY is None:
        _IFEVAL_REGISTRY = ensure_ifeval_official()
    if _IFEVAL_REGISTRY is None:
        return None
    instruction_ids = answer["instruction_id_list"]
    kwargs_list = answer["kwargs"]
    results = []
    for instruction_id, kwargs in zip(instruction_ids, kwargs_list):
        instruction_cls = _IFEVAL_REGISTRY.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)
        kwargs = {k: v for k, v in dict(kwargs).items() if v is not None}
        try:
            instruction.build_description(**kwargs)
            args = instruction.get_instruction_args()
            if args and "prompt" in args:
                instruction.build_description(prompt=answer.get("prompt", ""), **kwargs)
            results.append(bool(instruction.check_following(response)))
        except Exception:
            return None
    return all(results)


def grade_response(
    task: BenchmarkTask,
    response: str,
    allow_code_exec: bool,
    code_test_scope: str,
) -> bool | None:
    if task.grader_type == "python_public_tests":
        return grade_python_tests(response, task.answer, allow_code_exec, code_test_scope)
    if task.grader_type == "mbpp_tests":
        return grade_mbpp_tests(response, task.answer, allow_exec=allow_code_exec)
    if task.grader_type == "math_boxed":
        return grade_math_boxed(response, str(task.answer))
    if task.grader_type == "multiple_choice_letter":
        return grade_multiple_choice(response, str(task.answer))
    if task.grader_type == "ifeval":
        answer = dict(task.answer)
        answer["prompt"] = task.prompt
        return grade_ifeval(response, answer)
    if task.grader_type == "normalized_exact":
        return grade_normalized_exact(response, str(task.answer))
    if task.grader_type == "json_or_exact":
        return grade_json_or_exact(response, str(task.answer))
    raise ValueError(f"Unknown grader: {task.grader_type}")


def call_batch(
    client: OpenAI,
    model_name: str,
    task: BenchmarkTask,
    sample_indices: list[int],
    max_retries: int,
) -> list[dict[str, Any]]:
    last_error = ""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=build_messages(task),
                temperature=0.7,
                max_tokens=2048 if task.benchmark != "livecodebench" else 4096,
                n=len(sample_indices),
                logprobs=True,
                top_logprobs=5,
            )
            if len(response.choices) != len(sample_indices):
                if len(sample_indices) == 1:
                    raise RuntimeError(
                        f"model returned {len(response.choices)} choices for one requested sample"
                    )
                rows = []
                for sample_idx in sample_indices:
                    rows.extend(call_batch(client, model_name, task, [sample_idx], max_retries=max_retries))
                return rows
            rows = []
            for sample_idx, choice in zip(sample_indices, response.choices):
                rows.append(
                    {
                        "benchmark": task.benchmark,
                        "task_id": task.task_id,
                        "sample_idx": sample_idx,
                        "model_name": model_name,
                        "content": choice.message.content or "",
                        "logprobs": extract_logprobs(choice),
                    }
                )
            return rows
        except Exception as exc:
            err = str(exc)
            last_error = f"{type(exc).__name__}: {err[:500]}"
            if "401" in err or "403" in err:
                raise KeyError(err[:200])
            wait = min(90, (2**attempt) * (8 if "429" in err else 3))
            time.sleep(wait)
    raise RuntimeError(f"batch call failed after retries; last_error={last_error}")


def collect_samples(
    benchmark: str,
    tasks: list[BenchmarkTask],
    models: list[str],
    n_samples: int,
    batch_size: int,
    workers: int,
    max_retries: int,
    max_job_attempts: int,
) -> dict[str, int]:
    jobs: Queue[tuple[str, BenchmarkTask, list[int], int]] = Queue()
    for task in tasks:
        for model_key in models:
            missing = [i for i in range(n_samples) if not raw_path(task, model_key, i).exists()]
            for start in range(0, len(missing), batch_size):
                jobs.put((model_key, task, missing[start : start + batch_size], 1))
    total = jobs.qsize()
    if total == 0:
        return {"jobs": 0, "completed": 0, "cached": len(models) * len(tasks) * n_samples, "failed": 0}
    keys = list(NIM_API_KEYS)
    if not keys:
        raise RuntimeError("Set NIM_API_KEYS or configure MIRROR .env before collecting benchmark samples.")
    counters = {"completed": 0, "failed": 0, "requeued": 0}
    failures: list[dict[str, Any]] = []
    lock = threading.Lock()

    def worker(client: OpenAI) -> None:
        while True:
            try:
                model_key, task, sample_indices, attempt = jobs.get(timeout=3)
            except Empty:
                return
            try:
                rows = call_batch(client, MODELS[model_key], task, sample_indices, max_retries=max_retries)
                returned = {int(row["sample_idx"]) for row in rows}
                missing_returned = [idx for idx in sample_indices if idx not in returned]
                if missing_returned:
                    raise RuntimeError(
                        f"model returned incomplete batch; missing sample indices {missing_returned}"
                    )
                for row in rows:
                    path = raw_path(task, model_key, int(row["sample_idx"]))
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
                with lock:
                    counters["completed"] += 1
                    done = counters["completed"] + counters["failed"]
                    if done % 10 == 0 or done == total:
                        print(
                            f"  {benchmark}: done={done}/{total} "
                            f"completed={counters['completed']} failed={counters['failed']} "
                            f"requeued={counters['requeued']}",
                            flush=True,
                        )
            except KeyError:
                jobs.put((model_key, task, sample_indices, attempt))
                return
            except Exception as exc:
                with lock:
                    if attempt < max_job_attempts:
                        jobs.put((model_key, task, sample_indices, attempt + 1))
                        counters["requeued"] += 1
                    else:
                        counters["failed"] += 1
                        failures.append(
                            {
                                "benchmark": benchmark,
                                "model_key": model_key,
                                "model_name": MODELS.get(model_key),
                                "task_id": task.task_id,
                                "sample_indices": sample_indices,
                                "attempt": attempt,
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                            }
                        )
            finally:
                jobs.task_done()

    worker_count = max(1, workers)
    clients = [
        OpenAI(base_url=NIM_BASE_URL, api_key=keys[i % len(keys)], timeout=180.0)
        for i in range(worker_count)
    ]
    threads = [threading.Thread(target=worker, args=(client,), daemon=True) for client in clients]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failures:
        failure_path = OUT_DIR / benchmark / "collect_failures.jsonl"
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        with failure_path.open("a", encoding="utf-8") as f:
            for failure in failures:
                f.write(json.dumps(failure, ensure_ascii=False) + "\n")
        print(f"  wrote {len(failures)} collection failures to {failure_path}", flush=True)
    return {"jobs": total, **counters, "cached": 0}


def _measure_one_record(args: tuple[str, BenchmarkTask, str, int, bool, str]) -> tuple[str, dict[str, Any] | None, int]:
    benchmark, task, model_key, n_samples, allow_code_exec, code_test_scope = args
    responses = []
    missing = 0
    for sample_idx in range(n_samples):
        path = raw_path(task, model_key, sample_idx)
        if not path.exists():
            missing += 1
            continue
        try:
            responses.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            missing += 1
    if len(responses) != n_samples:
        return "missing", None, missing
    all_scores = []
    all_correct = []
    for response in sorted(responses, key=lambda r: int(r["sample_idx"])):
        label = grade_response(
            task,
            response.get("content", ""),
            allow_code_exec=allow_code_exec,
            code_test_scope=code_test_scope,
        )
        if label is None:
            return "ungraded", None, 0
        all_scores.append(compute_score(response.get("logprobs")))
        all_correct.append(bool(label))
    scores_correct = [s for s, c in zip(all_scores, all_correct) if c]
    scores_incorrect = [s for s, c in zip(all_scores, all_correct) if not c]
    kappa = None
    if scores_correct and scores_incorrect:
        u_stat, _ = stats.mannwhitneyu(scores_correct, scores_incorrect, alternative="greater")
        kappa = float(u_stat / (len(scores_correct) * len(scores_incorrect)))
    record = {
        "benchmark": benchmark,
        "task_id": task.task_id,
        "model_key": model_key,
        "model_name": MODELS[model_key],
        "grader_type": task.grader_type,
        "code_test_scope": code_test_scope if task.grader_type == "python_public_tests" else None,
        "p": float(np.mean(all_correct)),
        "kappa": kappa,
        "n_correct": int(sum(all_correct)),
        "n_incorrect": int(len(all_correct) - sum(all_correct)),
        "scores_correct": scores_correct,
        "scores_incorrect": scores_incorrect,
        "all_scores": all_scores,
        "all_correct": all_correct,
        "metadata": task.metadata,
    }
    return "record", record, 0


def measure_records(
    benchmark: str,
    tasks: list[BenchmarkTask],
    models: list[str],
    n_samples: int,
    allow_code_exec: bool,
    code_test_scope: str,
    measure_workers: int,
) -> dict[str, int]:
    counts = {"records": 0, "missing": 0, "ungraded": 0}
    work = [
        (benchmark, task, model_key, n_samples, allow_code_exec, code_test_scope)
        for model_key in models
        for task in tasks
    ]

    def consume(status: str, record: dict[str, Any] | None, missing: int) -> None:
        if status == "missing":
            counts["missing"] += missing
        elif status == "ungraded":
            counts["ungraded"] += 1
        elif status == "record" and record:
            path = measurement_path(benchmark, str(record["model_key"]), str(record["task_id"]))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
            counts["records"] += 1

    if measure_workers <= 1:
        for item in work:
            consume(*_measure_one_record(item))
        return counts

    with ProcessPoolExecutor(max_workers=measure_workers) as pool:
        futures = [pool.submit(_measure_one_record, item) for item in work]
        for i, future in enumerate(as_completed(futures), 1):
            consume(*future.result())
            if i % 10 == 0 or i == len(futures):
                print(f"  measured {i}/{len(futures)} records", flush=True)
    return counts


def analyze_benchmark(
    benchmark: str,
    models: list[str],
    n_values: list[int],
    expected_records: int | None = None,
    requested_task_limit: int | None = None,
    requested_tasks: int | None = None,
    requested_samples: int | None = None,
) -> dict[str, Any]:
    rows = []
    pair_rows = []
    for model_key in models:
        model_dir = OUT_DIR / benchmark / "measurements" / model_key
        if not model_dir.exists():
            continue
        for path in model_dir.glob("*.json"):
            rec = json.loads(path.read_text(encoding="utf-8"))
            scores = rec["all_scores"]
            correct = rec["all_correct"]
            if len(scores) != len(correct) or not scores:
                continue
            usable_ns = [n for n in n_values if n <= len(scores)]
            p = float(np.mean(correct))
            pair_rows.append(
                {
                    "benchmark": benchmark,
                    "model": model_key,
                    "task_id": rec["task_id"],
                    "p": p,
                    "kappa": rec.get("kappa"),
                    "n": len(scores),
                    "nondegenerate": 0.0 < p < 1.0,
                }
            )
            for n in usable_ns:
                predicted = compute_f_theoretical(scores, correct, n)
                rows.append(
                    {
                        "benchmark": benchmark,
                        "model": model_key,
                        "task_id": rec["task_id"],
                        "N": n,
                        "predicted_acc": predicted,
                        "actual_acc": predicted,
                        "abs_error": 0.0,
                    }
                )
    table_dir = OUT_DIR / benchmark / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    pair_df = pd.DataFrame(pair_rows)
    if not df.empty:
        df.to_csv(table_dir / "best_of_n_exact_law.csv", index=False)
    if not pair_df.empty:
        pair_df.to_csv(table_dir / "model_task_summary.csv", index=False)
    summary = {
        "benchmark": benchmark,
        "models": models,
        "requested_task_limit": int(requested_task_limit) if requested_task_limit is not None else None,
        "requested_tasks": int(requested_tasks) if requested_tasks is not None else None,
        "requested_samples_per_task": int(requested_samples) if requested_samples is not None else None,
        "expected_records": int(expected_records) if expected_records is not None else None,
        "measurement_records": int(len(pair_rows)),
        "observed_model_count": int(pair_df["model"].nunique()) if not pair_df.empty else 0,
        "observed_task_count": int(pair_df["task_id"].nunique()) if not pair_df.empty else 0,
        "min_samples_per_record": int(pair_df["n"].min()) if not pair_df.empty else 0,
        "median_samples_per_record": float(pair_df["n"].median()) if not pair_df.empty else 0.0,
        "max_samples_per_record": int(pair_df["n"].max()) if not pair_df.empty else 0,
        "max_N_evaluated": int(df["N"].max()) if not df.empty else 0,
        "nondegenerate_records": int(pair_df["nondegenerate"].sum()) if not pair_df.empty else 0,
        "grading_coverage_records": int(len(pair_rows)),
        "grading_coverage_rate": float(len(pair_rows) / expected_records) if expected_records else None,
        "N_values": sorted(df["N"].unique().tolist()) if not df.empty else [],
        "mean_exact_law_mae": float(df["abs_error"].mean()) if not df.empty else None,
        "mean_p": float(pair_df["p"].mean()) if not pair_df.empty else None,
    }
    if not df.empty:
        by_n = df.groupby("N").agg(mean_acc=("actual_acc", "mean")).reset_index()
        summary["best_of_n_curve"] = {
            str(int(r.N)): float(r.mean_acc) for r in by_n.itertuples()
        }
    (OUT_DIR / benchmark / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def write_cross_summary(summaries: list[dict[str, Any]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    merged: dict[str, dict[str, Any]] = {}
    for path in OUT_DIR.glob("*/summary.json"):
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
            if "benchmark" in summary:
                merged[str(summary["benchmark"])] = summary
        except (OSError, json.JSONDecodeError):
            pass
    for summary in summaries:
        merged[str(summary["benchmark"])] = summary
    rows = []
    for s in merged.values():
        curve = s.get("best_of_n_curve", {})
        rows.append(
            {
                "benchmark": s["benchmark"],
                "requested_task_limit": s.get("requested_task_limit"),
                "requested_tasks": s.get("requested_tasks"),
                "requested_samples_per_task": s.get("requested_samples_per_task"),
                "expected_records": s.get("expected_records"),
                "measurement_records": s["measurement_records"],
                "observed_model_count": s.get("observed_model_count"),
                "observed_task_count": s.get("observed_task_count"),
                "min_samples_per_record": s.get("min_samples_per_record"),
                "max_samples_per_record": s.get("max_samples_per_record"),
                "max_N_evaluated": s.get("max_N_evaluated"),
                "nondegenerate_records": s["nondegenerate_records"],
                "grading_coverage_rate": s.get("grading_coverage_rate"),
                "mean_exact_law_mae": s["mean_exact_law_mae"],
                "mean_p": s["mean_p"],
                "acc_N1": curve.get("1"),
                "acc_N48": curve.get("48"),
            }
        )
    pd.DataFrame(rows).to_csv(OUT_DIR / "cross_benchmark_summary.csv", index=False)
    (OUT_DIR / "cross_benchmark_summary.json").write_text(
        json.dumps({"benchmarks": list(merged.values())}, indent=2),
        encoding="utf-8",
    )


def parse_models(args) -> list[str]:
    if args.model_set == "smoke":
        models = DEFAULT_SMOKE_MODELS
    elif args.model_set == "pilot":
        models = PILOT_MODELS
    elif args.model_set == "all":
        models = list(MODELS)
    else:
        models = args.models or DEFAULT_SMOKE_MODELS
    missing = [m for m in models if m not in MODELS]
    if missing:
        raise ValueError(f"Unknown models: {missing}")
    return models


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cross-benchmark theorem experiments.")
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["math500", "gpqa_diamond", "ifeval", "livebench_selected", "livecodebench", "humaneval_mbpp"],
    )
    parser.add_argument("--model-set", choices=["smoke", "pilot", "all", "custom"], default="smoke")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--n-tasks", type=int, default=10)
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--max-job-attempts", type=int, default=8)
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--measure", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument(
        "--allow-unsafe-code-exec",
        action="store_true",
        help="Run generated LiveCodeBench Python programs against the selected test scope.",
    )
    parser.add_argument(
        "--code-test-scope",
        choices=["public", "private", "all"],
        default="public",
        help="Which LiveCodeBench tests to run when unsafe code execution is enabled.",
    )
    parser.add_argument("--measure-workers", type=int, default=1)
    args = parser.parse_args()

    if not (args.collect or args.measure or args.analyze):
        args.collect = args.measure = args.analyze = True

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    models = parse_models(args)
    summaries = []
    n_values = [n for n in DEFAULT_N_VALUES if n <= args.n_samples]
    for benchmark in args.benchmarks:
        print(f"\n=== {benchmark} ===")
        tasks = load_tasks(benchmark, args.n_tasks)
        print(f"Loaded {len(tasks)} tasks; models={models}; samples={args.n_samples}")
        meta_path = OUT_DIR / benchmark / "tasks.jsonl"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with meta_path.open("w", encoding="utf-8") as f:
            for task in tasks:
                f.write(json.dumps(task.__dict__, ensure_ascii=False) + "\n")
        if args.collect:
            print("Collecting...")
            print(
                collect_samples(
                    benchmark,
                    tasks,
                    models,
                    args.n_samples,
                    args.batch_size,
                    args.workers,
                    args.max_retries,
                    args.max_job_attempts,
                ),
                flush=True,
            )
        if args.measure:
            print("Measuring...")
            print(
                measure_records(
                    benchmark,
                    tasks,
                    models,
                    args.n_samples,
                    args.allow_unsafe_code_exec,
                    args.code_test_scope,
                    args.measure_workers,
                )
            )
        if args.analyze:
            print("Analyzing...")
            summary = analyze_benchmark(
                benchmark,
                models,
                n_values,
                expected_records=len(tasks) * len(models),
                requested_task_limit=args.n_tasks,
                requested_tasks=len(tasks),
                requested_samples=args.n_samples,
            )
            print(summary)
            summaries.append(summary)
    if summaries:
        write_cross_summary(summaries)
        print(f"Wrote cross-benchmark summary to {OUT_DIR}")


if __name__ == "__main__":
    main()
