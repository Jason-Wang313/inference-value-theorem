"""
NIM API client — uses n=16 batch completions for max throughput.
One API call returns all 16 samples for a problem. 17 keys in parallel.
"""

import time
import json
import hashlib
import threading
import os
from pathlib import Path
from openai import OpenAI
from queue import Queue, Empty

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import NIM_API_KEYS, NIM_BASE_URL, RATE_LIMIT_PER_KEY, N_SAMPLES

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve_cache_dir() -> Path:
    explicit = os.environ.get("INFERENCE_VALUE_RAW_DIR") or os.environ.get("NIM_CACHE_DIR")
    if explicit:
        return Path(explicit)

    local = PROJECT_ROOT / "data" / "raw"
    sibling = PROJECT_ROOT.parent / "inference-value-theorem" / "data" / "raw"
    try:
        local_has_cache = any(local.glob("*.json"))
    except OSError:
        local_has_cache = False
    if not local_has_cache and sibling.exists():
        return sibling
    return local


CACHE_DIR = _resolve_cache_dir()


def _cache_path(model, problem, sample_idx):
    h = hashlib.md5(f"{model}:{problem}:{sample_idx}".encode()).hexdigest()
    return CACHE_DIR / f"{h}.json"


def _extract_logprobs(choice):
    lp = choice.logprobs
    if lp is None:
        return None
    return [{"token": t.token, "logprob": t.logprob} for t in lp.content]


def compute_score(logprobs_list):
    """Mean log-probability as self-score."""
    if not logprobs_list:
        return float('-inf')
    return sum(t["logprob"] for t in logprobs_list) / len(logprobs_list)


class NIMClient:
    """Holds pre-built OpenAI clients for each key."""
    def __init__(self):
        self.keys = list(NIM_API_KEYS)
        self.clients = {k: OpenAI(base_url=NIM_BASE_URL, api_key=k) for k in self.keys}

    def get_cache_key(self, model, problem, sample_idx):
        return _cache_path(model, problem, sample_idx)


def _all_cached(model, problem, n_samples):
    """Check if all n_samples are already cached for this problem."""
    for i in range(n_samples):
        if not _cache_path(model, problem, i).exists():
            return False
    return True


def _call_batch(client, model, problem, n_samples, max_retries=4):
    """One API call with n=n_samples. Returns list of result dicts, or raises."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Solve the math problem. Show your work step by step. Put your final answer in \\boxed{}."},
                    {"role": "user", "content": problem}
                ],
                max_tokens=2048,
                temperature=0.7,
                n=n_samples,
                logprobs=True,
                top_logprobs=5,
            )

            results = []
            for i, choice in enumerate(response.choices):
                result = {
                    "content": choice.message.content,
                    "logprobs": _extract_logprobs(choice),
                    "model": model,
                    "problem": problem,
                    "sample_idx": i,
                }
                cache_file = _cache_path(model, problem, i)
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                cache_file.write_text(json.dumps(result), encoding="utf-8")
                results.append(result)
            return results

        except Exception as e:
            err = str(e)
            if "401" in err or "403" in err:
                raise KeyError(f"Dead key: {err[:80]}")
            wait = min(2 ** attempt * 3, 60)
            if "429" in err:
                wait = min(2 ** attempt * 5, 90)
            time.sleep(wait)

    raise RuntimeError(f"Failed after {max_retries} retries")


def _key_worker(key, client_map, model, work_queue, counter, n_samples):
    """One thread per API key. Each call gets n_samples completions."""
    openai_client = client_map[key]
    while True:
        try:
            item = work_queue.get(timeout=3)
        except Empty:
            if work_queue.empty():
                return
            continue

        prob_idx, problem = item
        try:
            if _all_cached(model, problem, n_samples):
                with counter["lock"]:
                    counter["cached"] += 1
                    counter["done"] += 1
            else:
                # Check which samples are missing and only fetch those via full batch
                _call_batch(openai_client, model, problem, n_samples)
                with counter["lock"]:
                    counter["new"] += 1
                    counter["done"] += 1
        except KeyError:
            work_queue.put(item)
            with counter["lock"]:
                counter["dead_keys"].add(key)
            return
        except Exception as e:
            with counter["lock"]:
                counter["errors"] += 1
                counter["done"] += 1

        work_queue.task_done()


def collect_parallel(nim_client, model_name, problems, n_samples, progress_cb=None):
    """Collect all samples using n=N_SAMPLES per API call, 17 keys in parallel."""
    work_queue = Queue()
    for prob_idx, prob in enumerate(problems):
        work_queue.put((prob_idx, prob["problem"]))

    total = len(problems)
    counter = {
        "done": 0, "cached": 0, "new": 0, "errors": 0,
        "lock": threading.Lock(), "dead_keys": set(),
    }

    threads = []
    for key in nim_client.keys:
        t = threading.Thread(
            target=_key_worker,
            args=(key, nim_client.clients, model_name, work_queue, counter, n_samples),
            daemon=True,
        )
        t.start()
        threads.append(t)

    while any(t.is_alive() for t in threads):
        time.sleep(1)
        if progress_cb:
            progress_cb(counter, total)

    for t in threads:
        t.join(timeout=5)

    return counter
