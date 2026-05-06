"""
Recovery collection for models with incomplete data.
- 8B: uses n=16 batch (same as 3B, which worked)
- 70B: uses n=1 individual calls (n=16 batch fails for 70B)

Usage:
  python experiments/01_collect_recovery.py --model 8B
  python experiments/01_collect_recovery.py --model 70B
"""

import sys
import json
import argparse
import time
import hashlib
import threading
from pathlib import Path
from queue import Queue, Empty
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import MODELS, N_SAMPLES, N_PROBLEMS, NIM_API_KEYS, NIM_BASE_URL

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

CACHE_DIR = Path(__file__).parent.parent / "data" / "raw"


def _cache_path(model, problem, sample_idx):
    h = hashlib.md5(f"{model}:{problem}:{sample_idx}".encode()).hexdigest()
    return CACHE_DIR / f"{h}.json"


def _extract_logprobs(choice):
    lp = choice.logprobs
    if lp is None:
        return None
    return [{"token": t.token, "logprob": t.logprob} for t in lp.content]


def _missing_samples(model, problem, n_samples):
    missing = []
    for i in range(n_samples):
        if not _cache_path(model, problem, i).exists():
            missing.append(i)
    return missing


def _call_single(client, model, problem, sample_idx, max_retries=3):
    """One API call with n=1. Returns result dict or raises."""
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
                n=1,
                logprobs=True,
                top_logprobs=5,
            )
            choice = response.choices[0]
            result = {
                "content": choice.message.content,
                "logprobs": _extract_logprobs(choice),
                "model": model,
                "problem": problem,
                "sample_idx": sample_idx,
            }
            cache_file = _cache_path(model, problem, sample_idx)
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(result), encoding="utf-8")
            return result
        except Exception as e:
            err = str(e)
            if "401" in err or "403" in err:
                raise KeyError(f"Dead key: {err[:80]}")
            if "429" in err:
                wait = 5 + attempt * 5
            else:
                wait = min(2 ** attempt * 2, 20)
            time.sleep(wait)
    raise RuntimeError(f"Failed after {max_retries} retries")


def _call_batch(client, model, problem, n_samples, max_retries=4):
    """One API call with n=n_samples. Returns list of result dicts."""
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


# ---- Workers ----

def _batch_worker(key, clients, model, work_queue, counter, n_samples):
    """Thread worker for n=16 batch mode (8B)."""
    client = clients[key]
    while True:
        try:
            item = work_queue.get(timeout=3)
        except Empty:
            if work_queue.empty():
                return
            continue
        prob_idx, problem = item
        try:
            missing = _missing_samples(model, problem, n_samples)
            if not missing:
                with counter["lock"]:
                    counter["cached"] += 1
                    counter["done"] += 1
            else:
                _call_batch(client, model, problem, n_samples)
                with counter["lock"]:
                    counter["new"] += 1
                    counter["done"] += 1
        except KeyError:
            work_queue.put(item)
            with counter["lock"]:
                counter["dead_keys"].add(key)
            return
        except Exception:
            with counter["lock"]:
                counter["errors"] += 1
                counter["done"] += 1
        work_queue.task_done()


def _individual_worker(key, clients, model, work_queue, counter, stop_event):
    """Thread worker for n=1 individual mode. Work items are (prob_idx, problem, sample_idx)."""
    client = clients[key]
    consecutive_empty = 0
    while not stop_event.is_set():
        try:
            item = work_queue.get(timeout=2)
            consecutive_empty = 0
        except Empty:
            consecutive_empty += 1
            if consecutive_empty >= 5:
                return
            continue
        prob_idx, problem, sample_idx = item
        try:
            if _cache_path(model, problem, sample_idx).exists():
                with counter["lock"]:
                    counter["cached"] += 1
                    counter["done"] += 1
            else:
                _call_single(client, model, problem, sample_idx)
                with counter["lock"]:
                    counter["new"] += 1
                    counter["done"] += 1
                time.sleep(0.5)
        except KeyError:
            work_queue.put(item)
            with counter["lock"]:
                counter["dead_keys"].add(key)
            return
        except Exception:
            work_queue.put(item)
            with counter["lock"]:
                counter["errors"] += 1
            time.sleep(2)
        work_queue.task_done()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()))
    parser.add_argument("--n_samples", type=int, default=N_SAMPLES)
    parser.add_argument("--n_problems", type=int, default=N_PROBLEMS)
    parser.add_argument("--n_keys", type=int, default=None, help="Limit number of API keys used")
    args = parser.parse_args()

    model_key = args.model
    model_name = MODELS[model_key]

    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "math500.jsonl"

    problems = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "problem" in obj and "answer" in obj:
                problems.append(obj)
            if len(problems) >= args.n_problems:
                break

    print(f"Loaded {len(problems)} problems.")

    keys_to_use = NIM_API_KEYS[:args.n_keys] if args.n_keys else NIM_API_KEYS
    clients = {k: OpenAI(base_url=NIM_BASE_URL, api_key=k) for k in keys_to_use}

    use_batch = False
    batch_n = 1

    print(f"Model: {model_key} ({model_name})")
    print(f"Mode: {'batch n=' + str(batch_n) if use_batch else 'individual n=1'}")

    if use_batch:
        work_queue = Queue()
        n_queued = 0
        for i, prob in enumerate(problems):
            missing = _missing_samples(model_name, prob["problem"], args.n_samples)
            if missing:
                work_queue.put((i, prob["problem"]))
                n_queued += 1
        total = len(problems)
        print(f"Need collection: {n_queued}/{total} problems")
    else:
        work_queue = Queue()
        n_queued = 0
        for i, prob in enumerate(problems):
            for s in range(args.n_samples):
                if not _cache_path(model_name, prob["problem"], s).exists():
                    work_queue.put((i, prob["problem"], s))
                    n_queued += 1
        total_samples = len(problems) * args.n_samples
        print(f"Need collection: {n_queued}/{total_samples} individual samples")
        total = total_samples

    counter = {
        "done": 0, "cached": 0, "new": 0, "errors": 0,
        "lock": threading.Lock(), "dead_keys": set(),
    }
    stop_event = threading.Event()

    start_time = time.time()
    pbar = tqdm(total=n_queued + (len(problems) - n_queued if use_batch else total_samples - n_queued),
                desc=f"Collecting {model_key}") if tqdm else None

    threads_per_key = 1
    threads = []
    for key in keys_to_use:
        for _ in range(threads_per_key):
            if use_batch:
                t = threading.Thread(target=_batch_worker,
                                     args=(key, clients, model_name, work_queue, counter, args.n_samples),
                                     daemon=True)
            else:
                t = threading.Thread(target=_individual_worker,
                                     args=(key, clients, model_name, work_queue, counter, stop_event),
                                     daemon=True)
            t.start()
            threads.append(t)

    last_done = 0
    while any(t.is_alive() for t in threads):
        time.sleep(1)
        done = counter["done"]
        if pbar and done > last_done:
            pbar.update(done - last_done)
            pbar.set_postfix_str(
                f"cached={counter['cached']} new={counter['new']} err={counter['errors']} dead={len(counter['dead_keys'])}"
            )
            last_done = done

    for t in threads:
        t.join(timeout=5)

    if pbar:
        pbar.update(counter["done"] - last_done)
        pbar.close()

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"RECOVERY COLLECTION SUMMARY")
    print(f"{'='*60}")
    print(f"Model:        {model_key} ({model_name})")
    print(f"Mode:         {'batch n=' + str(batch_n) if use_batch else 'individual n=1'}")
    print(f"Cached:       {counter['cached']}")
    print(f"New calls:    {counter['new']}")
    print(f"Errors:       {counter['errors']}")
    print(f"Dead keys:    {len(counter['dead_keys'])}")
    print(f"Elapsed:      {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"{'='*60}")

    # Verify completeness
    complete = 0
    for prob in problems:
        missing = _missing_samples(model_name, prob["problem"], args.n_samples)
        if not missing:
            complete += 1
    print(f"\nComplete problems: {complete}/{len(problems)}")


if __name__ == "__main__":
    main()
