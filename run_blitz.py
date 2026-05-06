"""
Blitz collection: all remaining models in one shared queue, 24 workers.
70B uses n=2, all others use n=16. Maximizes throughput by spreading
requests across different models so per-model rate limits don't collide.

Usage:
    python run_blitz.py                  # default 24 workers
    python run_blitz.py --workers 20     # fewer workers
"""

import sys
import json
import time
import hashlib
import argparse
import threading
import logging
import random
from pathlib import Path
from queue import Queue, Empty
from itertools import cycle

sys.path.insert(0, str(Path(__file__).parent))
from config import NIM_API_KEYS, NIM_BASE_URL, MODELS, N_SAMPLES, N_PROBLEMS

from openai import OpenAI

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

PROJECT = Path(__file__).resolve().parent
RAW_DIR = PROJECT / "data" / "raw"
LOG_DIR = PROJECT / "logs"

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "blitz.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

SKIP_MODELS = {"meta/llama-3.3-70b-instruct", "mistralai/mistral-nemotron"}
SMALL_BATCH_MODELS = set()
SMALL_BATCH_N = 2


def cache_path(model, problem, idx):
    h = hashlib.md5(f"{model}:{problem}:{idx}".encode()).hexdigest()
    return RAW_DIR / f"{h}.json"


def load_math500():
    p = PROJECT / "data" / "math500.jsonl"
    problems = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "problem" in obj and "answer" in obj:
                problems.append(obj)
            if len(problems) >= N_PROBLEMS:
                break
    return problems


def call_api(client, model, problem, n, start_idx, max_retries=5):
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Solve the math problem. Show your work step by step. Put your final answer in \\boxed{}."},
                    {"role": "user", "content": problem},
                ],
                max_tokens=2048,
                temperature=0.7,
                n=n,
                logprobs=True,
                top_logprobs=1,
            )
            for i, choice in enumerate(resp.choices):
                lp = choice.logprobs
                logprobs = None
                if lp and lp.content:
                    logprobs = [{"token": t.token, "logprob": t.logprob} for t in lp.content]
                result = {
                    "content": choice.message.content,
                    "logprobs": logprobs,
                    "model": model,
                    "problem": problem,
                    "sample_idx": start_idx + i,
                }
                fp = cache_path(model, problem, start_idx + i)
                fp.write_text(json.dumps(result), encoding="utf-8")
            return True
        except Exception as e:
            err = str(e)
            if "401" in err or "403" in err:
                raise KeyError(err[:120])
            wait = min(2 ** attempt * 4, 90)
            if "429" in err:
                wait = min(2 ** attempt * 6, 90)
            time.sleep(wait)
    raise RuntimeError(f"Failed after {max_retries} retries")


def worker(wid, key_iter, key_lock, work_q, counter):
    client_cache = {}
    while True:
        try:
            item = work_q.get(timeout=5)
        except Empty:
            if work_q.empty():
                return
            continue

        model, problem, start_idx, batch_n = item
        try:
            already = all(cache_path(model, problem, start_idx + j).exists() for j in range(batch_n))
            if already:
                with counter["lock"]:
                    counter["cached"] += 1
                    counter["done"] += 1
                work_q.task_done()
                continue

            with key_lock:
                key = next(key_iter)
            if key not in client_cache:
                client_cache[key] = OpenAI(base_url=NIM_BASE_URL, api_key=key)

            call_api(client_cache[key], model, problem, batch_n, start_idx)
            with counter["lock"]:
                counter["new"] += 1
                counter["done"] += 1
        except KeyError as e:
            log.warning(f"W{wid} dead key: {e}")
            work_q.put(item)
        except Exception as e:
            log.error(f"W{wid} {model[:20]} err: {str(e)[:120]}")
            with counter["lock"]:
                counter["errors"] += 1
                counter["done"] += 1

        work_q.task_done()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=48)
    args = parser.parse_args()

    problems = load_math500()
    print(f"Loaded {len(problems)} problems, {len(NIM_API_KEYS)} keys, {args.workers} workers")

    work_q = Queue()
    total = 0

    for model_key, model_name in MODELS.items():
        if model_name in SKIP_MODELS:
            continue
        if model_name in SMALL_BATCH_MODELS:
            for prob in problems:
                for start in range(0, N_SAMPLES, SMALL_BATCH_N):
                    bn = min(SMALL_BATCH_N, N_SAMPLES - start)
                    missing = any(not cache_path(model_name, prob["problem"], start + j).exists() for j in range(bn))
                    if missing:
                        work_q.put((model_name, prob["problem"], start, bn))
                        total += 1
        else:
            for prob in problems:
                if not all(cache_path(model_name, prob["problem"], i).exists() for i in range(N_SAMPLES)):
                    work_q.put((model_name, prob["problem"], 0, N_SAMPLES))
                    total += 1

    print(f"Total work items: {total}")
    if total == 0:
        print("Nothing to do!")
        return

    # Shuffle so different models are interleaved — avoids hammering one model
    items = list(work_q.queue)
    random.shuffle(items)
    work_q = Queue()
    for item in items:
        work_q.put(item)

    counter = {"done": 0, "cached": 0, "new": 0, "errors": 0, "lock": threading.Lock()}
    key_iter = cycle(NIM_API_KEYS)
    key_lock = threading.Lock()

    pbar = tqdm(total=total, desc="Blitz collection", smoothing=0.1) if tqdm else None

    threads = []
    for wid in range(args.workers):
        t = threading.Thread(target=worker, args=(wid, key_iter, key_lock, work_q, counter), daemon=True)
        t.start()
        threads.append(t)

    last = 0
    while any(t.is_alive() for t in threads):
        time.sleep(1)
        cur = counter["new"] + counter["errors"] + counter["cached"]
        if pbar and cur > last:
            pbar.update(cur - last)
            pbar.set_postfix_str(f"ok={counter['new']} err={counter['errors']} skip={counter['cached']}")
            last = cur

    for t in threads:
        t.join(timeout=5)
    if pbar:
        pbar.close()

    print(f"\nDONE: new={counter['new']} errors={counter['errors']} cached={counter['cached']}")


if __name__ == "__main__":
    main()
