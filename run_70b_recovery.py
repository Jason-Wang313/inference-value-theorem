"""
70B recovery: uses n=2 batch size instead of n=16.
Fills in missing samples individually. Safe to run alongside or after main collection.

Usage:
    python run_70b_recovery.py               # default: 8 workers, n=2
    python run_70b_recovery.py --batch 4     # try n=4
"""

import sys
import json
import time
import hashlib
import argparse
import threading
import logging
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
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "70b_recovery2.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


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


def call_small_batch(client, model, problem, start_idx, batch_n, max_retries=5):
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
                n=batch_n,
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
            wait = min(2 ** attempt * 5, 120)
            if "429" in err:
                wait = min(2 ** attempt * 8, 120)
            time.sleep(wait)
    raise RuntimeError(f"Failed after {max_retries} retries")


def worker(wid, key_iter, key_lock, work_q, counter, batch_n):
    client_cache = {}
    while True:
        try:
            item = work_q.get(timeout=3)
        except Empty:
            if work_q.empty():
                return
            continue

        pidx, problem, start_idx, model_name = item
        try:
            already = all(cache_path(model_name, problem, start_idx + j).exists() for j in range(batch_n))
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

            call_small_batch(client_cache[key], model_name, problem, start_idx, batch_n)
            with counter["lock"]:
                counter["new"] += 1
                counter["done"] += 1
        except KeyError as e:
            log.warning(f"W{wid} dead key: {e}")
            work_q.put(item)
        except Exception as e:
            log.error(f"W{wid} prob={pidx} idx={start_idx}: {str(e)[:150]}")
            with counter["lock"]:
                counter["errors"] += 1
                counter["done"] += 1

        work_q.task_done()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=2, help="Batch size per API call (default: 2)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--models", type=str, default="70B", help="Comma-separated model keys (default: 70B)")
    args = parser.parse_args()

    model_keys = [k.strip() for k in args.models.split(",")]
    problems = load_math500()
    batch_n = args.batch

    work_q = Queue()
    total_work = 0
    for mk in model_keys:
        model_name = MODELS[mk]
        for pidx, prob in enumerate(problems):
            for start in range(0, N_SAMPLES, batch_n):
                actual_batch = min(batch_n, N_SAMPLES - start)
                missing = any(not cache_path(model_name, prob["problem"], start + j).exists() for j in range(actual_batch))
                if missing:
                    work_q.put((pidx, prob["problem"], start, model_name))
                    total_work += 1

    log.info(f"Recovery ({','.join(model_keys)}): {total_work} batch calls needed (n={batch_n}), {args.workers} workers")
    if total_work == 0:
        log.info("Nothing to do — 70B fully cached!")
        return

    counter = {"done": 0, "cached": 0, "new": 0, "errors": 0, "lock": threading.Lock()}
    key_iter = cycle(NIM_API_KEYS)
    key_lock = threading.Lock()

    pbar = tqdm(total=total_work, desc=f"Recovery ({','.join(model_keys)})") if tqdm else None

    threads = []
    for wid in range(args.workers):
        t = threading.Thread(target=worker, args=(wid, key_iter, key_lock, work_q, counter, batch_n), daemon=True)
        t.start()
        threads.append(t)

    last = 0
    while any(t.is_alive() for t in threads):
        time.sleep(1)
        cur = counter["new"] + counter["errors"] + counter["cached"]
        if pbar and cur > last:
            pbar.update(cur - last)
            pbar.set_postfix_str(f"ok={counter['new']} err={counter['errors']} cached={counter['cached']}")
            last = cur

    for t in threads:
        t.join(timeout=5)
    if pbar:
        pbar.close()

    log.info(f"70B DONE: new={counter['new']} errors={counter['errors']} cached={counter['cached']}")


if __name__ == "__main__":
    main()
