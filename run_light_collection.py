"""
Lightweight collection: 4 worker threads, 22 keys round-robin, one model at a time.
Keeps RAM under ~2 GB by reusing a small pool of OpenAI clients.

Usage:
    python run_light_collection.py                  # all uncollected models
    python run_light_collection.py --model 70B      # single model
    python run_light_collection.py --workers 2      # fewer threads (even lighter)
"""

import sys
import json
import time
import random
import hashlib
import argparse
import threading
import logging
from pathlib import Path
from queue import Queue, Empty
from itertools import cycle

sys.path.insert(0, str(Path(__file__).parent))
from config import NIM_API_KEYS, NIM_BASE_URL, MODELS, N_SAMPLES, N_PROBLEMS

try:
    from openai import OpenAI
except ImportError:
    print("pip install openai"); sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

PROJECT = Path(__file__).resolve().parent
RAW_DIR = PROJECT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = PROJECT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "light_collection.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)


def cache_path(model, problem, idx):
    h = hashlib.md5(f"{model}:{problem}:{idx}".encode()).hexdigest()
    return RAW_DIR / f"{h}.json"


def all_cached(model, problem, n):
    return all(cache_path(model, problem, i).exists() for i in range(n))


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


def call_single(client, model, problem, sample_idx, max_retries=8):
    """Single n=1 call with jittered backoff. SDK retries disabled to prevent storm."""
    fp = cache_path(model, problem, sample_idx)
    if fp.exists():
        return True
    last_err = None
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
                n=1,
                logprobs=True,
                top_logprobs=1,
            )
            choice = resp.choices[0]
            lp = choice.logprobs
            logprobs = None
            if lp and lp.content:
                logprobs = [{"token": t.token, "logprob": t.logprob} for t in lp.content]
            result = {
                "content": choice.message.content,
                "logprobs": logprobs,
                "model": model,
                "problem": problem,
                "sample_idx": sample_idx,
            }
            fp.write_text(json.dumps(result), encoding="utf-8")
            return True
        except Exception as e:
            last_err = str(e)
            if "401" in last_err or "403" in last_err:
                raise KeyError(last_err[:120])
            if "404" in last_err:
                raise RuntimeError(f"Model not found (404): {last_err[:120]}")
            time.sleep(min(2 ** attempt, 10) + random.random())
    return False


def call_batch_rotating(model, problem, n_samples, key_iter, key_lock, client_cache):
    """Collect all samples for a problem, rotating keys per call."""
    ok = 0
    fail = 0
    for i in range(n_samples):
        fp = cache_path(model, problem, i)
        if fp.exists():
            ok += 1
            continue
        with key_lock:
            key = next(key_iter)
        if key not in client_cache:
            client_cache[key] = OpenAI(base_url=NIM_BASE_URL, api_key=key, max_retries=2)
        if call_single(client_cache[key], model, problem, i, max_retries=3):
            ok += 1
        else:
            fail += 1
    if fail > 0:
        raise RuntimeError(f"{fail}/{n_samples} samples failed")
    return True


def worker(wid, key_iter, key_lock, model_name, work_q, counter, n_samples):
    """Worker thread: rotates keys per API call for even distribution."""
    client_cache = {}
    while True:
        try:
            item = work_q.get(timeout=3)
        except Empty:
            if work_q.empty():
                return
            continue

        pidx, problem = item
        try:
            if all_cached(model_name, problem, n_samples):
                with counter["lock"]:
                    counter["cached"] += 1
                    counter["done"] += 1
                work_q.task_done()
                continue

            call_batch_rotating(model_name, problem, n_samples, key_iter, key_lock, client_cache)
            with counter["lock"]:
                counter["new"] += 1
                counter["done"] += 1
        except KeyError as e:
            log.warning(f"W{wid} dead key on problem {pidx}: {e}")
            work_q.put(item)
            with counter["lock"]:
                counter["dead_keys"].add(str(e)[:40])
        except Exception as e:
            log.error(f"W{wid} error on problem {pidx}: {str(e)[:200]}")
            with counter["lock"]:
                counter["errors"] += 1
                counter["done"] += 1

        work_q.task_done()


def collect_model(model_key, model_name, problems, n_samples, n_workers):
    needed = [
        (i, p) for i, p in enumerate(problems)
        if not all_cached(model_name, p["problem"], n_samples)
    ]
    log.info(f"{model_key}: {len(needed)} problems need collection, {len(problems)-len(needed)} cached")
    if not needed:
        return {"cached": len(problems), "new": 0, "errors": 0, "done": len(problems), "dead_keys": set()}

    work_q = Queue()
    for pidx, prob in needed:
        work_q.put((pidx, prob["problem"]))

    counter = {
        "done": 0, "cached": len(problems) - len(needed),
        "new": 0, "errors": 0,
        "lock": threading.Lock(), "dead_keys": set(),
    }

    key_iter = cycle(NIM_API_KEYS)
    key_lock = threading.Lock()

    pbar = tqdm(total=len(needed), desc=f"Collecting {model_key}") if tqdm else None

    threads = []
    for wid in range(n_workers):
        t = threading.Thread(target=worker, args=(wid, key_iter, key_lock, model_name, work_q, counter, n_samples), daemon=True)
        t.start()
        threads.append(t)

    last_new = 0
    while any(t.is_alive() for t in threads):
        time.sleep(1)
        cur = counter["new"] + counter["errors"]
        if pbar and cur > last_new:
            pbar.update(cur - last_new)
            pbar.set_postfix_str(f"ok={counter['new']} err={counter['errors']} dead={len(counter['dead_keys'])}")
            last_new = cur

    for t in threads:
        t.join(timeout=5)
    if pbar:
        pbar.update(counter["new"] + counter["errors"] - last_new)
        pbar.close()

    counter["done"] = counter["cached"] + counter["new"] + counter["errors"]
    return counter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, help="Single model key to collect")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker threads (default: 4)")
    parser.add_argument("--n_samples", type=int, default=N_SAMPLES)
    args = parser.parse_args()

    problems = load_math500()
    log.info(f"Loaded {len(problems)} problems, {len(NIM_API_KEYS)} API keys, {args.workers} workers")

    if args.model:
        targets = {args.model: MODELS[args.model]}
    else:
        targets = dict(MODELS)

    for model_key, model_name in targets.items():
        log.info(f"\n{'='*60}")
        log.info(f"START: {model_key} ({model_name})")
        start = time.time()

        for attempt in range(50):
            try:
                rc = collect_model(model_key, model_name, problems, args.n_samples, args.workers)
                elapsed = time.time() - start
                log.info(f"DONE: {model_key} pass {attempt+1} — cached={rc['cached']} new={rc['new']} errors={rc['errors']} dead_keys={len(rc['dead_keys'])} time={elapsed/60:.1f}min")
                if rc['errors'] == 0:
                    break
                log.info(f"Retrying {model_key} — {rc['errors']} problems had failures")
            except Exception as e:
                log.error(f"FATAL: {model_key} — {e}")
                break

    log.info("ALL MODELS COMPLETE")


if __name__ == "__main__":
    main()
