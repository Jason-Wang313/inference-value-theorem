"""
Gap filler: fills missing cache entries with retries and per-model cooldown.
When a model hits consecutive 429s, it's paused for 30 min to let the rate limit expire.

Usage:
    python run_gap_filler.py                          # all active models, 10 workers
    python run_gap_filler.py --workers 6               # fewer workers
    python run_gap_filler.py --models KimiK2,MiniMax  # specific models only
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
from collections import defaultdict
from itertools import cycle

sys.path.insert(0, str(Path(__file__).parent))
from config import NIM_API_KEYS, NIM_BASE_URL, MODELS, N_SAMPLES, N_PROBLEMS

try:
    from openai import OpenAI
except ImportError:
    print("pip install openai"); sys.exit(1)

PROJECT = Path(__file__).resolve().parent
RAW_DIR = PROJECT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = PROJECT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "gap_filler.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

COOLDOWN_SECONDS = 120   # 2 min brief pause
COOLDOWN_THRESHOLD = 200 # effectively disabled — let SDK handle 429 retries


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


def scan_missing(problems, target_models, n_samples):
    log.info("Building file index...")
    existing = set(f.name for f in RAW_DIR.iterdir() if f.suffix == ".json")
    log.info(f"Indexed {len(existing)} cached files")

    by_model = defaultdict(list)
    for model_key, model_name in target_models.items():
        for pidx, p in enumerate(problems):
            prob_text = p["problem"]
            missing_for_prob = []
            for sidx in range(n_samples):
                h = hashlib.md5(f"{model_name}:{prob_text}:{sidx}".encode()).hexdigest()
                if f"{h}.json" not in existing:
                    missing_for_prob.append(sidx)
            if missing_for_prob:
                for sidx in missing_for_prob:
                    by_model[model_name].append((model_name, prob_text, sidx, len(missing_for_prob)))
    for model_name in by_model:
        by_model[model_name].sort(key=lambda x: x[3])

    all_items = []
    model_iters = {m: iter(items) for m, items in by_model.items()}
    while model_iters:
        exhausted = []
        for m in list(model_iters.keys()):
            item = next(model_iters[m], None)
            if item is None:
                exhausted.append(m)
            else:
                all_items.append((item[0], item[1], item[2]))
        for m in exhausted:
            del model_iters[m]

    return all_items


def is_cooled_down(model, cooldowns, cd_lock):
    with cd_lock:
        if model not in cooldowns:
            return False
        until, _ = cooldowns[model]
        return time.time() < until


def record_fail(model, cooldowns, cd_lock):
    with cd_lock:
        if model in cooldowns:
            until, count = cooldowns[model]
            count += 1
        else:
            until = 0
            count = 1
        if count >= COOLDOWN_THRESHOLD:
            until = time.time() + COOLDOWN_SECONDS
            short = [k for k, v in MODELS.items() if v == model]
            label = short[0] if short else model
            log.warning(f"COOLDOWN: {label} hit {count} consecutive fails — pausing for {COOLDOWN_SECONDS//60} min")
        cooldowns[model] = (until, count)


def record_success(model, cooldowns, cd_lock):
    with cd_lock:
        if model in cooldowns:
            _, old_count = cooldowns[model]
            if old_count > 0:
                short = [k for k, v in MODELS.items() if v == model]
                label = short[0] if short else model
                log.info(f"RECOVERED: {label} responding again (was at {old_count} consecutive fails)")
            cooldowns[model] = (0, 0)


def worker(wid, work_items, idx_lock, idx_counter, key_iter, key_lock, client_cache, stats, stop_event, cooldowns, cd_lock):
    while not stop_event.is_set():
        with idx_lock:
            i = idx_counter[0]
            if i >= len(work_items):
                return
            idx_counter[0] += 1

        model, problem, sidx = work_items[i]
        fp = cache_path(model, problem, sidx)
        if fp.exists():
            with stats["lock"]:
                stats["skipped"] += 1
            continue

        if is_cooled_down(model, cooldowns, cd_lock):
            with stats["lock"]:
                stats["cooled"] += 1
            continue

        with key_lock:
            key = next(key_iter)
        if key not in client_cache:
            client_cache[key] = OpenAI(base_url=NIM_BASE_URL, api_key=key, max_retries=3)
        try:
            resp = client_cache[key].chat.completions.create(
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
                timeout=300,
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
                "sample_idx": sidx,
            }
            fp.write_text(json.dumps(result), encoding="utf-8")
            with stats["lock"]:
                stats["ok"] += 1
            record_success(model, cooldowns, cd_lock)
            time.sleep(0.3 + random.random() * 0.3)
        except Exception as e:
            short = [k for k, v in MODELS.items() if v == model]
            label = short[0] if short else model
            log.warning(f"W{wid} {label}: {str(e)[:120]}")
            with stats["lock"]:
                stats["fail"] += 1
            record_fail(model, cooldowns, cd_lock)
            time.sleep(1.0 + random.random())


def run_cycle(problems, target_models, n_samples, n_workers, stop_event, cooldowns, cd_lock):
    work_items = scan_missing(problems, target_models, n_samples)
    if not work_items:
        return 0

    model_counts = defaultdict(int)
    for m, _, _ in work_items:
        model_counts[m] += 1
    log.info(f"Gap scan: {len(work_items)} missing samples across {len(model_counts)} models")
    for m, c in sorted(model_counts.items(), key=lambda x: -x[1]):
        short = [k for k, v in MODELS.items() if v == m]
        label = short[0] if short else m
        log.info(f"  {label}: {c} missing")

    stats = {"ok": 0, "fail": 0, "skipped": 0, "cooled": 0, "lock": threading.Lock()}
    idx_counter = [0]
    idx_lock = threading.Lock()
    key_iter = cycle(NIM_API_KEYS)
    key_lock = threading.Lock()

    threads = []
    for wid in range(n_workers):
        client_cache = {}
        t = threading.Thread(
            target=worker,
            args=(wid, work_items, idx_lock, idx_counter, key_iter, key_lock, client_cache, stats, stop_event, cooldowns, cd_lock),
            daemon=True,
        )
        t.start()
        threads.append(t)

    start = time.time()
    while any(t.is_alive() for t in threads):
        time.sleep(10)
        elapsed = time.time() - start
        progress = idx_counter[0]
        rate = stats["ok"] / (elapsed / 60) if elapsed > 0 else 0

        # check cooldown status
        with cd_lock:
            in_cd = []
            for m, (until, count) in cooldowns.items():
                if time.time() < until:
                    short = [k for k, v in MODELS.items() if v == m]
                    label = short[0] if short else m
                    remaining = int(until - time.time())
                    in_cd.append(f"{label}({remaining}s)")

        cd_str = f" | cooldown: {', '.join(in_cd)}" if in_cd else ""
        log.info(f"Progress: {progress}/{len(work_items)} | ok={stats['ok']} fail={stats['fail']} skip={stats['skipped']} cd={stats['cooled']} | {rate:.1f} files/min{cd_str}")

        # if all models in cooldown, sleep until earliest expires
        if in_cd and len(in_cd) == len(model_counts):
            with cd_lock:
                earliest = min(until for until, _ in cooldowns.values() if until > time.time())
            wait = earliest - time.time() + 5
            if wait > 0:
                log.info(f"All models in cooldown — sleeping {int(wait)}s")
                time.sleep(min(wait, 300))

        if elapsed > 1800:
            log.info("30 min elapsed — triggering re-scan")
            break

    stop_event.set()
    for t in threads:
        t.join(timeout=5)

    log.info(f"Cycle done: ok={stats['ok']} fail={stats['fail']} skip={stats['skipped']} cd={stats['cooled']}")
    return stats["ok"]


def main():
    parser = argparse.ArgumentParser(description="Gap filler with per-model cooldown")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--models", type=str, default=None, help="Comma-separated model keys")
    parser.add_argument("--n_samples", type=int, default=N_SAMPLES)
    args = parser.parse_args()

    problems = load_math500()
    log.info(f"Loaded {len(problems)} problems, {len(NIM_API_KEYS)} keys, {args.workers} workers")

    if args.models:
        keys = [k.strip() for k in args.models.split(",")]
        target_models = {k: MODELS[k] for k in keys}
    else:
        active = ["Qwen397B", "MiniMax"]
        target_models = {k: MODELS[k] for k in active if k in MODELS}

    log.info(f"Targeting {len(target_models)} models: {', '.join(target_models.keys())}")

    cooldowns = {}
    cd_lock = threading.Lock()

    cycle_num = 0
    while True:
        cycle_num += 1
        log.info(f"\n{'='*50}")
        log.info(f"CYCLE {cycle_num}")
        stop_event = threading.Event()
        filled = run_cycle(problems, target_models, args.n_samples, args.workers, stop_event, cooldowns, cd_lock)
        if filled == 0:
            missing = scan_missing(problems, target_models, args.n_samples)
            if not missing:
                log.info("ALL GAPS FILLED — exiting")
                break
            log.info(f"Still {len(missing)} gaps, starting next cycle")
        time.sleep(5)

    log.info("Gap filler complete")


if __name__ == "__main__":
    main()
