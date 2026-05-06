"""
Wait for collection to finish, then run the full analysis pipeline:
  02_measure → 03_predict → 04_validate → 05_phase → 06_diagnostic → 07_figures

Polls for the collection process by checking if any python run_light_collection.py
is still alive. Once it's gone, runs each step sequentially.

Usage:
    python run_pipeline_after_collection.py          # wait + run all
    python run_pipeline_after_collection.py --now    # skip wait, run immediately
"""

import subprocess
import sys
import time
import argparse
import logging
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
LOG_DIR = PROJECT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "pipeline.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

STEPS = [
    ("02_measure",    [sys.executable, "experiments/02_measure.py"]),
    ("03_predict",    [sys.executable, "experiments/03_predict.py"]),
    ("04_validate_3B",       [sys.executable, "experiments/04_validate.py", "--model", "3B"]),
    ("04_validate_8B",       [sys.executable, "experiments/04_validate.py", "--model", "8B"]),
    ("04_validate_Super49B", [sys.executable, "experiments/04_validate.py", "--model", "Super49B"]),
    ("04_validate_Ultra253B",[sys.executable, "experiments/04_validate.py", "--model", "Ultra253B"]),
    ("04_validate_70B",      [sys.executable, "experiments/04_validate.py", "--model", "70B"]),
    ("04_validate_Maverick17B",      [sys.executable, "experiments/04_validate.py", "--model", "Maverick17B"]),
    ("04_validate_Qwen122B",         [sys.executable, "experiments/04_validate.py", "--model", "Qwen122B"]),
    ("04_validate_Qwen397B",         [sys.executable, "experiments/04_validate.py", "--model", "Qwen397B"]),
    ("04_validate_QwenNext80B",      [sys.executable, "experiments/04_validate.py", "--model", "QwenNext80B"]),
    ("04_validate_Gemma3n2B",        [sys.executable, "experiments/04_validate.py", "--model", "Gemma3n2B"]),
    ("04_validate_Gemma3n4B",        [sys.executable, "experiments/04_validate.py", "--model", "Gemma3n4B"]),
    ("04_validate_MistralLarge675B", [sys.executable, "experiments/04_validate.py", "--model", "MistralLarge675B"]),
    ("04_validate_MistralNemotron",  [sys.executable, "experiments/04_validate.py", "--model", "MistralNemotron"]),
    ("04_validate_MistralSmall119B", [sys.executable, "experiments/04_validate.py", "--model", "MistralSmall119B"]),
    ("04_validate_Mixtral8x7B",      [sys.executable, "experiments/04_validate.py", "--model", "Mixtral8x7B"]),
    ("04_validate_Mixtral8x22B",     [sys.executable, "experiments/04_validate.py", "--model", "Mixtral8x22B"]),
    ("04_validate_Ministral14B",     [sys.executable, "experiments/04_validate.py", "--model", "Ministral14B"]),
    ("04_validate_Super120B",        [sys.executable, "experiments/04_validate.py", "--model", "Super120B"]),
    ("04_validate_Dracarys70B",      [sys.executable, "experiments/04_validate.py", "--model", "Dracarys70B"]),
    ("04_validate_Stockmark100B",    [sys.executable, "experiments/04_validate.py", "--model", "Stockmark100B"]),
    ("04_validate_GLM5",             [sys.executable, "experiments/04_validate.py", "--model", "GLM5"]),
    ("04_validate_MiniMax",          [sys.executable, "experiments/04_validate.py", "--model", "MiniMax"]),
    ("04_validate_KimiK2",           [sys.executable, "experiments/04_validate.py", "--model", "KimiK2"]),
    ("05_phase",      [sys.executable, "experiments/05_phase.py"]),
    ("06_diagnostic",  [sys.executable, "experiments/06_diagnostic.py"]),
    ("07_figures",     [sys.executable, "experiments/07_figures.py"]),
]


def is_collection_running():
    try:
        r = subprocess.run(
            ["tasklist"],
            capture_output=True, text=True, timeout=10,
        )
        lines = r.stdout.lower().splitlines()
        python_pids = [l for l in lines if "python" in l]
        if len(python_pids) <= 1:
            return False
        r2 = subprocess.run(
            ["wmic", "process", "where", "name='python.exe'", "get", "CommandLine,ProcessId"],
            capture_output=True, text=True, timeout=10,
        )
        return "run_light_collection" in r2.stdout
    except Exception:
        return False


def wait_for_collection():
    log.info("Waiting for run_light_collection.py to finish...")
    while is_collection_running():
        time.sleep(60)
    log.info("Collection process finished (or not found). Starting pipeline.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--now", action="store_true", help="Skip waiting for collection")
    args = parser.parse_args()

    if not args.now:
        wait_for_collection()

    for step_name, cmd in STEPS:
        log.info(f"{'='*60}")
        log.info(f"RUNNING: {step_name}")
        start = time.time()
        try:
            result = subprocess.run(cmd, cwd=str(PROJECT), timeout=14400)
            elapsed = time.time() - start
            if result.returncode == 0:
                log.info(f"OK: {step_name} ({elapsed/60:.1f}min)")
            else:
                log.warning(f"FAILED: {step_name} exit={result.returncode} ({elapsed/60:.1f}min)")
        except subprocess.TimeoutExpired:
            log.error(f"TIMEOUT: {step_name} (>240min)")
        except Exception as e:
            log.error(f"ERROR: {step_name}: {e}")

    log.info("PIPELINE COMPLETE")


if __name__ == "__main__":
    main()
