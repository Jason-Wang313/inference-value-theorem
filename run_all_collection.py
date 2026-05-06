"""Run collection for all 4 models sequentially."""
import subprocess
import sys
import time

MODELS = ["3B", "8B", "70B", "405B"]

for model in MODELS:
    print(f"\n{'='*60}")
    print(f"STARTING COLLECTION: {model}")
    print(f"{'='*60}\n")
    start = time.time()

    result = subprocess.run(
        [sys.executable, "experiments/01_collect.py", "--model", model],
        cwd=r"C:\Users\wangz\Downloads\inference-value-theorem",
    )

    elapsed = time.time() - start
    print(f"\n{model} finished in {elapsed/60:.1f} minutes (exit code: {result.returncode})")

print("\n\nALL COLLECTIONS COMPLETE")
