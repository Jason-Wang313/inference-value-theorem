# Inference Value Theorem Experiments

This repository contains the empirical artifacts and experiment runners for
the corrected Theorem 1 best-of-N selector audit.

## Du-aligned experiments

The Du-aligned extension is implemented in:

```bash
python experiments/09_du_aligned.py --pilot-seeds 20
```

It writes the complete bundle to `results/du_aligned/`, including the summary
report, JSON results, CSV tables, and PDF figures for:

- moment hierarchy vs AUC,
- pilot sample complexity,
- score-function comparison,
- verifier-style reranking diagnostics,
- posterior monotonicity.

Additional best-paper closure experiments:

```bash
python experiments/13_adaptive_allocation.py --pilot-k 16 --seeds 20
python experiments/12_expanded_pilot.py --collect --measure --target-samples 256 --workers 4
python experiments/11_model_judge.py --live --models 3B 8B 70B --n-problems 50 --max-samples 48 --workers 12 --request-delay 3 --rate-limit-cooldown 90 --max-task-attempts 16 --live-batch-size 8
python experiments/14_live_judge_analysis.py
python experiments/15_paper_claims_status.py
```

Cross-benchmark expansion:

```bash
python experiments/16_cross_benchmark.py --benchmarks gpqa_diamond ifeval livebench_selected livecodebench --model-set all --n-tasks 20 --n-samples 48 --batch-size 1 --workers 192 --collect --allow-unsafe-code-exec --code-test-scope public
```

```bash
python experiments/16_cross_benchmark.py --benchmarks gpqa_diamond ifeval livebench_selected livecodebench --model-set all --n-tasks 20 --n-samples 48 --measure --analyze --allow-unsafe-code-exec --code-test-scope public --measure-workers 12
```

The current cross-benchmark expansion covers GPQA Diamond, IFEval, LiveBench
selected subsets, and LiveCodeBench on the 16-model stable live NIM panel with
20 tasks/benchmark and 48 samples/model/task. All four benchmarks have
320/320 measured model/task records, 100% grading coverage, exact-law MAE
0.0, and non-degenerate counts of 251, 164, 141, and 262 respectively.
Unavailable/deprecated/rate-limited provider endpoints are excluded from this
live panel and are commented in `config.py`.

Current closure artifacts include a 5-model, 100-problem expanded pilot with
256 samples/problem for all five models, enabling five-model K=128/192 pilot
curves; adaptive allocation on the same 5-model set; and a completed live LLM
judge subset with 135/135 model/problem pairs and 6480/6480 judgments.

The project reads NIM keys from `NIM_API_KEYS` when set. If that variable is
absent on this machine, `config.py` falls back to `C:\Users\wangz\MIRROR\.env`
and composes the `NVIDIA_NIM_API_KEY*` entries without printing key material.
When this checkout's `data/raw` directory is empty, `src/nim_client.py` falls
back to the sibling `C:\Users\wangz\Downloads\inference-value-theorem\data\raw`
cache; set `INFERENCE_VALUE_RAW_DIR` to override that behavior.

API keys are intentionally not committed. Set `NIM_API_KEYS` in the
environment before running any collection scripts.
