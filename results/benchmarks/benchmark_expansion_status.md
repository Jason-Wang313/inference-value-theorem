# Cross-Benchmark Expansion Status

## Current Status
- Runner: `experiments/16_cross_benchmark.py`.
- Benchmarks: GPQA Diamond, IFEval, LiveBench selected subsets, LiveCodeBench.
- Model panel: all stable high-volume live NIM endpoints currently enabled in `config.py`.
- Scale completed: 16 models, 20 tasks/benchmark, 48 samples/model/task.
- Coverage: 320/320 measured records and 100% grading coverage for every benchmark.
- Exact-law MAE: 0.0 for every benchmark.
- Final summary: `results/benchmarks/cross_benchmark_summary.csv`.

## Final Results
| benchmark | records | nondegenerate | mean p | N=1 | N=48 |
| --- | ---: | ---: | ---: | ---: | ---: |
| GPQA Diamond | 320/320 | 251 | 0.5393 | 0.5393 | 0.5890 |
| IFEval | 320/320 | 164 | 0.7770 | 0.7770 | 0.7962 |
| LiveBench selected | 320/320 | 141 | 0.2184 | 0.2184 | 0.2546 |
| LiveCodeBench | 320/320 | 262 | 0.4836 | 0.4836 | 0.5246 |

## Gate Decision
- Evidence gate passes for dataset loading, caching, grading, theorem analysis, and non-degenerate coverage.
- The 16-model stable-endpoint panel is complete and paper-usable as cross-domain generalization evidence.
- LiveCodeBench is scoped to public tests.
- Unavailable/deprecated/rate-limited provider endpoints are excluded from the live panel and documented in `config.py` plus `collect_failures.jsonl`; do not claim those endpoints were benchmarked live.

## Reproduction Commands
```bash
python experiments/16_cross_benchmark.py --benchmarks gpqa_diamond ifeval livebench_selected livecodebench --model-set all --n-tasks 20 --n-samples 48 --batch-size 1 --workers 192 --collect --allow-unsafe-code-exec --code-test-scope public
```

```bash
python experiments/16_cross_benchmark.py --benchmarks gpqa_diamond ifeval livebench_selected livecodebench --model-set all --n-tasks 20 --n-samples 48 --measure --analyze --allow-unsafe-code-exec --code-test-scope public --measure-workers 12
```
