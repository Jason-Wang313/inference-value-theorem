# Paper Claims Status

## Core theorem evidence
- Existing Du-aligned bundle validates the exact finite empirical selector law across 23 models with overall MAE 0.000622.
- Existing moment-hierarchy results show AUC/kappa is exact for N=2 and insufficient for high N; moment predictor is exact up to floating-point error.

## Adaptive allocation
- Status: completed.
- Models: ['3B', '70B', 'MistralSmall119B', 'Super120B', 'Super49B'].
- Policies: ['auc_kappa_based', 'moment_law', 'oracle', 'p_based', 'uniform'].
- Reference-budget ranking: oracle=0.6644, auc_kappa_based=0.6359, moment_law=0.6351, p_based=0.6255, uniform=0.6226.
- Moment-law improvement over uniform at reference budget: 0.0125.
- AUC/kappa-policy improvement over uniform at reference budget: 0.0133.
- Interpretation: adaptive allocation is now evidenced, but the current finite-pilot moment policy is modest; oracle remains the upper bound.

## Expanded pilot sample complexity
- Status: completed artifact with max measured samples/problem = 256.
- Models: ['3B', '70B', 'Super49B', 'Super120B', 'MistralSmall119B'].
- K values tested: [8, 16, 24, 32, 48, 64, 96, 128, 192].
- N values tested: [2, 8, 16, 32, 48, 64, 96, 128].
- N=8 held-out MAE trend: K=8: 0.093, K=16: 0.069, K=24: 0.057, K=32: 0.051, K=48: 0.043, K=64: 0.039, K=96: 0.035, K=128: 0.034, K=192: 0.039.
- Cache coverage toward current target: 3B: 100/100 records at n=256; 70B: 100/100 records at n=256; MistralSmall119B: 100/100 records at n=256; Super120B: 100/100 records at n=256; Super49B: 100/100 records at n=256.
- Interpretation: K=128/192 curves are now evidenced for all five expanded-pilot models; the five-model curve is complete through K=192.

## Live LLM judge score
- Status: completed.
- Complete model/problem pairs analyzed: 135.
- Current manifest coverage: 135/135 complete model/problem pairs and 6480/6480 judgments for the 45-problem manifest.
- Best completed manifest coverage: 135/135 complete model/problem pairs and 6480/6480 judgments for the 45-problem manifest.
- Models: ['3B', '70B', '8B'].
- N values: [1, 2, 4, 8, 16, 32, 48].
- Exact-law MAE for live judge score: 0.000000.
- Mean live-judge AUC: 0.7032; mean logprob AUC on same pairs: 0.5355.
- N=48 live judge accuracy: 0.5608; mean-logprob same-pair accuracy: 0.4963; delta: 0.0644.
- Interpretation: the score-agnostic live-judge claim is now supported on the completed 45-problem stratified subset; keep claims about judge superiority scoped to this subset.

## Cross-benchmark task broadening
- Status: completed.
- Benchmarks: gpqa_diamond, ifeval, livebench_selected, livecodebench.
- Scale: 16-model stable live NIM panel, 20 tasks/benchmark, 48 samples/model/task.
- Coverage: gpqa_diamond: 320/320, ifeval: 320/320, livebench_selected: 320/320, livecodebench: 320/320.
- Non-degenerate records: gpqa_diamond: 251, ifeval: 164, livebench_selected: 141, livecodebench: 262.
- Exact-law MAE by benchmark: gpqa_diamond: 0.000000, ifeval: 0.000000, livebench_selected: 0.000000, livecodebench: 0.000000.
- N=48 best-of-N accuracy by benchmark: gpqa_diamond=0.5890, ifeval=0.7962, livebench_selected=0.2546, livecodebench=0.5246.
- Scope note: excludes provider EOL/degraded/rate-limited endpoints documented in `config.py` and benchmark collection failure logs.
- Interpretation: task broadening is now evidenced on four additional benchmarks spanning science QA, instruction following, mixed LiveBench tasks, and executable code across the stable live endpoint panel; LiveCodeBench is scoped to public tests.

## Remaining ideal-paper gaps
- None for expanded-pilot K=128/192.
- Optional: add a second external verifier if the paper needs broader judge-generalization evidence.
- Optional: add hidden/private LiveCodeBench scoring if the paper needs a stronger code appendix; do not claim unavailable provider endpoints were benchmarked live.
- Do not claim global optimal allocation, broad judge superiority, or tiny-pilot sufficiency.
