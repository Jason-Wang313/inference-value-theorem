# Ideal Claim Gate Report

- Created UTC: `2026-05-29T12:59:02+00:00`
- Smoke mode: `True`
- All claim gates passed: `False`
- Summary: `{'PASS': 6, 'WARN': 1, 'INFO': 1, 'MISSING': 5}`
- Claim summary: `{'PASS': 6, 'MISSING': 5}`
- Diagnostic summary: `{'WARN': 1, 'INFO': 1}`

## Integrity Policy

- Do not fabricate responses, scores, labels, or confidence intervals.
- Do not change ground truth or omit failing slices to pass a gate.
- Tune estimators and allocation policies only on pilot/calibration data.
- Treat independent evaluation gates as final evidence for paper claims.

## Gates

| Gate | Category | Claim Blocking | Status | Value | Threshold | Evidence | Required Action |
| --- | --- | ---: | --- | ---: | --- | --- | --- |
| core_exact_law_mae | claim | True | PASS | 0.0006223688355332208 | <= 0.001 | `C:\Users\wangz\Downloads\inference-value-theorem-github\results\du_aligned\du_experiment_results.json` | If this fails, inspect tie handling, score alignment, and measurement/simulation parity; do not change labels or omit failures. |
| moment_hierarchy_exactness | claim | True | PASS | 7.44110972799246e-18 | <= 1e-9 | `C:\Users\wangz\Downloads\inference-value-theorem-github\results\du_aligned\du_experiment_results.json` | If this fails, fix the moment/rank formula before scaling any new experiments. |
| expanded_pilot_proxy_K128_N8 | diagnostic | False | WARN | 0.04014398587894077 | <= 0.03 | `C:\Users\wangz\Downloads\inference-value-theorem-github\results\maxed_out\expanded_pilot_proxy\expanded_proxy_locked_summary.csv` | If this diagnostic proxy fails, do not overclaim tiny-pilot sufficiency; the next legitimate step is the full 4096-sample locked split, with estimator selection on calibration data only and final reporting on the independent evaluation pool. |
| expanded_pilot_proxy_K256_N8 | diagnostic | False | INFO | None | <= 0.02 | `C:\Users\wangz\Downloads\inference-value-theorem-github\results\maxed_out\expanded_pilot_proxy\expanded_proxy_locked_summary.csv` | This proxy pool cannot certify the gate at this K; run the 4096-sample locked split and select estimators only on calibration data. |
| maxed_heldout_K128_N8 | claim | True | MISSING | None | <= 0.03 | `C:\Users\wangz\Downloads\inference-value-theorem-github\results\maxed_out\heldout_forecasting\smoke_tables\heldout_locked_estimator_summary.csv` | If this fails, improve estimators on pilot/calibration only, add samples, or scope the claim to the observed sample-complexity curve. |
| maxed_heldout_K256_N8 | claim | True | MISSING | None | <= 0.02 | `C:\Users\wangz\Downloads\inference-value-theorem-github\results\maxed_out\heldout_forecasting\smoke_tables\heldout_locked_estimator_summary.csv` | If this fails, improve estimators on pilot/calibration only, add samples, or scope the claim to the observed sample-complexity curve. |
| maxed_heldout_K512_N8 | claim | True | MISSING | None | <= 0.015 | `C:\Users\wangz\Downloads\inference-value-theorem-github\results\maxed_out\heldout_forecasting\smoke_tables\heldout_locked_estimator_summary.csv` | If this fails, improve estimators on pilot/calibration only, add samples, or scope the claim to the observed sample-complexity curve. |
| live_judge_delta_over_logprob | claim | True | PASS | 0.06444006895093203 | >= 0.03 | `C:\Users\wangz\Downloads\inference-value-theorem-github\results\du_aligned\live_judge_score_summary.json` | If this fails, add independent judges, calibrate judge ensembles on calibration data, and evaluate once on held-out pairs. |
| maxed_live_judge_coverage | claim | True | PASS | 135 | >= 6 | `C:\Users\wangz\Downloads\inference-value-theorem-github\results\du_aligned\live_judge_score_summary.json` | Run the live judge command queue to completion; keep existing 135-pair result scoped as a subset until then. |
| cross_benchmark_exact_law | claim | True | PASS | {"mae": [0.0, 0.0, 0.0, 0.0], "coverage": [1.0, 1.0, 1.0, 1.0]} | MAE <= 0.005 and coverage=1.0 | `C:\Users\wangz\Downloads\inference-value-theorem-github\results\benchmarks\cross_benchmark_summary.json` | If this fails, fix grader/score alignment before adding more benchmarks. |
| cross_benchmark_family_count | claim | True | MISSING | {"completed": 4, "missing": ["math500", "humaneval_mbpp"]} | >= 6 | `C:\Users\wangz\Downloads\inference-value-theorem-github\results\benchmarks\cross_benchmark_summary.json` | Run and measure the two missing families, MATH500 and MBPP-backed humaneval_mbpp, before claiming six-family generality. |
| maxed_cross_benchmark_scale | claim | True | MISSING | {"required": {"benchmarks": ["math500", "gpqa_diamond", "ifeval", "livebench_selected", "livecodebench", "humaneval_mbpp"], "task_limit": 3, "samples_per_task": 4, "model_count": 2}, "complete": ["gpqa_diamond", "ifeval", "livebench_selected", "livecodebench"], "incomplete": {"math500": {"issues": ["missing_family"]}, "humaneval_mbpp": {"issues": ["missing_family"]}}} | 6 families at manifest task limit, sample depth, model count, and 100% measured coverage | `C:\Users\wangz\Downloads\inference-value-theorem-github\results\benchmarks\cross_benchmark_summary.json` | Run the full cross-benchmark command queue with 500 requested tasks, 128 samples/task, and the full live model panel; then regenerate analysis. |
| adaptive_allocation_delta | claim | True | PASS | 0.012464669900157044 | > 0 | `C:\Users\wangz\Downloads\inference-value-theorem-github\results\du_aligned\adaptive_allocation_summary.json` | If this fails, tune allocation policies on calibration seeds only and preserve oracle/uniform baselines. |

## Claim Rule

Only claim-blocking gates marked `PASS` can support maximum-strength paper wording. Any claim-blocking `FAIL` or `MISSING` gate must either be fixed with more evidence or scoped as a limitation. Diagnostic rows are pre-flight checks: disclose them, but do not treat them as final independent-evaluation evidence.
