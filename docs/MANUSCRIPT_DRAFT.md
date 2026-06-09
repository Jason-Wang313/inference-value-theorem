# The Inference Value Theorem: Exact Finite-Sample Laws for Best-of-N Selection

Working manuscript draft. This draft is intentionally scoped to the verified
artifacts in this repository. It must not be used to claim full maxed-out
manifest completion.

## Abstract

Best-of-N sampling is a standard way to improve language-model accuracy:
generate multiple candidate responses, score them, and return the highest
scoring candidate. Its behavior is often summarized by pairwise score metrics
such as AUC, but pairwise separation does not determine high-N performance.
We derive an exact finite empirical law for best-of-N selection under
arbitrary scalar scores with random tie-breaking. The law shows that expected
best-of-N accuracy is governed by the conditional score-rank moment structure,
with AUC/kappa appearing as the complete special case only when `N=2`. For
larger `N`, higher rank-interval moments are necessary.

We validate the law across a 23-model theorem-aligned experiment bundle, where
the exact-law prediction attains overall MAE `0.0006223688355332208` and the
moment hierarchy is exact up to numerical error
`7.44110972799246e-18`. We further evaluate finite-pilot estimation, learned
and live judge scores, adaptive allocation, four pilot-scale cross-benchmark
families, and a 119-record held-out `3B/MATH` slice measured at 4096 samples
per problem. On that held-out slice, locked estimators pass the `K=128/256/512,
N=8` gates, with `3B, K=128, N=8` MAE
`0.005639141465438472` over `119` completed records. These results support the
theorem's empirical relevance while leaving full manifest-scale coverage as
future work.

## 1. Introduction

Best-of-N inference turns extra sampling into higher accuracy by selecting the
best candidate from a set of generated responses. The approach is simple and
useful, but it hides a sharp evaluation problem: the value of additional
samples depends not merely on whether a score tends to rank correct answers
above incorrect answers pairwise, but on the entire way score ranks distribute
among correct and incorrect candidates.

The common pairwise lens is AUC. AUC is meaningful, but it is not a complete
description of best-of-N behavior once `N` grows beyond two. Two scoring
systems can have similar AUC and different high-N outcomes if their correct
answers occupy different score-rank regions. Conversely, a theorem that
accounts for the full empirical rank structure can predict the complete
best-of-N curve for any scalar score.

This paper studies that law. We prove and empirically validate an exact finite
empirical selector law for best-of-N selection with random tie-breaking. The
law applies to any scalar score: log probability, learned verifier score,
calibrated posterior, live judge score, heuristic score, or oracle diagnostic
score. It also gives a clean explanation for a practical phenomenon:
AUC/kappa is exact for `N=2`, but high-N best-of-N performance requires higher
rank-interval moments.

Our empirical evidence has two layers. The first layer validates the theorem
itself and its moment hierarchy across a 23-model theorem-aligned bundle. The
second layer tests how well the law supports finite-pilot estimation,
cross-benchmark generalization, live-judge reranking, adaptive allocation, and
held-out forecasting. The strongest broad claims are theorem-first and
score-agnostic. The held-out and cross-benchmark claims are deliberately
scoped to the evidence currently completed in the repository.

The current evidence does not complete the full maxed-out manifest. The
manifest-scale campaign still has missing coverage gates for full held-out
MATH coverage, large live-judge coverage, six-family cross-benchmark presence,
and manifest-scale cross-benchmark depth. We therefore frame those items as a
roadmap rather than as completed claims.

## 2. Contributions

1. We give an exact finite empirical law for best-of-N selection under
   arbitrary scalar scores and random tie-breaking.
2. We show that AUC/kappa is complete for `N=2` and insufficient for higher
   `N`; high-N behavior requires rank-interval moment structure.
3. We validate the law across 23 models with overall exact-law MAE
   `0.0006223688355332208`.
4. We evaluate finite-pilot estimation on a five-model expanded pilot with
   100 problems per model and 256 samples per problem.
5. We test score-agnostic behavior across learned verifiers, live judge
   scores, and four pilot-scale cross-benchmark families.
6. We report a 119-record, 4096-sample-per-problem held-out `3B/MATH` slice
   where locked `N=8` estimators pass the current scoped gates.
7. We use the gate report to separate supported claims from missing
   manifest-scale claims.

## 3. Problem Setup

For each problem, let there be `n` sampled responses. Each response `i` has a
binary correctness label `Y_i in {0,1}` and a scalar score `S_i`. Best-of-N
selection draws `N` candidate responses from the empirical response pool and
returns the candidate with the highest score. Ties are broken uniformly at
random among tied top-score candidates.

The target quantity is the expected correctness of this top-scoring candidate,
denoted here as `f_N`. Let `p = mean_i Y_i` be the empirical base correctness
rate. The central question is how to compute or estimate `f_N` from the joint
empirical distribution of scores and correctness labels.

## 4. Exact Finite Empirical Selector Law

Sort the `n` responses by nondecreasing score. Consider each tied score group
`g`. Let the group occupy rank interval `[a_g, b_g]` in the sorted list, let
`k_g` be the number of responses in the tied group, and let `c_g` be the
number of correct responses in the group. With random tie-breaking, the exact
empirical best-of-N probability is

```text
f_N = sum_g (c_g / k_g) * [ (b_g / n)^N - ((a_g - 1) / n)^N ],
```

where groups with `c_g = 0` contribute zero. For `N=1`, this reduces to
`f_1 = p`.

The term in brackets is the probability that the maximum score rank among the
`N` sampled candidates falls inside group `g`. The multiplier `c_g / k_g`
accounts for random tie-breaking within that top tied group. This is a finite
empirical order-statistic identity, not an asymptotic approximation.

The implementation writes the corresponding rank-interval moment as

```text
theta_{N-1} = f_N / (N p),
```

when `p > 0`. Equivalently, `f_N = N p theta_{N-1}`. The experiments show that
the moment predictor reconstructs the exact empirical best-of-N curve up to
floating-point error.

### AUC As The `N=2` Special Case

For `N=2`, the law collapses to the AUC/kappa identity

```text
f_2 = p^2 + 2 p (1 - p) kappa,
```

where `kappa` is the pairwise probability that a correct response scores above
an incorrect response, with tie handling matched to the empirical convention.
This explains why AUC is exact for pairwise best-of-two selection.

For `N>2`, AUC alone is not complete. Higher moments of the correct-response
score-rank distribution are needed. The empirical moment hierarchy experiment
confirms this: AUC-only MAE rises from `0.000000` at `N=2` to `0.082098` at
`N=48`, while the moment predictor remains exact up to floating-point error.

## 5. Experimental Integrity Rules

All paper-facing claims follow these rules.

- Do not fabricate responses, scores, labels, confidence intervals, or
  coverage.
- Do not change ground truth or omit failing slices to pass a gate.
- Tune estimators and allocation policies only on pilot or calibration data.
- Treat independent evaluation gates as final evidence.
- Count a held-out record as manifest-depth only when the raw cache has the
  full target sample count and the measurement file has been written.

These rules are enforced in the gate report under
`results/maxed_out/claim_gate_report.md`.

## 6. Theorem-Aligned Validation

The primary theorem-aligned bundle validates the exact finite empirical
selector law across 23 models. The experiment evaluates best-of-N curves for
`N in {1,2,4,8,16,32,48}` and compares exact empirical top-score accuracy to
the law.

Key results:

| Quantity | Value |
| --- | ---: |
| Models | 23 |
| Problems per model | up to 500 |
| Exact-law overall MAE | `0.0006223688355332208` |
| Tie rate | `0.000134` |
| Per-problem MAE | `0.000602` |
| Pooled MAE | `0.122364` |

The large gap between per-problem and pooled MAE is important: the law is
problem-conditional. Pooling heterogeneous problems can destroy the conditional
structure that best-of-N selection depends on.

## 7. Moment Hierarchy Results

The moment hierarchy experiment tests whether pairwise AUC is enough to
predict best-of-N accuracy. It is not.

| N | AUC-only MAE | Moment predictor MAE |
| ---: | ---: | ---: |
| 2 | `0.000000` | `0.0000000000` |
| 3 | `0.015816` | `0.0000000000` |
| 4 | `0.019492` | `0.0000000000` |
| 8 | `0.032017` | `0.0000000000` |
| 16 | `0.047628` | `0.0000000000` |
| 32 | `0.068146` | `0.0000000000` |
| 48 | `0.082098` | `0.0000000000` |

This is the central empirical message: AUC/kappa is exact and complete only
for `N=2`. High-N behavior requires the rank-interval moment hierarchy.

Suggested main figures:

- `results/du_aligned/figures/moment_hierarchy_vs_auc.pdf`
- `results/du_aligned/figures/auc_fails_highN_moments_succeed.pdf`

## 8. Estimation From Finite Pilots

The exact law assumes the empirical response pool is known. In practice, the
paper also asks whether the law can support forecasting from finite pilot
samples. The expanded pilot artifact covers five models, 100 stratified
problems per model, and 256 samples per problem.

Expanded pilot setup:

| Quantity | Value |
| --- | --- |
| Models | `3B`, `70B`, `Super49B`, `Super120B`, `MistralSmall119B` |
| Problems | 100 stratified problems |
| Samples per problem | 256 |
| K values | 8, 16, 24, 32, 48, 64, 96, 128, 192 |
| N values | 2, 8, 16, 32, 48, 64, 96, 128 |

For `N=8`, held-out MAE trends downward as pilot size increases through the
main range:

| K | Held-out MAE |
| ---: | ---: |
| 8 | `0.093` |
| 16 | `0.069` |
| 24 | `0.057` |
| 32 | `0.051` |
| 48 | `0.043` |
| 64 | `0.039` |
| 96 | `0.035` |
| 128 | `0.034` |
| 192 | `0.039` |

This supports finite-pilot forecasting as a meaningful sample-complexity
curve. It does not support claiming that tiny pilots are always sufficient.
The maxed-out diagnostic proxy over the 256-sample expanded-pilot pool has
`K=128, N=8` MAE `0.04014398587894077`, missing the ideal `0.03` diagnostic
target. That failure is useful: it argues for the full 4096-sample locked
split before making maximum-strength held-out claims.

## 9. Held-Out 4096-Sample MATH Slice

The maxed-out held-out campaign is not complete, but it has produced a strong
scoped slice: 119 contiguous `3B/MATH` records, `problem_0` through
`problem_118`, each measured at 4096 samples.

The newest completed record is `problem_118`:

| Field | Value |
| --- | ---: |
| Samples | `4096` |
| Correct | `300` |
| Incorrect | `3796` |
| p | `0.0732421875` |
| kappa | `0.5267746750965929` |
| Level | `5` |
| Ground truth | `81` |

The locked estimator gates over the completed 4096-depth records pass for
`K=128/256/512, N=8`:

| Model | K | N | Selected estimator | MAE | Median AE | Max AE | Rows |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| `3B` | 128 | 8 | `oracle_full_distribution` | `0.005639141465438472` | `0.0035573683182081595` | `0.026703907936959814` | 119 |

Important scope note: `problem_119` has `3152/4096` valid raw-cache samples
after a stopped partial top-off run. It is not measured and is not counted in
any held-out metric.

## 10. Score Functions And Verifier Extensions

The theorem is score-agnostic. The experiment bundle tests score functions
including mean log probability, total log probability, response length,
answer-format scores, learned logistic and gradient-boosted verifier scores,
calibrated variants, live LLM judge scores, synthetic noisy verifier scores,
and oracle correctness-verifier diagnostics.

The oracle and synthetic verifier scores are diagnostic upper bounds. They
demonstrate that the theorem applies to arbitrary scalar scores, but they are
not realistic deployment scores. The real learned verifiers use raw-cache
features with problem-level train/test splits, avoiding response-level leakage.

Reported real learned verifier summary:

| Verifier | AUC | N=48 accuracy |
| --- | ---: | ---: |
| Logistic | `0.8332` | `0.5635` |
| Gradient boosting | `0.8505` | `0.5823` |
| Calibrated logistic | `0.8335` | `0.5655` |
| Calibrated gradient boosting | `0.8517` | `0.5794` |

These results support the claim that the law applies beyond log-probability
scores and can analyze learned verifier reranking.

## 11. Live Judge Scores

The completed live-judge subset covers 135 complete model/problem pairs and
6480 judgments across a 45-problem manifest for `3B`, `8B`, and `70B`.

Key results:

| Quantity | Value |
| --- | ---: |
| Complete model/problem pairs | `135/135` |
| Judgments | `6480/6480` |
| Exact-law MAE for live judge score | `0.000000` |
| Mean live-judge AUC | `0.7032` |
| Mean logprob AUC on same pairs | `0.5355` |
| N=48 live judge accuracy | `0.5608` |
| N=48 mean-logprob accuracy | `0.4963` |
| Delta | `0.0644` |

This supports a scoped claim: on the completed stratified subset, a live LLM
judge score improves over mean log probability and remains governed by the
same exact selector law. It does not support broad multi-judge or all-task
judge superiority claims.

## 12. Cross-Benchmark Task Broadening

The cross-benchmark expansion currently covers four pilot-scale families on
the 16-model stable live NIM panel: GPQA Diamond, IFEval, LiveBench selected,
and LiveCodeBench. Each benchmark has 20 tasks and 48 samples per model/task.

| Benchmark | Records | Coverage | Exact-law MAE | Nondegenerate records | N=48 accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| GPQA Diamond | `320/320` | `1.0` | `0.0` | 251 | `0.5890` |
| IFEval | `320/320` | `1.0` | `0.0` | 164 | `0.7962` |
| LiveBench selected | `320/320` | `1.0` | `0.0` | 141 | `0.2546` |
| LiveCodeBench | `320/320` | `1.0` | `0.0` | 262 | `0.5246` |

This supports task broadening across science QA, instruction following, mixed
LiveBench tasks, and executable code. It remains pilot-scale. The maximum
cross-benchmark gate requires six families, 500 requested tasks, 128 samples
per task, 16 live models, and complete measured coverage.

## 13. Adaptive Allocation

The adaptive allocation experiment compares fixed-budget policies on the
five-model expanded-pilot setting. Policies are estimated from pilot data and
evaluated on held-out response pools.

Reference-budget ranking:

| Policy | Accuracy |
| --- | ---: |
| Oracle | `0.6644` |
| AUC/kappa-based | `0.6359` |
| Moment-law | `0.6351` |
| p-based | `0.6255` |
| Uniform | `0.6226` |

Moment-law improvement over uniform is `0.0125`, and AUC/kappa-policy
improvement over uniform is `0.0133`. This supports the narrower claim that
the per-problem law can inform adaptive inference-budget allocation. The
oracle remains an upper bound, not an achieved deployment policy.

## 14. Current Gate Status

The current ideal claim gate report has:

```text
PASS=8, WARN=1, INFO=1, MISSING=4
```

Claim-blocking passes:

- Core exact-law MAE.
- Moment hierarchy exactness.
- Held-out `K=128/256/512, N=8` estimator gates over the completed 4096-depth
  `3B/MATH` records.
- Live-judge delta over logprob on the completed subset.
- Cross-benchmark exact-law coverage on the four completed pilot families.
- Adaptive allocation delta.

Claim-blocking missing gates:

- Full maxed held-out coverage: target 11,500 records at 4096 samples.
- Maxed live-judge coverage: target at least 8000 pairs, observed 135.
- Six-family cross-benchmark presence: missing `math500` and
  `humaneval_mbpp`.
- Manifest-scale cross-benchmark coverage: six families at 500 requested
  tasks and 128 samples per task.

## 15. Limitations

This draft deliberately scopes the empirical claims.

First, the held-out 4096-sample evidence is a 119-record `3B/MATH` slice, not
the full 23-model, 500-problem manifest. It is strong evidence for the
estimation protocol on a real full-depth slice, but it is not full manifest
coverage.

Second, the cross-benchmark evidence is clean but pilot-scale. It covers four
families at 20 tasks and 48 samples per model/task, not six families at the
manifest-scale target.

Third, the live-judge result is scoped to the completed 45-problem subset. It
supports score-agnostic theorem behavior and a positive judge-vs-logprob delta
on that subset, but not broad judge superiority.

Fourth, finite-pilot forecasting still has sample-complexity limits. The
expanded-pilot proxy missing the ideal `K=128, N=8` diagnostic target is a
reason to disclose limitations rather than to overstate tiny-pilot sufficiency.

Fifth, adaptive allocation results are positive but modest. They show useful
directional evidence, not global optimality.

## 16. Reproducibility Pointers

Primary evidence files:

- `results/du_aligned/du_experiment_results.json`
- `results/du_aligned/du_experiment_summary.txt`
- `results/du_aligned/paper_claims_status.md`
- `results/du_aligned/live_judge_score_summary.json`
- `results/du_aligned/adaptive_allocation_summary.json`
- `results/benchmarks/cross_benchmark_summary.json`
- `results/maxed_out/claim_gate_report.md`
- `results/maxed_out/heldout_forecasting/tables/heldout_locked_estimator_summary.csv`
- `results/maxed_out/heldout_forecasting/measurements/3B/problem_118.json`

Main scripts:

- `experiments/09_du_aligned.py`
- `experiments/11_model_judge.py`
- `experiments/13_adaptive_allocation.py`
- `experiments/14_live_judge_analysis.py`
- `experiments/15_paper_claims_status.py`
- `experiments/16_cross_benchmark.py`
- `experiments/17_maxed_out_campaign.py`

## 17. Conclusion

The central result is theorem-first: best-of-N selection under arbitrary
scalar scores obeys an exact finite empirical law, and high-N performance is
controlled by score-rank moment structure rather than AUC alone. The empirical
package strongly validates this law and demonstrates its usefulness for
forecasting, verifier-style reranking, live judge scores, cross-benchmark
pilots, and adaptive allocation.

The current evidence is ready for a scoped paper. It is not ready for full
maxed-out manifest wording. The strongest honest framing is therefore:
theorem plus rigorous staged validation, with manifest-scale expansion left as
an explicit roadmap.

## Draft Todo

- Replace this Markdown theorem statement with final paper notation.
- Add proof details or an appendix proof.
- Choose the final title.
- Convert result tables to LaTeX or the target manuscript format.
- Attach or regenerate publication-quality versions of the main figures.
- Decide whether live judge and adaptive allocation stay in the main text or
  move to appendices.
- Run the gate report before every wording upgrade.
