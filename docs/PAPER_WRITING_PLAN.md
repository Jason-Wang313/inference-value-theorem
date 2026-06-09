# Paper Writing Plan

This plan converts the current verified artifacts into a scoped manuscript.
It should be used for writing now, without waiting for the full maxed-out
manifest, while keeping the maximum-scope claims clearly marked as future or
incomplete evidence.

Current working draft: `docs/MANUSCRIPT_DRAFT.md`.

## Writing Decision

Start writing now.

The current package is rigorous enough for a serious scoped paper: the exact
finite empirical selector law is strongly validated, the theorem-driven
moment hierarchy is numerically exact, the held-out 4096-sample slice is real
and nontrivial, four cross-benchmark pilots are clean, the live-judge subset
improves over mean logprob, and adaptive allocation is positive.

The current package is not enough for full maxed-out generality. The paper
should not claim complete manifest coverage, all-model held-out coverage,
large-scale live-judge coverage, or six-family manifest-scale cross-benchmark
coverage.

## Current Evidence Snapshot

- Core exact law: PASS, overall MAE `0.0006223688355332208`.
- Moment hierarchy exactness: PASS, max error `7.44110972799246e-18`.
- Held-out 4096-depth slice: 119 contiguous `3B/MATH` records,
  `problem_0` through `problem_118`.
- Held-out locked estimator: PASS for `K=128/256/512, N=8`; locked
  `3B, K=128, N=8` MAE is `0.005639141465438472` over `num_rows=119`.
- Latest measured held-out record: `problem_118`, `p=0.0732421875`,
  300 correct out of 4096, `kappa=0.5267746750965929`.
- Partial raw cache: `problem_119` has `3152/4096` valid samples but is not
  measured and must not be counted in held-out metrics.
- Live-judge subset: PASS, `N=48` live-judge accuracy delta over mean logprob
  is `0.06444006895093203` on the completed scoped subset.
- Cross-benchmark exact law: PASS on four pilot families, each with exact-law
  MAE `0.0` and coverage `1.0`.
- Adaptive allocation: PASS, moment-law improvement over uniform is
  `0.012464669900157044`.
- Full gate report: `PASS=8`, `WARN=1`, `INFO=1`, `MISSING=4`.

## Main Claim Shape

Recommended main claim:

> We derive and empirically validate an exact finite-sample law for best-of-N
> selection under arbitrary scalar scores, showing that expected best-of-N
> accuracy is governed by the full conditional score-rank moment structure
> rather than by pairwise AUC alone. Across theorem-aligned synthetic and live
> LLM experiments, the law predicts best-of-N behavior with high numerical
> accuracy, explains when AUC/kappa summaries fail, and supports calibrated
> estimator and allocation strategies.

Recommended empirical scope:

- "23-model theorem-aligned exact-law validation" is supported.
- "Five-model expanded-pilot sample-complexity evidence" is supported.
- "119 full-depth 4096-sample held-out `3B/MATH` records" is supported.
- "Four pilot-scale cross-benchmark families across a 16-model stable live
  panel" is supported.
- "Scoped live-judge improvement over mean logprob" is supported.
- "Positive adaptive-allocation evidence" is supported.

Avoid these claims:

- Full maxed-out manifest completion.
- Universal 23-model held-out MATH generality at 4096 samples.
- Six-family manifest-scale cross-benchmark generality.
- Broad live-judge superiority beyond the completed subset.
- Tiny-pilot sufficiency for the maximum held-out claim.
- Global optimality of the adaptive allocation policy.

## Suggested Title Options

1. `The Inference Value Theorem: Exact Finite-Sample Laws for Best-of-N Selection`
2. `Beyond AUC: Exact Moment Laws for Best-of-N Inference`
3. `Predicting Best-of-N Accuracy from Score-Rank Moment Structure`

The first title is best if the theorem is the center of the paper. The second
is sharper if the intended audience already understands verifier/reranker
evaluation and needs the AUC limitation foregrounded.

## Abstract Skeleton

Best-of-N sampling is widely used to improve LLM accuracy, but its empirical
behavior is often summarized through pairwise score metrics such as AUC. We
show that the expected accuracy of best-of-N selection obeys an exact
finite-sample law determined by the conditional rank/moment structure of the
score distribution. This law explains why AUC is exact for `N=2` but
insufficient for larger `N`, and it yields calibrated estimators for
best-of-N performance from finite pilot samples. We validate the law across a
23-model theorem-aligned bundle, a five-model expanded pilot, 119 full-depth
4096-sample held-out `3B/MATH` records, four pilot-scale cross-benchmark
families, a scoped live-judge reranking subset, and adaptive allocation
experiments. The results support the theorem's empirical relevance while
leaving manifest-scale coverage as future work.

## Section Outline

1. Introduction
   - Motivate best-of-N selection, verifier/reranker scores, and why pairwise
     summaries are not enough.
   - State the central theorem and empirical thesis.
   - Be explicit that the empirical package is scoped, not full manifest
     completion.

2. Problem Setup
   - Define candidate responses, correctness labels, scalar scores, and
     best-of-N selection.
   - Define empirical score-rank quantities and the target expected
     best-of-N accuracy.

3. The Exact Finite-Sample Law
   - State the theorem.
   - Explain the moment/rank hierarchy.
   - Show why AUC/kappa is exact for `N=2` and insufficient for higher `N`.

4. Estimation From Pilot Samples
   - Describe the estimators evaluated in the repo.
   - Separate pilot/calibration/evaluation roles.
   - Emphasize the integrity rule: no tuning on independent evaluation.

5. Theorem-Aligned Validation
   - Use `results/du_aligned/du_experiment_results.json`.
   - Report exact-law MAE `0.0006223688355332208`.
   - Report moment hierarchy max error `7.44110972799246e-18`.

6. Expanded Pilot and Held-Out Forecasting
   - Use five-model expanded-pilot sample-complexity curves.
   - Report the diagnostic proxy honestly: `K=128, N=8` MAE `0.04014` misses
     the ideal `0.03` diagnostic target.
   - Present the 119 full-depth `3B/MATH` held-out records as a strong
     independent slice.
   - Report locked `3B, K=128, N=8` MAE `0.005639141465438472`.

7. Cross-Benchmark Task Broadening
   - Present GPQA Diamond, IFEval, LiveBench selected, and LiveCodeBench.
   - Report 320/320 measured records per family, coverage `1.0`, exact-law
     MAE `0.0`.
   - State that these are pilot-scale: 20 tasks, 48 samples/task, 16 live
     models.

8. Live Judge Scores
   - Present the completed scoped live-judge subset.
   - Report `N=48` live judge accuracy `0.5608`, mean-logprob accuracy
     `0.4963`, delta `0.0644`.
   - Avoid broad multi-judge claims.

9. Adaptive Allocation
   - Compare uniform, p-based, AUC/kappa, moment-law, and oracle policies.
   - Report moment-law improvement over uniform `0.0125`.
   - Describe oracle as an upper bound, not an achieved policy.

10. Limitations and Maxed-Out Roadmap
    - Full manifest still targets 11,500 held-out MATH records at 4096
      samples, 8,000+ live-judge pairs, and six cross-benchmark families at
      manifest scale.
    - Current gate report has four claim-blocking missing gates.
    - `problem_119` partial cache is not counted.

11. Conclusion
    - Reiterate the theorem-first contribution and the scoped empirical
      validation.

## Figures And Tables

Use existing figures/tables where possible:

- Theorem law parity figure from `results/du_aligned/figures/`.
- Moment hierarchy or AUC insufficiency figure from `results/du_aligned/`.
- Expanded pilot sample-complexity table from
  `results/du_aligned/paper_claims_status.md`.
- Held-out locked estimator table from
  `results/maxed_out/heldout_forecasting/tables/heldout_locked_estimator_summary.csv`.
- Cross-benchmark summary from
  `results/benchmarks/cross_benchmark_summary.json`.
- Live judge summary from `results/du_aligned/live_judge_score_summary.json`.
- Adaptive allocation summary from
  `results/du_aligned/adaptive_allocation_summary.json`.
- Gate report table from `results/maxed_out/claim_gate_report.md`, likely in
  an appendix or reproducibility section.

## Claim Boundary Table

| Claim | Status | Paper Placement |
| --- | --- | --- |
| Exact finite empirical selector law | Ready | Main theorem and main experiment |
| AUC/kappa insufficiency for high `N` | Ready | Main theorem/results |
| Five-model expanded pilot curves | Ready | Main empirical section |
| 119 full-depth `3B/MATH` held-out records | Ready, scoped | Main empirical section |
| Four-family pilot cross-benchmark exact-law validation | Ready, scoped | Main empirical section |
| Live judge beats mean logprob on completed subset | Ready, scoped | Main or appendix |
| Adaptive allocation improves over uniform | Ready, modest | Main or appendix |
| Full 23-model held-out MATH manifest | Missing | Future work/limitations |
| Six-family manifest-scale cross-benchmark | Missing | Future work/limitations |
| Broad live-judge generality | Missing | Future work/limitations |

## Immediate Writing Tasks

1. Decide final manuscript location and format.
2. Polish the working draft in `docs/MANUSCRIPT_DRAFT.md`.
3. Replace the Markdown theorem statement with final paper notation and proof.
4. Draft methods from the existing experiment scripts, preserving the
   pilot/calibration/evaluation split.
5. Build the main results table from the evidence snapshot above.
6. Add limitations before polishing the abstract, so the paper cannot drift
   into overclaiming.
7. Use the gate report as a final wording check before submission.
