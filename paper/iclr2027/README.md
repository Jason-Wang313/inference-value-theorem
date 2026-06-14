# ICLR 2027 Submission Draft

This directory contains the anonymous ICLR-targeted manuscript source for:

`When Pairwise Ranking Fails: Score-Rank Evaluation for Multi-Sample Language Models`

The manuscript is scoped to verified artifacts in this repository and is
framed as an LLM response-pool evaluation paper, not an architecture paper. It
should not be edited to claim full maxed-out manifest completion until the
missing claim gates in `results/maxed_out/claim_gate_report.md` pass.

## Template Status

As of 2026-06-09, this package uses the official ICLR 2026 template as the
latest available official ICLR style source:

- Author guide: https://iclr.cc/Conferences/2026/AuthorGuide
- Template ZIP: https://github.com/ICLR/Master-Template/raw/master/iclr2026.zip

The file `iclr2026_conference.sty` is the unmodified official style file.
The file `iclr2027_conference.sty` is a provisional copy with only the
running-header year changed from 2026 to 2027. Replace it with the official
ICLR 2027 style file when that guide is published.

## Build

From this directory:

```powershell
.\build.ps1 -Clean -Package
```

Expected exported artifacts:

- `%USERPROFILE%\OneDrive\Desktop\best-of-n-llm-v3.pdf`
- `%USERPROFILE%\Downloads\best-of-n-llm-v3-source.zip`

## Evidence Sources

Primary artifact inputs:

- `results/du_aligned/du_experiment_results.json`
- `results/du_aligned/du_experiment_summary.txt`
- `results/du_aligned/paper_claims_status.md`
- `results/du_aligned/live_judge_score_summary.json`
- `results/du_aligned/adaptive_allocation_summary.json`
- `results/benchmarks/cross_benchmark_summary.json`
- `results/maxed_out/claim_gate_report.md`
- `results/maxed_out/heldout_forecasting/tables/heldout_locked_estimator_summary.csv`
- `results/maxed_out/heldout_forecasting/measurements/3B/problem_118.json`
- `results/v3_cached_evidence/summary.json`

Main claim boundary:

- Supported: exact finite problem-conditional score-rank law for LLM
  response-pool selection, AUC/kappa insufficiency for high `N`, 23-model
  response-pool validation, five-model finite-pilot curves, scoped 119-record
  4096-depth 3B/MATH held-out slice, scoped live judge subset, four
  pilot-scale cross-benchmark families, and adaptive allocation.
- Not supported yet: full 23-model held-out MATH manifest, six-family
  manifest-scale benchmark coverage, broad live-judge superiority, or
  tiny-pilot sufficiency.
- Out of scope: architecture-specific claims about world models, planning,
  robotics, diffusion policies, retrieval systems, JEPA, CEM/MCTS, simulators,
  object-centric control, or trajectory transformers.

## V3 Audit

After building, run:

```powershell
python ..\..\scripts\run_v3_claim_audit.py
```

The audit checks the 25-page submission threshold, repo/Desktop PDF hash
agreement, v3 evidence summary, missing-gate disclosure, stale v2/title text,
LaTeX blocking warnings, and the Desktop source-map row.
