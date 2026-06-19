# Final Audit

## Main Thesis

Multi-sample language-model systems cannot be evaluated by a pairwise ranking
number alone. Once a system draws many candidate answers and returns the
highest-scoring candidate, expected selected correctness depends on the full
problem-conditioned score-rank distribution.

## v4 Novelty

The v4 paper is framed as a score-rank evaluation paper for LLM response pools,
not as a generic architecture or "Best-of-N" wrapper. The contribution is an
exact finite response-pool law, a gated evidence package, and a reporting
standard that separates supported high-N selection claims from incomplete
manifest-scale targets.

## Duplicate-Risk Boundary

The manuscript explicitly excludes world models, planning architectures,
robotics, diffusion policies, retrieval, control, and other neighboring paper
families. Its scientific object is the LLM response pool: a fixed benchmark
problem, a sampled candidate set, binary correctness labels, scalar scores,
ties, and a draw-with-replacement selection rule. This makes the paper
distinguishable beside sibling papers even when reviewers compare them side by
side.

## Evidence Package

The checked evidence includes:

- 35,964 model/problem/N triples for the exact response-pool law.
- Mean exact-law MAE `6.22e-4`.
- AUC-only MAE `0.0821` at `N=48`, showing why pairwise ranking is incomplete.
- Five benchmark-card families spanning MATH, GPQA Diamond, IFEval, LiveBench,
  and LiveCodeBench.
- Four completed cross-benchmark pilot families.
- 119 completed 4096-depth held-out 3B/MATH records.
- 135 live-judge model/problem pairs and 6,480 stored judgments.
- A claim-gate ledger with 8 supported claims and 4 disclosed missing scale
  gates.

## Proof Status

The formal result is an exact finite-pool order-statistic law. For a fixed
problem-conditioned empirical response pool, scalar score, correctness label,
and uniform tie-breaking rule, the expected correctness of selecting the
highest-scoring response from `N` independent draws is determined by the
score-rank intervals and labels. The theorem is not used to claim universal
LLM improvement, benchmark saturation, or architecture-specific behavior.

## Real-Benchmark Coverage

The paper includes benchmark-card evidence for MATH, GPQA Diamond, IFEval,
LiveBench, and LiveCodeBench. It also reports a held-out 4096-depth MATH slice.
These are not presented as full manifest completion: the paper states the
remaining scale gaps directly and uses them as claim boundaries.

## Biggest Remaining Weaknesses

- The held-out 4096-depth evidence is 119 completed records, not the full
  11,500-record manifest.
- Cross-benchmark evidence is pilot-scale for the completed families.
- Live-judge evidence supports a scoped subset, not broad judge dominance.
- Finite-pilot forecasting remains sample-limited.
- Adaptive allocation is directionally useful but not globally optimal.

## Submission-Readiness Judgment

Submission-ready v4 as a scoped LLM response-pool evaluation paper. The current
artifact is not a maxed-out empirical manifest and does not claim to be one.
Its strength is that the theorem, evidence, baselines, stress tests, missing
gates, and duplicate-risk boundary are all explicit and locally auditable.

## Verification

- `python -m py_compile config.py experiments\15_paper_claims_status.py experiments\16_cross_benchmark.py experiments\18_v4_protocol_evidence.py scripts\run_v4_claim_audit.py`: passed.
- Protocol evidence regeneration via `paper\iclr2027\build.ps1 -Clean`: passed.
- Boxed-link PDF rebuild from frozen paper artifacts: passed.
- Claim audit: `submission audit complete: best-of-n-llm v4`.
- Final LaTeX log scan: no undefined citations, unresolved references, rerun
  requests, overfull boxes, or hyperref/natbib blocking warnings.
- Final PDF page count: 26 pages.
- Final PDF SHA256:
  `5B517B673EAA75437E071DA52ED27D80A391A2203C9D2A6866D61D77B4B06C02`
- PDF annotation audit found 27 link annotations: 14 green citation boxes and
  13 red internal-reference boxes. All annotations use 1pt visible borders
  matching the VLA role-model profile.
- Visual QA inspected rendered pages 1, 2, 3, 4, 5, 6, 7, 9, 15, 22, and 26.

## Final PDF Path

`C:\Users\wangz\OneDrive\Desktop\best-of-n-llm-v4.pdf`

## GitHub Repo URL

`https://github.com/Jason-Wang313/best-of-n-llm`
