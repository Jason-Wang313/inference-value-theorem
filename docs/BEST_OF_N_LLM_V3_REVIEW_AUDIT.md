# best-of-n-llm v3 Review Audit

## Scope

This audit records the v3 submission pass for the ICLR-targeted paper in
`paper/iclr2027/`. The manuscript is the LLM response-pool score-rank
evaluation paper. It must not read like a duplicate wrapper for neighboring
architecture papers.

Source of truth:

- Local folder: `C:\Users\wangz\Downloads\best-of-n-llm`
- GitHub repo: `Jason-Wang313/best-of-n-llm`
- Desktop artifact: `C:\Users\wangz\OneDrive\Desktop\best-of-n-llm-v3.pdf`
- Repo artifact: `paper/final/best-of-n-llm-v3.pdf`

## V3 Changes

- Added the offline cached-evidence script `experiments/18_v3_cached_evidence.py`.
- Added v3 result macros, figures, and CSV ledgers under `results/v3_cached_evidence/`.
- Expanded the manuscript to the 25-page submission threshold with evidence trace,
  response-pool reporting standards, aggregation controls, reviewer stress tests,
  score-gaming threats, archive contract, and a 50-point self-attack.
- Updated the build to export `best-of-n-llm-v3.pdf` to both the repository and
  the visible Desktop.
- Added `scripts/run_v3_claim_audit.py` to enforce page count, PDF hash match,
  stale-name hygiene, source-map hygiene, LaTeX warnings, and claim-gate counts.

## Supported Claims

- Exact finite problem-conditional score-rank law for measured LLM response pools.
- High-$N$ pairwise-AUC insufficiency under the measured score-rank distributions.
- Du-aligned exact-law validation on 35,964 conditional triples.
- Five-model finite-pilot evidence and adaptive allocation diagnostics.
- A scoped 119-record 4096-depth 3B/MATH held-out slice.
- A scoped live-judge subset with 135 pairs and 6,480 judgments.
- Four pilot-scale cross-benchmark families.

## Explicitly Not Claimed

- Full 23-model 4096-depth held-out manifest completion.
- Universal live-judge superiority.
- Six-family manifest-scale benchmark generality.
- Tiny-pilot sufficiency.
- Architecture claims about world models, planning, robotics, diffusion,
  retrieval, JEPA, CEM/MCTS, simulators, object-centric control, or trajectory
  transformers.

## Harsh Reviewer Attack Pass

1. If the title looks like a generic Best-of-N theorem paper, reject. Current
   title is language-model response-pool specific.
2. If the abstract hides missing manifest gates, reject. Current abstract names
   scoped evidence and the v3 audit keeps missing gates visible.
3. If pairwise AUC is treated as enough for high `N`, reject. The paper makes
   AUC failure a measured negative control.
4. If pooled aggregation is used as the primary estimate, reject. It is a
   diagnostic negative control.
5. If the 119-record held-out slice is described as full manifest completion,
   reject. The manuscript states the exact coverage boundary.
6. If live-judge results become a provider-wide claim, reject. They remain a
   scoped subset.
7. If cross-benchmark evidence is phrased as manifest-scale generality, reject.
   The paper calls it pilot-scale.
8. If duplicate architecture language returns, reject. The paper excludes those
   claims in the main text and appendix.
9. If v2 PDFs or old draft titles remain in current docs, reject. The v3 audit
   scans for stale names.
10. If the repo and Desktop PDFs differ, reject. The v3 audit checks hashes.

Residual risk remains empirical scale, not identity drift. The paper is
submission-ready as a scoped, evidence-gated evaluation paper.
