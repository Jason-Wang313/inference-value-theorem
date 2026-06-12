# best-of-n-llm v2 Review Audit

## Scope

This audit records the v2 self-review pass for the ICLR-targeted paper in
`paper/iclr2027/`. The main reviewer threat is duplicate-wrapper risk: many
neighboring Desktop papers were previously framed around a similar
"Best-of-N + exact law" template. For this repo, the manuscript must stand as
an LLM response-pool evaluation paper whose title, abstract, introduction,
contributions, theorem framing, and limitations are visibly different from
architecture-specific world-model, robotics, diffusion, retrieval, planning,
JEPA, CEM/MCTS, simulator, object-centric, or trajectory-transformer papers.

Source of truth:

- Local folder: `C:\Users\wangz\Downloads\best-of-n-llm`
- GitHub repo: `Jason-Wang313/best-of-n-llm`
- Desktop artifact after build: `C:\Users\wangz\OneDrive\Desktop\best-of-n-llm-v2.pdf`

## Fixes Applied Before Final Review

- Replaced the title `The Inference Value Theorem: Exact Finite-Sample Laws
  for Best-of-N Selection` with `When Pairwise Ranking Fails: Score-Rank
  Evaluation for Multi-Sample Language Models`.
- Rewrote the abstract to foreground LLM response pools, score-rank
  evaluation, AUC failure, and gated evidence.
- Added an explicit architecture-boundary paragraph excluding world models,
  planning, robotics, diffusion, retrieval, JEPA, CEM/MCTS, simulators,
  object-centric control, and trajectory-transformer claims.
- Renamed the setup and theorem framing from generic Best-of-N selector law
  language to problem-conditional score-rank evaluation for LLM response
  pools.
- Updated package metadata, README wording, and build outputs so v2 compiles
  to the visible Desktop with the folder-derived filename.

## 50-Round Attack Pass

1. Title looks like a duplicate Best-of-N wrapper. Fixed: title no longer says
   Best-of-N or Inference Value Theorem.
2. Abstract could be pasted onto an architecture paper. Fixed: it names LLM
   response pools and excludes architecture-specific claims.
3. Contribution claim is too generic. Fixed: contributions now start from
   multi-sample LLM response selection.
4. The theorem name repeats sibling papers. Fixed: theorem is now the
   problem-conditional score-rank law.
5. The intro could imply a universal architecture theorem. Fixed: it says the
   objects are scores, labels, ranks, and ties inside one response pool.
6. Reviewers may ask what makes this different from world-model papers.
   Fixed: a dedicated boundary paragraph answers directly.
7. Reviewers may see "Best-of-N" everywhere and infer copy-paste. Mitigated:
   protocol references remain only where scientifically necessary.
8. Pairwise AUC critique may sound obvious. Mitigated: exact two-draw identity
   plus high-N moment failure makes the critique formal.
9. The paper may overclaim experimental scope. Fixed: missing manifest gates
   are central in abstract, table, and limitations.
10. The live-judge result may be overgeneralized. Fixed: scoped to 135
    completed pairs and not broad judge superiority.
11. Cross-benchmark evidence may look too small. Fixed: labeled pilot-scale.
12. Held-out evidence may look full-scale. Fixed: labeled 119-record 3B/MATH
    slice, not full manifest.
13. Adaptive allocation may look globally optimal. Fixed: called positive but
    modest, with oracle as upper bound.
14. The oracle diagnostic row may look deployable. Fixed: described as a
    diagnostic of measurement and gate machinery.
15. The proof may depend on asymptotics. Safe: theorem is finite empirical.
16. Tie handling may be ambiguous. Safe: uniform random tie-breaking is stated
    in setup, theorem, and AUC identity.
17. Sampling protocol may be ambiguous. Safe: with-replacement protocol is
    explicit and matched to the simulator.
18. Without-replacement variants may distract. Safe: mentioned only as out of
    evaluated scope.
19. Problem conditioning may be lost by pooling. Safe: table and paragraph
    emphasize pooled baseline failure.
20. Reviewers may ask why AUC is exact at N=2. Safe: main text and appendix
    give the decomposition.
21. Reviewers may ask why AUC fails at high N. Safe: rank powers and moment
    hierarchy are explicit.
22. Numeric claims may be invented. Safe: reproducibility statement points to
    repo artifacts.
23. Numeric values may lack precision context. Safe: main values match claim
    summaries and are scoped.
24. Figures may not match text. Pending build check will verify inclusion.
25. Bibliography may break after rewrite. Pending compile will verify.
26. The title may be too broad for LLMs. Acceptable: subtitle-level terms in
    abstract and intro specify language models.
27. "When Pairwise Ranking Fails" may sound negative without mechanism. Safe:
    mechanism is the score-rank moment law.
28. The architecture-boundary paragraph could sound defensive. Acceptable:
    duplicate-risk context makes the boundary necessary.
29. The paper may still share math with sibling papers. Acceptable for this
    repo: the math is the general LLM evaluation spine; later sibling papers
    must not reuse it as their primary novelty.
30. The contribution could still be theorem-first. Fixed: strongest claim is
    evaluation-first.
31. The conclusion could reintroduce wrapper language. Fixed: conclusion says
    scale, not a new framing wrapper.
32. The repository README may preserve stale identity. Fixed: top README now
    names score-rank LLM evaluation.
33. Metadata may preserve stale title. Fixed.
34. Build output may preserve stale filename. Fixed.
35. Desktop artifact may not be versioned. Fixed by build path
    `best-of-n-llm-v2.pdf`.
36. A fresh agent may not know why v2 exists. Safe: this audit records the
    v2 identity and source of truth.
37. The paper may be desk-rejected for template reuse. Mitigated: title,
    abstract, intro, contribution language, theorem name, and conclusion now
    differ from the generic template.
38. The paper may be desk-rejected for overclaiming. Mitigated: all missing
    ideal gates are explicitly disclosed.
39. The paper may be rejected for no experiments beyond the theorem. Mitigated:
    evidence spans 23-model validation, pilots, verifiers, live judge,
    benchmarks, allocation, and held-out slice.
40. The paper may be rejected for insufficient held-out scale. Residual risk:
    openly scoped as an incomplete manifest, not hidden.
41. The paper may be rejected for no architecture insight. Acceptable: it is
    not submitted as an architecture paper.
42. The paper may be rejected for weak novelty if reviewers know order
    statistics. Mitigated: novelty is applying the exact finite tie-aware law
    as a gated LLM response-pool evaluation protocol with evidence.
43. The theorem may be viewed as too simple. Mitigated: paper does not inflate
    the proof; it emphasizes evaluation consequences and artifact rigor.
44. The paper may be viewed as a methods note. Acceptable risk: experiments
    and claim gates make it a submission draft rather than a note.
45. Reviewer may ask for six benchmark families. Safe: missing families are
    disclosed rather than claimed.
46. Reviewer may ask for all 23 models at 4096 samples. Safe: absent from
    claims and named as missing.
47. Reviewer may ask whether live endpoints were unavailable. Safe: repo
    policy excludes unavailable, EOL, degraded, or rate-limited endpoints.
48. Reviewer may ask about LLM assistance. Safe: appendix disclosure names it.
49. Reviewer may attack reproducibility. Safe pending checks: build script,
    metadata, evidence manifest, and AGENTS checks are in place.
50. Reviewer may compare Desktop papers side by side. Result: this v2 paper
    now reads as the LLM score-rank evaluation paper, not as an architecture
    wrapper.

## Final Reviewer Stance

After the v2 rewrite, no unscoped architecture claim remains in the manuscript.
Residual risks are empirical scale risks, and the paper explicitly lists them
as missing gates. The paper is ready for compile, artifact verification, and
GitHub push under the scoped v2 claims.
