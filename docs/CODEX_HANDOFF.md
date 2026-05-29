# Codex Handoff

## 1. Current Goal
Write the actual inference-value theorem manuscript from the pushed evidence package.

Current status: the experiment evidence package is complete and pushed, but the manuscript has not been written in this repository.

## 2. Important Repo Facts
- Repository path: `C:\Users\wangz\Downloads\inference-value-theorem-github`.
- GitHub remote: `https://github.com/Jason-Wang313/inference-value-theorem.git`.
- Last verified evidence commit before this handoff setup: `d598e6391 Add paper closure experiments and cross-benchmark validation`.
- This repo contains experiment code, results, figures, tables, and claim-status docs.
- No manuscript `.tex` file was found in this repo during the last inspection.
- Separate manuscript-like files exist under `C:\Users\wangz\MIRROR\paper`, but that is a different repo (`https://github.com/Jason-Wang313/Mirror.git`) and was last observed on branch `submission-v39`.
- API keys are not committed. `config.py` can read `NIM_API_KEYS` or fall back to `C:\Users\wangz\MIRROR\.env` without printing key material.
- `src/nim_client.py` can fall back to `C:\Users\wangz\Downloads\inference-value-theorem\data\raw` when this checkout's `data/raw` is empty.

## 3. Files Changed And Why
- `README.md`: documents the Du-aligned and cross-benchmark experiment commands and current closure artifacts.
- `config.py`: reads NIM keys from environment or mirror `.env`; disables unavailable, EOL, degraded, or high-volume rate-limited NIM endpoints.
- `experiments/13_adaptive_allocation.py`: adaptive compute-allocation experiment.
- `experiments/14_live_judge_analysis.py`: live judge score analysis.
- `experiments/15_paper_claims_status.py`: writes consolidated paper-claim status.
- `experiments/16_cross_benchmark.py`: runs GPQA Diamond, IFEval, LiveBench selected, and LiveCodeBench collection/measurement/analysis.
- `results/du_aligned/`: updated Du-aligned summaries, tables, figures, expanded pilot outputs, adaptive allocation outputs, and live judge outputs.
- `results/benchmarks/`: cross-benchmark raw samples, measurements, summaries, tables, tasks, and status docs.
- `AGENTS.md`: durable repo instructions for future Codex sessions.
- `docs/CODEX_HANDOFF.md`: this handoff file.

## 4. Commands And Checks
Passed before evidence commit `d598e6391`:
- `python -m py_compile config.py experiments\15_paper_claims_status.py experiments\16_cross_benchmark.py`
- `git diff --check`
- targeted secret/key-pattern scan over source/docs/status files
- oversized-file check for files above GitHub's large-file danger threshold
- full cross-benchmark artifact audit over raw sample files, measurement records, grading coverage, exact-law MAE, model count, task count, and `N=48`
- `git push origin main`
- final sync checks showed local `main`, `origin/main`, and GitHub remote head at `d598e63919eaffd8b31a2b3d3794e11fed401224`

Current cross-benchmark summary:

| Benchmark | Records | Coverage | Exact-law MAE | Nondegenerate | N=48 |
| --- | ---: | ---: | ---: | ---: | ---: |
| GPQA Diamond | 320/320 | 1.0 | 0.0 | 251 | 0.5890 |
| IFEval | 320/320 | 1.0 | 0.0 | 164 | 0.7962 |
| LiveBench selected | 320/320 | 1.0 | 0.0 | 141 | 0.2546 |
| LiveCodeBench | 320/320 | 1.0 | 0.0 | 262 | 0.5246 |

Other completed artifacts:
- Core Du-aligned bundle validates the exact finite empirical selector law across 23 models with overall MAE `0.000622`.
- Expanded pilot sample-complexity artifact covers 5 models, 100 problems, 256 samples/problem, and K values through 192.
- Live LLM judge artifact covers 135/135 model/problem pairs and 6480/6480 judgments on a 45-problem manifest.
- Adaptive allocation artifact compares uniform, p-based, AUC/kappa, moment-law, and oracle policies.

## 5. Known Bugs Or Open Questions
- The actual inference-value theorem manuscript is not written in this repo.
- UNKNOWN whether the final manuscript should live in this repo, in `C:\Users\wangz\MIRROR\paper`, or in a new paper directory.
- Optional: add a second external verifier if broader judge-generalization evidence is needed.
- Optional: add hidden/private LiveCodeBench scoring if a stronger code appendix is needed.
- Do not claim all 23 models for the cross-benchmark live panel; claim the 16 stable high-volume live NIM endpoints only.
- Do not claim global optimal allocation, broad judge superiority, or tiny-pilot sufficiency.

## 6. Next Recommended Steps
1. Decide the manuscript location.
2. Create or update a LaTeX manuscript using the pushed evidence package.
3. Use `results/du_aligned/paper_claims_status.md` and `results/benchmarks/benchmark_expansion_status.md` as source-of-truth summaries.
4. Build paper sections: abstract, introduction, theorem, `N=2` AUC identity, moment hierarchy for `N>2`, experiments, cross-benchmark broadening, score-agnostic verifier results, adaptive allocation, limitations, and related work.
5. Compile the PDF and verify that all numeric claims trace to committed result artifacts.
6. Before `/clear`, update this file with the current goal, changed files, checks run, known unknowns, and next steps.
