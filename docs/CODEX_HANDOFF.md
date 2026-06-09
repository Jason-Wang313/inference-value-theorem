# Codex Handoff

## 1. Current Goal
Execute the maxed-out experiment plan by adding a locked, resumable campaign layer on top of the existing evidence package.

Current status: maxed-out campaign orchestration is implemented locally, and full-scale held-out collection has started. 119 contiguous 3B/MATH records are complete at 4096 samples (`problem_0` through `problem_118`). The newest measured record, `problem_118`, has `p=0.0732421875` (300 correct out of 4096), `kappa=0.5267746750965929`, level `5`, and ground truth `81`. A stopped partial top-off run left `problem_119` at `3152/4096` valid raw-cache samples; it is not a measured record yet. The remaining maxed-out claim gates still require broad manifest-scale coverage. A scoped manuscript writing plan now exists at `docs/PAPER_WRITING_PLAN.md`, and the first working draft is `docs/MANUSCRIPT_DRAFT.md`; use these to write now without claiming full maxed-out completion.

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
- `experiments/17_maxed_out_campaign.py`: new maxed-out campaign orchestrator for manifest locking, command queues, MATH collection, MATH measurement, held-out analysis, and status reports. The collector now retires 401/403 auth-failing keys and requeues their batches instead of marking those batches failed; it also treats OpenAI-style `Request timed out.` errors as retryable by matching `timed out`, treats provider `DEGRADED` errors as retryable by matching `degraded`, treats provider 500/inference-connection errors as retryable, treats invalid cache JSON as missing during resume, and writes cache records through a temp file before replacing the final JSON.
- `experiments/16_cross_benchmark.py`: extended with MATH500 and MBPP-backed `humaneval_mbpp` loaders/graders so the six-family cross-benchmark command is runnable.
- `results/maxed_out/`: generated full and smoke manifests, command queues, campaign status files, and smoke held-out artifacts.
- `README.md`: documents the new maxed-out campaign commands and smoke workflow.

## 4. Commands And Checks
Passed before evidence commit `d598e6391`:
- `python -m py_compile config.py experiments\15_paper_claims_status.py experiments\16_cross_benchmark.py`
- `git diff --check`
- targeted secret/key-pattern scan over source/docs/status files
- oversized-file check for files above GitHub's large-file danger threshold
- full cross-benchmark artifact audit over raw sample files, measurement records, grading coverage, exact-law MAE, model count, task count, and `N=48`
- `git push origin main`
- final sync checks showed local `main`, `origin/main`, and GitHub remote head at `d598e63919eaffd8b31a2b3d3794e11fed401224`

Most recent continuation for 3B/problem-51:
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 52 --problem-indices 51 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96 --timeout 300 --dry-run`
- Initial collection and several lower-worker resumes filled the cache but left one interrupted invalid JSON record; after the collector validity patch, `--dry-run` reported exactly one missing valid sample.
- `python -m py_compile experiments\17_maxed_out_campaign.py`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 52 --problem-indices 51 --target-samples 4096 --batch-size 1 --workers 4 --request-delay 2 --progress-every 1 --timeout 60 --dry-run`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 52 --problem-indices 51 --target-samples 4096 --batch-size 1 --workers 4 --request-delay 2 --progress-every 1 --timeout 60`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 52 --problem-indices 51 --target-samples 4096 --batch-size 1 --workers 1 --request-delay 2 --progress-every 1 --timeout 60 --dry-run`
- `python experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 52 --problem-indices 51 --target-samples 4096`
- `python experiments\17_maxed_out_campaign.py analyze-heldout --models 3B`
- `python experiments\17_maxed_out_campaign.py gate-report`
- `python experiments\17_maxed_out_campaign.py status`

Most recent continuation for 3B/problem-52:
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 53 --problem-indices 52 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96 --timeout 300 --dry-run`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 53 --problem-indices 52 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 64 --timeout 180`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 53 --problem-indices 52 --target-samples 4096 --batch-size 8 --workers 1 --request-delay 2 --progress-every 1 --timeout 60 --dry-run`
- `python experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 53 --problem-indices 52 --target-samples 4096`
- `python experiments\17_maxed_out_campaign.py analyze-heldout --models 3B`
- `python experiments\17_maxed_out_campaign.py gate-report`
- `python experiments\17_maxed_out_campaign.py status`

Most recent continuation for 3B/problem-61:
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 62 --problem-indices 61 --target-samples 512 --batch-size 1 --workers 4 --request-delay 3 --progress-every 64 --timeout 90 --max-job-attempts 2`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 62 --problem-indices 61 --target-samples 1024 --batch-size 1 --workers 4 --request-delay 3 --progress-every 128 --timeout 90 --max-job-attempts 2`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 62 --problem-indices 61 --target-samples 2048 --batch-size 1 --workers 4 --request-delay 3 --progress-every 128 --timeout 90 --max-job-attempts 2`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 62 --problem-indices 61 --target-samples 3072 --batch-size 1 --workers 4 --request-delay 3 --progress-every 128 --timeout 90 --max-job-attempts 2`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 62 --problem-indices 61 --target-samples 3072 --batch-size 1 --workers 1 --request-delay 3 --progress-every 1 --timeout 120 --max-job-attempts 4`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 62 --problem-indices 61 --target-samples 4096 --batch-size 1 --workers 4 --request-delay 3 --progress-every 128 --timeout 90 --max-job-attempts 2`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 62 --problem-indices 61 --target-samples 4096 --batch-size 1 --workers 1 --request-delay 3 --progress-every 1 --timeout 120 --max-job-attempts 4`
- `python experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 62 --problem-indices 61 --target-samples 4096 --progress-every 1`
- `python experiments\17_maxed_out_campaign.py analyze-heldout --models 3B`
- `python experiments\17_maxed_out_campaign.py gate-report`
- `python experiments\17_maxed_out_campaign.py status`

Most recent continuation for 3B/problem-62:
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 63 --problem-indices 62 --target-samples 512 --batch-size 1 --workers 4 --request-delay 3 --progress-every 64 --timeout 90 --max-job-attempts 2`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 63 --problem-indices 62 --target-samples 1024 --batch-size 1 --workers 4 --request-delay 3 --progress-every 128 --timeout 90 --max-job-attempts 2`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 63 --problem-indices 62 --target-samples 1024 --batch-size 1 --workers 1 --request-delay 3 --progress-every 1 --timeout 120 --max-job-attempts 4`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 63 --problem-indices 62 --target-samples 2048 --batch-size 1 --workers 4 --request-delay 3 --progress-every 128 --timeout 90 --max-job-attempts 2`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 63 --problem-indices 62 --target-samples 3072 --batch-size 1 --workers 8 --request-delay 2 --progress-every 128 --timeout 120 --max-job-attempts 3`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 63 --problem-indices 62 --target-samples 3072 --batch-size 1 --workers 1 --request-delay 3 --progress-every 1 --timeout 120 --max-job-attempts 4`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 63 --problem-indices 62 --target-samples 4096 --batch-size 1 --workers 12 --request-delay 2 --progress-every 128 --timeout 120 --max-job-attempts 3`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 63 --problem-indices 62 --target-samples 4096 --batch-size 1 --workers 1 --request-delay 3 --progress-every 1 --timeout 120 --max-job-attempts 4`
- `python experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 63 --problem-indices 62 --target-samples 4096 --progress-every 1`
- `python experiments\17_maxed_out_campaign.py analyze-heldout --models 3B`
- `python experiments\17_maxed_out_campaign.py gate-report`
- `python experiments\17_maxed_out_campaign.py status`
- Notes: the 3072 and 4096 main runs each had one inference-service 500 failure, and both were recovered by single-worker top-off. The final measured record has `p=0.015869140625` (65 correct out of 4096) and `kappa=0.4005305039787798`.

Most recent continuation for 3B/problem-63:
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 64 --problem-indices 63 --target-samples 512 --batch-size 1 --workers 4 --request-delay 3 --progress-every 64 --timeout 90 --max-job-attempts 2`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 64 --problem-indices 63 --target-samples 512 --batch-size 1 --workers 1 --request-delay 3 --progress-every 1 --timeout 120 --max-job-attempts 4`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 64 --problem-indices 63 --target-samples 1024 --batch-size 1 --workers 4 --request-delay 3 --progress-every 128 --timeout 90 --max-job-attempts 2`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 64 --problem-indices 63 --target-samples 2048 --batch-size 1 --workers 4 --request-delay 3 --progress-every 128 --timeout 90 --max-job-attempts 2`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 64 --problem-indices 63 --target-samples 2048 --batch-size 1 --workers 1 --request-delay 3 --progress-every 1 --timeout 120 --max-job-attempts 4`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 64 --problem-indices 63 --target-samples 3072 --batch-size 1 --workers 8 --request-delay 2 --progress-every 128 --timeout 120 --max-job-attempts 3`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 64 --problem-indices 63 --target-samples 3072 --batch-size 1 --workers 1 --request-delay 3 --progress-every 1 --timeout 120 --max-job-attempts 4`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 64 --problem-indices 63 --target-samples 4096 --batch-size 1 --workers 12 --request-delay 2 --progress-every 128 --timeout 120 --max-job-attempts 3`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 64 --problem-indices 63 --target-samples 4096 --batch-size 1 --workers 1 --request-delay 3 --progress-every 1 --timeout 120 --max-job-attempts 4`
- `python experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 64 --problem-indices 63 --target-samples 4096 --progress-every 1`
- `python experiments\17_maxed_out_campaign.py analyze-heldout --models 3B`
- `python experiments\17_maxed_out_campaign.py gate-report`
- `python experiments\17_maxed_out_campaign.py status`
- Notes: collection recovered isolated inference-service 500s by single-worker top-off at each affected checkpoint. The final measured record has `p=0.068603515625` (281 correct out of 4096) and `kappa=0.5514437764396953`.

Most recent continuation for 3B/problem-64:
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 65 --problem-indices 64 --target-samples 512 --batch-size 1 --workers 4 --request-delay 3 --progress-every 64 --timeout 90 --max-job-attempts 2`
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 65 --problem-indices 64 --target-samples 512 --batch-size 1 --workers 1 --request-delay 3 --progress-every 1 --timeout 120 --max-job-attempts 4`
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 65 --problem-indices 64 --target-samples 1024 --batch-size 1 --workers 4 --request-delay 3 --progress-every 128 --timeout 90 --max-job-attempts 2`
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 65 --problem-indices 64 --target-samples 1024 --batch-size 1 --workers 1 --request-delay 3 --progress-every 1 --timeout 120 --max-job-attempts 4`
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 65 --problem-indices 64 --target-samples 2048 --batch-size 1 --workers 4 --request-delay 3 --progress-every 128 --timeout 90 --max-job-attempts 2`
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 65 --problem-indices 64 --target-samples 2048 --batch-size 1 --workers 1 --request-delay 3 --progress-every 1 --timeout 120 --max-job-attempts 4`
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 65 --problem-indices 64 --target-samples 3072 --batch-size 1 --workers 8 --request-delay 2 --progress-every 128 --timeout 120 --max-job-attempts 3`
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 65 --problem-indices 64 --target-samples 3072 --batch-size 1 --workers 1 --request-delay 3 --progress-every 1 --timeout 120 --max-job-attempts 4`
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 65 --problem-indices 64 --target-samples 4096 --batch-size 1 --workers 12 --request-delay 2 --progress-every 128 --timeout 120 --max-job-attempts 3`
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 65 --problem-indices 64 --target-samples 4096 --batch-size 1 --workers 1 --request-delay 3 --progress-every 1 --timeout 120 --max-job-attempts 4`
- `python experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 65 --problem-indices 64 --target-samples 4096 --progress-every 1`
- `python experiments\17_maxed_out_campaign.py analyze-heldout --models 3B`
- `python experiments\17_maxed_out_campaign.py gate-report`
- `python experiments\17_maxed_out_campaign.py status`
- Notes: collection recovered provider 500s and isolated failed tails by single-worker top-off at each affected checkpoint. The final measured record has `p=0.002197265625` (9 correct out of 4096), `kappa=0.42508767637223716`, and ground truth `\frac{35}{64}`.

Most recent continuation for 3B/problem-65:
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 66 --problem-indices 65 --target-samples 512 --batch-size 1 --workers 4 --request-delay 3 --progress-every 64 --timeout 90 --max-job-attempts 2`
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 66 --problem-indices 65 --target-samples 512 --batch-size 1 --workers 1 --request-delay 3 --progress-every 1 --timeout 120 --max-job-attempts 4`
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 66 --problem-indices 65 --target-samples 1024 --batch-size 1 --workers 4 --request-delay 3 --progress-every 128 --timeout 90 --max-job-attempts 2`
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 66 --problem-indices 65 --target-samples 1024 --batch-size 1 --workers 1 --request-delay 3 --progress-every 1 --timeout 120 --max-job-attempts 4`
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 66 --problem-indices 65 --target-samples 2048 --batch-size 1 --workers 4 --request-delay 3 --progress-every 128 --timeout 90 --max-job-attempts 2`
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 66 --problem-indices 65 --target-samples 2048 --batch-size 1 --workers 1 --request-delay 3 --progress-every 1 --timeout 120 --max-job-attempts 4`
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 66 --problem-indices 65 --target-samples 3072 --batch-size 1 --workers 8 --request-delay 2 --progress-every 128 --timeout 120 --max-job-attempts 3`
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 66 --problem-indices 65 --target-samples 4096 --batch-size 1 --workers 12 --request-delay 2 --progress-every 128 --timeout 120 --max-job-attempts 3`
- `python experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 66 --problem-indices 65 --target-samples 4096 --progress-every 1`
- `python experiments\17_maxed_out_campaign.py analyze-heldout --models 3B`
- `python experiments\17_maxed_out_campaign.py gate-report`
- `python experiments\17_maxed_out_campaign.py status`
- Notes: problem-65 started with 256 valid cached samples. The 512 and 1024 checkpoints each needed one single-worker top-off, the 2048 checkpoint needed three top-offs, and the 3072 and 4096 checkpoints completed cleanly. The final measured record has `p=0.955078125` (3912 correct out of 4096), `kappa=0.5722373188405797`, and ground truth `1`.

Most recent continuation for 3B/problem-66:
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 67 --problem-indices 66 --target-samples 512 --batch-size 1 --workers 4 --request-delay 3 --progress-every 64 --timeout 90 --max-job-attempts 2`
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 67 --problem-indices 66 --target-samples 1024 --batch-size 1 --workers 4 --request-delay 3 --progress-every 128 --timeout 90 --max-job-attempts 2`
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 67 --problem-indices 66 --target-samples 2048 --batch-size 1 --workers 4 --request-delay 3 --progress-every 128 --timeout 90 --max-job-attempts 2`
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 67 --problem-indices 66 --target-samples 2048 --batch-size 1 --workers 1 --request-delay 3 --progress-every 1 --timeout 120 --max-job-attempts 4`
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 67 --problem-indices 66 --target-samples 3072 --batch-size 1 --workers 8 --request-delay 2 --progress-every 128 --timeout 120 --max-job-attempts 3`
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 67 --problem-indices 66 --target-samples 4096 --batch-size 1 --workers 12 --request-delay 2 --progress-every 128 --timeout 120 --max-job-attempts 3`
- `python experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 67 --problem-indices 66 --target-samples 4096 --progress-every 1`
- `python experiments\17_maxed_out_campaign.py analyze-heldout --models 3B`
- `python experiments\17_maxed_out_campaign.py gate-report`
- `python experiments\17_maxed_out_campaign.py status`
- Notes: problem-66 started with 256 valid cached samples. The 512 and 1024 checkpoints completed cleanly; the 2048 checkpoint had one recovered requeue and two provider-500 failures at sample slots 1798 and 1959, both recovered by single-worker top-off; the 3072 and 4096 checkpoints completed cleanly. The final measured record has `p=0.16650390625` (682 correct out of 4096), `kappa=0.5328090989834853`, and ground truth `x^3+3x-6`.

Most recent continuation for 3B/problem-67:
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 68 --problem-indices 67 --target-samples 512 --batch-size 1 --workers 4 --request-delay 3 --progress-every 64 --timeout 90 --max-job-attempts 2`
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 68 --problem-indices 67 --target-samples 1024 --batch-size 1 --workers 4 --request-delay 3 --progress-every 128 --timeout 90 --max-job-attempts 2`
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 68 --problem-indices 67 --target-samples 2048 --batch-size 1 --workers 4 --request-delay 3 --progress-every 128 --timeout 90 --max-job-attempts 2`
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 68 --problem-indices 67 --target-samples 2048 --batch-size 1 --workers 1 --request-delay 3 --progress-every 1 --timeout 120 --max-job-attempts 4`
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 68 --problem-indices 67 --target-samples 3072 --batch-size 1 --workers 8 --request-delay 2 --progress-every 128 --timeout 120 --max-job-attempts 3`
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 68 --problem-indices 67 --target-samples 3072 --batch-size 1 --workers 1 --request-delay 3 --progress-every 1 --timeout 120 --max-job-attempts 4`
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 68 --problem-indices 67 --target-samples 4096 --batch-size 1 --workers 12 --request-delay 2 --progress-every 128 --timeout 120 --max-job-attempts 3`
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 68 --problem-indices 67 --target-samples 4096 --batch-size 1 --workers 1 --request-delay 3 --progress-every 1 --timeout 120 --max-job-attempts 4`
- `python experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 68 --problem-indices 67 --target-samples 4096 --progress-every 1`
- `python experiments\17_maxed_out_campaign.py analyze-heldout --models 3B`
- `python experiments\17_maxed_out_campaign.py gate-report`
- `python experiments\17_maxed_out_campaign.py status`
- Notes: problem-67 started with 64 valid cached samples. The 512 checkpoint completed cleanly; the 1024 checkpoint had three recovered requeues and no failures; the 2048 checkpoint had one provider-500 failure at sample slot 1989 recovered by single-worker top-off; the 3072 checkpoint had one provider-500 failure at sample slot 2057 recovered by top-off; and the 4096 checkpoint had two recovered requeues plus provider-500 failures at sample slots 3201 and 3361, both recovered by top-off. The final measured record has `p=0.970947265625` (3977 correct out of 4096), `kappa=0.854079021601097`, and ground truth `10`.

Most recent continuation for 3B/problem-68:
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 69 --problem-indices 68 --target-samples 512 --batch-size 1 --workers 4 --request-delay 3 --progress-every 64 --timeout 90 --max-job-attempts 2`
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 69 --problem-indices 68 --target-samples 1024 --batch-size 1 --workers 4 --request-delay 3 --progress-every 128 --timeout 90 --max-job-attempts 2`
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 69 --problem-indices 68 --target-samples 2048 --batch-size 1 --workers 4 --request-delay 3 --progress-every 128 --timeout 90 --max-job-attempts 2`
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 69 --problem-indices 68 --target-samples 2048 --batch-size 1 --workers 1 --request-delay 3 --progress-every 1 --timeout 120 --max-job-attempts 4`
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 69 --problem-indices 68 --target-samples 3072 --batch-size 1 --workers 8 --request-delay 2 --progress-every 128 --timeout 120 --max-job-attempts 3`
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 69 --problem-indices 68 --target-samples 3072 --batch-size 1 --workers 1 --request-delay 3 --progress-every 1 --timeout 120 --max-job-attempts 4`
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 69 --problem-indices 68 --target-samples 4096 --batch-size 1 --workers 12 --request-delay 2 --progress-every 128 --timeout 120 --max-job-attempts 3`
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 69 --problem-indices 68 --target-samples 4096 --batch-size 1 --workers 8 --request-delay 2 --progress-every 128 --timeout 120 --max-job-attempts 3`
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 69 --problem-indices 68 --target-samples 4096 --batch-size 1 --workers 1 --request-delay 3 --progress-every 1 --timeout 120 --max-job-attempts 4`
- `python experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 69 --problem-indices 68 --target-samples 4096 --progress-every 1`
- `python experiments\17_maxed_out_campaign.py analyze-heldout --models 3B`
- `python experiments\17_maxed_out_campaign.py gate-report`
- `python experiments\17_maxed_out_campaign.py status`
- Notes: problem-68 started with 256 valid cached samples. The 512 checkpoint completed cleanly; the 1024 checkpoint had four recovered requeues and no failures; the 2048 checkpoint had six recovered requeues and one provider-500 failure at sample slot 1508 recovered by top-off; the 3072 checkpoint had three recovered requeues and provider-500 failures at sample slots 2583 and 3030, both recovered by top-off. The first 4096 launch exited early after adding eight samples and left no finished status, so the checkpoint was rerun with 8 workers; that retry had one provider-500 failure at sample slot 3448, recovered by top-off. The final measured record has `p=0.0810546875` (332 correct out of 4096), `kappa=0.4233344109701292`, and ground truth `46`.

Previous continuation for 3B/problem-69:
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 70 --problem-indices 69 --target-samples 512 --batch-size 1 --workers 4 --request-delay 3 --progress-every 64 --timeout 90 --max-job-attempts 2`
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 70 --problem-indices 69 --target-samples 1024 --batch-size 1 --workers 4 --request-delay 3 --progress-every 128 --timeout 90 --max-job-attempts 2`
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 70 --problem-indices 69 --target-samples 1024 --batch-size 1 --workers 1 --request-delay 3 --progress-every 1 --timeout 120 --max-job-attempts 4`
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 70 --problem-indices 69 --target-samples 2048 --batch-size 1 --workers 4 --request-delay 3 --progress-every 128 --timeout 90 --max-job-attempts 2`
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 70 --problem-indices 69 --target-samples 2048 --batch-size 1 --workers 1 --request-delay 3 --progress-every 1 --timeout 120 --max-job-attempts 4`
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 70 --problem-indices 69 --target-samples 3072 --batch-size 1 --workers 8 --request-delay 2 --progress-every 128 --timeout 120 --max-job-attempts 3`
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 70 --problem-indices 69 --target-samples 3072 --batch-size 1 --workers 1 --request-delay 3 --progress-every 1 --timeout 120 --max-job-attempts 4`
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 70 --problem-indices 69 --target-samples 4096 --batch-size 1 --workers 8 --request-delay 2 --progress-every 128 --timeout 120 --max-job-attempts 3`
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 70 --problem-indices 69 --target-samples 4096 --batch-size 1 --workers 1 --request-delay 3 --progress-every 1 --timeout 120 --max-job-attempts 4`
- `python experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 70 --problem-indices 69 --target-samples 4096 --progress-every 1`
- `python experiments\17_maxed_out_campaign.py analyze-heldout --models 3B`
- `python experiments\17_maxed_out_campaign.py gate-report`
- `python experiments\17_maxed_out_campaign.py status`
- Notes: problem-69 started with 64 valid cached samples. The 512 checkpoint completed cleanly; the 1024 checkpoint had three recovered requeues and provider-500 failures at sample slots 544, 608, and 777, all recovered by top-off; the 2048 checkpoint had one provider-500 failure at sample slot 1518 recovered by top-off; the 3072 checkpoint had provider-500 failures at sample slots 2084 and 2830, both recovered by top-off. A first background 4096 launch exited immediately without touching cache or status; the foreground 8-worker 4096 run then filled the checkpoint except provider-500 failures at sample slots 3597 and 3636, both recovered by top-off. The final measured record has `p=0.392822265625` (1609 correct out of 4096), `kappa=0.4213207623083165`, and ground truth `-1`.

Previous continuation for 3B/problem-70:
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 71 --problem-indices 70 --target-samples 512 --batch-size 1 --workers 4 --request-delay 3 --progress-every 64 --timeout 90 --max-job-attempts 2`
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 71 --problem-indices 70 --target-samples 512 --batch-size 1 --workers 1 --request-delay 3 --progress-every 1 --timeout 120 --max-job-attempts 4`
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 71 --problem-indices 70 --target-samples 1024 --batch-size 1 --workers 4 --request-delay 3 --progress-every 128 --timeout 90 --max-job-attempts 2`
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 71 --problem-indices 70 --target-samples 2048 --batch-size 1 --workers 4 --request-delay 3 --progress-every 128 --timeout 90 --max-job-attempts 2`
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 71 --problem-indices 70 --target-samples 2048 --batch-size 1 --workers 1 --request-delay 3 --progress-every 1 --timeout 120 --max-job-attempts 4`
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 71 --problem-indices 70 --target-samples 3072 --batch-size 1 --workers 8 --request-delay 2 --progress-every 128 --timeout 120 --max-job-attempts 3`
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 71 --problem-indices 70 --target-samples 4096 --batch-size 1 --workers 8 --request-delay 2 --progress-every 128 --timeout 120 --max-job-attempts 3`
- `python -u experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 71 --problem-indices 70 --target-samples 4096 --progress-every 1`
- `python experiments\17_maxed_out_campaign.py analyze-heldout --models 3B`
- `python experiments\17_maxed_out_campaign.py gate-report`
- `python experiments\17_maxed_out_campaign.py status`
- Notes: problem-70 started with 64 valid cached samples. The 512 checkpoint had one provider-500 failure at sample slot 382, recovered by one-worker top-off; the 1024 checkpoint completed cleanly; the 2048 checkpoint had one provider-500 failure at sample slot 1779, recovered by top-off; the 3072 and 4096 checkpoints completed cleanly with 8 workers. The final measured record has `p=0.908935546875` (3723 correct out of 4096), `kappa=0.6878990753082606`, and ground truth `40_9`.

Previous continuation for 3B/problem-71:
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 72 --problem-indices 71 --target-samples 512 --batch-size 1 --workers 4 --request-delay 3 --progress-every 64 --timeout 90 --max-job-attempts 2`
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 72 --problem-indices 71 --target-samples 1024 --batch-size 1 --workers 4 --request-delay 3 --progress-every 128 --timeout 90 --max-job-attempts 2`
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 72 --problem-indices 71 --target-samples 2048 --batch-size 1 --workers 4 --request-delay 3 --progress-every 128 --timeout 90 --max-job-attempts 2`
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 72 --problem-indices 71 --target-samples 3072 --batch-size 1 --workers 8 --request-delay 2 --progress-every 128 --timeout 120 --max-job-attempts 3`
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 72 --problem-indices 71 --target-samples 4096 --batch-size 1 --workers 8 --request-delay 2 --progress-every 128 --timeout 120 --max-job-attempts 3`
- `python -u experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 72 --problem-indices 71 --target-samples 4096 --progress-every 1`
- `python experiments\17_maxed_out_campaign.py analyze-heldout --models 3B`
- `python experiments\17_maxed_out_campaign.py gate-report`
- `python experiments\17_maxed_out_campaign.py status`
- Notes: problem-71 started with 96 valid cached samples. The 512 and 1024 checkpoints completed cleanly; the 2048 checkpoint had one recovered internal requeue and no failed batches; the 3072 and 4096 checkpoints completed cleanly with 8 workers. The final measured record is an all-incorrect stress case with `p=0.0` (0 correct out of 4096), `kappa=null`, and ground truth `2516_8`.

Previous continuation for 3B/problem-72:
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 73 --problem-indices 72 --target-samples 512 --batch-size 1 --workers 4 --request-delay 3 --progress-every 64 --timeout 90 --max-job-attempts 2`
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 73 --problem-indices 72 --target-samples 1024 --batch-size 1 --workers 4 --request-delay 3 --progress-every 128 --timeout 90 --max-job-attempts 2`
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 73 --problem-indices 72 --target-samples 2048 --batch-size 1 --workers 4 --request-delay 3 --progress-every 128 --timeout 90 --max-job-attempts 2`
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 73 --problem-indices 72 --target-samples 3072 --batch-size 1 --workers 8 --request-delay 2 --progress-every 128 --timeout 120 --max-job-attempts 3`
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 73 --problem-indices 72 --target-samples 4096 --batch-size 1 --workers 8 --request-delay 2 --progress-every 128 --timeout 120 --max-job-attempts 3`
- `python -u experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 73 --problem-indices 72 --target-samples 4096 --progress-every 1`
- `python experiments\17_maxed_out_campaign.py analyze-heldout --models 3B`
- `python experiments\17_maxed_out_campaign.py gate-report`
- `python experiments\17_maxed_out_campaign.py status`
- Notes: problem-72 started with 64 valid cached samples. The 512 checkpoint had one recovered internal requeue and no failed batches; the 1024 and 2048 checkpoints completed cleanly; the 3072 checkpoint had one recovered internal requeue and no failed batches; the 4096 checkpoint completed cleanly. The final measured record has `p=0.918212890625` (3761 correct out of 4096), `kappa=0.6825550524431816`, and ground truth `3`.

Previous continuation for 3B/problem-73:
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 74 --problem-indices 73 --target-samples 512 --batch-size 1 --workers 4 --request-delay 3 --progress-every 64 --timeout 90 --max-job-attempts 2`
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 74 --problem-indices 73 --target-samples 1024 --batch-size 1 --workers 8 --request-delay 2 --progress-every 128 --timeout 120 --max-job-attempts 3`
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 74 --problem-indices 73 --target-samples 2048 --batch-size 1 --workers 8 --request-delay 2 --progress-every 128 --timeout 120 --max-job-attempts 3`
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 74 --problem-indices 73 --target-samples 3072 --batch-size 1 --workers 8 --request-delay 2 --progress-every 128 --timeout 120 --max-job-attempts 3`
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 74 --problem-indices 73 --target-samples 4096 --batch-size 1 --workers 8 --request-delay 2 --progress-every 128 --timeout 120 --max-job-attempts 3`
- `python -u experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 74 --problem-indices 73 --target-samples 4096 --progress-every 1`
- `python experiments\17_maxed_out_campaign.py analyze-heldout --models 3B`
- `python experiments\17_maxed_out_campaign.py gate-report`
- `python experiments\17_maxed_out_campaign.py status`
- Notes: problem-73 started with 64 valid cached samples. The 512 checkpoint completed slowly but cleanly with 4 workers; the 1024 checkpoint completed cleanly with 8 workers; the 2048 checkpoint had one recovered internal requeue and no failed batches; the 3072 checkpoint completed cleanly; the 4096 checkpoint had one recovered internal requeue and no failed batches. The final measured record has `p=0.903564453125` (3701 correct out of 4096), `kappa=0.5612673960852181`, and ground truth `\frac{3\sqrt{3}}{4}`.

Previous continuation for 3B/problem-74:
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 75 --problem-indices 74 --target-samples 512 --batch-size 1 --workers 8 --request-delay 2 --progress-every 64 --timeout 120 --max-job-attempts 3`
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 75 --problem-indices 74 --target-samples 1024 --batch-size 1 --workers 8 --request-delay 2 --progress-every 128 --timeout 120 --max-job-attempts 3`
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 75 --problem-indices 74 --target-samples 2048 --batch-size 1 --workers 8 --request-delay 2 --progress-every 128 --timeout 120 --max-job-attempts 3`
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 75 --problem-indices 74 --target-samples 3072 --batch-size 1 --workers 8 --request-delay 2 --progress-every 128 --timeout 120 --max-job-attempts 3`
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 75 --problem-indices 74 --target-samples 4096 --batch-size 1 --workers 8 --request-delay 2 --progress-every 128 --timeout 120 --max-job-attempts 3`
- `python -u experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 75 --problem-indices 74 --target-samples 4096 --progress-every 1`
- `python experiments\17_maxed_out_campaign.py analyze-heldout --models 3B`
- `python experiments\17_maxed_out_campaign.py gate-report`
- `python experiments\17_maxed_out_campaign.py status`
- Notes: problem-74 started with 96 valid cached samples. The 512, 1024, 2048, and 3072 checkpoints completed cleanly with zero failed batches; the 4096 run outlived the shell-tool timeout but the campaign process kept running and finished with zero failed batches. The final measured record has `p=0.160888671875` (659 correct out of 4096), `kappa=0.5059110818933299`, and ground truth `\cot x`.

Previous continuation for 3B/problem-75:
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 76 --problem-indices 75 --target-samples 512 --batch-size 1 --workers 8 --request-delay 2 --progress-every 64 --timeout 120 --max-job-attempts 3`
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 76 --problem-indices 75 --target-samples 1024 --batch-size 1 --workers 8 --request-delay 2 --progress-every 128 --timeout 120 --max-job-attempts 3`
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 76 --problem-indices 75 --target-samples 2048 --batch-size 1 --workers 8 --request-delay 2 --progress-every 128 --timeout 120 --max-job-attempts 3`
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 76 --problem-indices 75 --target-samples 3072 --batch-size 1 --workers 8 --request-delay 2 --progress-every 128 --timeout 120 --max-job-attempts 3`
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 76 --problem-indices 75 --target-samples 4096 --batch-size 1 --workers 8 --request-delay 2 --progress-every 128 --timeout 120 --max-job-attempts 3`
- `python -u experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 76 --problem-indices 75 --target-samples 4096 --progress-every 1`
- `python experiments\17_maxed_out_campaign.py analyze-heldout --models 3B`
- `python experiments\17_maxed_out_campaign.py gate-report`
- `python experiments\17_maxed_out_campaign.py status`
- Notes: problem-75 started with 64 valid cached samples. The 512, 1024, 2048, 3072, and 4096 checkpoints all completed cleanly with zero requeues and zero failed batches. The final measured record has `p=0.273193359375` (1119 correct out of 4096), `kappa=0.604058880970971`, and ground truth `\frac{11}{36}`.

Previous continuation for 3B/problem-76:
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 77 --problem-indices 76 --target-samples 512 --batch-size 1 --workers 8 --request-delay 2 --progress-every 64 --timeout 120 --max-job-attempts 3`
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 77 --problem-indices 76 --target-samples 1024 --batch-size 1 --workers 8 --request-delay 2 --progress-every 128 --timeout 120 --max-job-attempts 3`
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 77 --problem-indices 76 --target-samples 2048 --batch-size 1 --workers 8 --request-delay 2 --progress-every 128 --timeout 120 --max-job-attempts 3`
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 77 --problem-indices 76 --target-samples 3072 --batch-size 1 --workers 8 --request-delay 2 --progress-every 128 --timeout 120 --max-job-attempts 3`
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 77 --problem-indices 76 --target-samples 4096 --batch-size 1 --workers 8 --request-delay 2 --progress-every 128 --timeout 120 --max-job-attempts 3`
- `python -u experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 77 --problem-indices 76 --target-samples 4096 --progress-every 1`
- `python experiments\17_maxed_out_campaign.py analyze-heldout --models 3B`
- `python experiments\17_maxed_out_campaign.py gate-report`
- `python experiments\17_maxed_out_campaign.py status`
- Notes: problem-76 started with 64 valid cached samples. The 512, 2048, 3072, and 4096 checkpoints completed cleanly; the 1024 checkpoint had one recovered internal requeue and no failed batches. The final measured record has `p=0.862548828125` (3533 correct out of 4096), `kappa=0.690766932836755`, and ground truth `0`.

Previous continuation for 3B/problem-77:
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 78 --problem-indices 77 --target-samples 512 --batch-size 1 --workers 8 --request-delay 2 --progress-every 64 --timeout 120 --max-job-attempts 3`
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 78 --problem-indices 77 --target-samples 1024 --batch-size 1 --workers 8 --request-delay 2 --progress-every 128 --timeout 120 --max-job-attempts 3`
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 78 --problem-indices 77 --target-samples 2048 --batch-size 1 --workers 8 --request-delay 2 --progress-every 128 --timeout 120 --max-job-attempts 3`
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 78 --problem-indices 77 --target-samples 3072 --batch-size 1 --workers 8 --request-delay 2 --progress-every 128 --timeout 120 --max-job-attempts 3`
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 78 --problem-indices 77 --target-samples 4096 --batch-size 1 --workers 8 --request-delay 2 --progress-every 128 --timeout 120 --max-job-attempts 3`
- `python -u experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 78 --problem-indices 77 --target-samples 4096 --progress-every 1`
- `python experiments\17_maxed_out_campaign.py analyze-heldout --models 3B`
- `python experiments\17_maxed_out_campaign.py gate-report`
- `python experiments\17_maxed_out_campaign.py status`
- Notes: problem-77 started with 64 valid cached samples. The 512 checkpoint completed cleanly; the 1024 checkpoint had one recovered internal requeue and no failed batches; the 2048, 3072, and 4096 checkpoints each outlived the shell-tool timeout but the campaign processes kept running and finished with zero failed batches. The final measured record has `p=0.975830078125` (3997 correct out of 4096), `kappa=0.8014015562176683`, and ground truth `4`.

Previous continuation for 3B/problem-78:
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 79 --problem-indices 78 --target-samples 512 --batch-size 1 --workers 8 --request-delay 2 --progress-every 64 --timeout 120 --max-job-attempts 3`
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 79 --problem-indices 78 --target-samples 512 --batch-size 1 --workers 16 --request-delay 2 --progress-every 32 --timeout 120 --max-job-attempts 3`
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 79 --problem-indices 78 --target-samples 512 --batch-size 4 --workers 4 --request-delay 2 --progress-every 8 --timeout 180 --max-job-attempts 4`
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 79 --problem-indices 78 --target-samples 512 --batch-size 1 --workers 4 --request-delay 3 --progress-every 16 --timeout 300 --max-job-attempts 5`
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 79 --problem-indices 78 --target-samples 1024 --batch-size 1 --workers 4 --request-delay 3 --progress-every 32 --timeout 300 --max-job-attempts 5`
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 79 --problem-indices 78 --target-samples 2048 --batch-size 1 --workers 4 --request-delay 3 --progress-every 64 --timeout 300 --max-job-attempts 5`
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 79 --problem-indices 78 --target-samples 3072 --batch-size 1 --workers 4 --request-delay 3 --progress-every 64 --timeout 300 --max-job-attempts 5`
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 79 --problem-indices 78 --target-samples 4096 --batch-size 1 --workers 4 --request-delay 3 --progress-every 64 --timeout 300 --max-job-attempts 5`
- `python -u experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 79 --problem-indices 78 --target-samples 4096 --progress-every 1`
- `python experiments\17_maxed_out_campaign.py analyze-heldout --models 3B`
- `python experiments\17_maxed_out_campaign.py gate-report`
- `python experiments\17_maxed_out_campaign.py status`
- Notes: problem-78 started with 96 valid cached samples. The first 512 attempts using higher concurrency were slow and left missing slots, so they were stopped and resumed from the valid cache; the long-timeout single-sample profile with 4 workers filled the remaining 512 slots and then completed the 1024, 2048, 3072, and 4096 checkpoints cleanly with zero failed batches. The final measured record has `p=0.048583984375` (199 correct out of 4096), `kappa=0.6807555870190057`, and ground truth `(-2,1)`.

Most recent continuation for 3B/problem-79:
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 80 --problem-indices 79 --target-samples 512 --batch-size 1 --workers 4 --request-delay 3 --progress-every 32 --timeout 300 --max-job-attempts 5`
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 80 --problem-indices 79 --target-samples 1024 --batch-size 1 --workers 4 --request-delay 3 --progress-every 64 --timeout 300 --max-job-attempts 5`
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 80 --problem-indices 79 --target-samples 2048 --batch-size 1 --workers 4 --request-delay 3 --progress-every 64 --timeout 300 --max-job-attempts 5`
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 80 --problem-indices 79 --target-samples 2048 --batch-size 1 --workers 1 --request-delay 3 --progress-every 1 --timeout 300 --max-job-attempts 5`
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 80 --problem-indices 79 --target-samples 3072 --batch-size 1 --workers 4 --request-delay 3 --progress-every 64 --timeout 300 --max-job-attempts 5`
- `python -u experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 80 --problem-indices 79 --target-samples 4096 --batch-size 1 --workers 4 --request-delay 3 --progress-every 64 --timeout 300 --max-job-attempts 5`
- `python -u experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 80 --problem-indices 79 --target-samples 4096 --progress-every 1`
- `python experiments\17_maxed_out_campaign.py analyze-heldout --models 3B`
- `python experiments\17_maxed_out_campaign.py gate-report`
- `python experiments\17_maxed_out_campaign.py status`
- Notes: problem-79 started with 96 valid cached samples. The 512, 1024, 3072, and 4096 checkpoints completed cleanly; the 2048 checkpoint had one connection-error failed slot at sample 1255, recovered by a one-worker top-off. The final measured record has `p=0.98095703125` (4018 correct out of 4096), `kappa=0.6894934333958724`, and ground truth `2`.

Most recent continuation for 3B/problem-60:
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 61 --problem-indices 60 --target-samples 512 --batch-size 1 --workers 4 --request-delay 3 --progress-every 64 --timeout 90 --max-job-attempts 2`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 61 --problem-indices 60 --target-samples 1024 --batch-size 1 --workers 4 --request-delay 3 --progress-every 128 --timeout 90 --max-job-attempts 2`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 61 --problem-indices 60 --target-samples 1024 --batch-size 1 --workers 1 --request-delay 3 --progress-every 1 --timeout 120 --max-job-attempts 4`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 61 --problem-indices 60 --target-samples 2048 --batch-size 1 --workers 4 --request-delay 3 --progress-every 128 --timeout 90 --max-job-attempts 2`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 61 --problem-indices 60 --target-samples 2048 --batch-size 1 --workers 1 --request-delay 3 --progress-every 1 --timeout 120 --max-job-attempts 4`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 61 --problem-indices 60 --target-samples 3072 --batch-size 1 --workers 4 --request-delay 3 --progress-every 128 --timeout 90 --max-job-attempts 2`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 61 --problem-indices 60 --target-samples 3072 --batch-size 1 --workers 1 --request-delay 2 --progress-every 1 --timeout 60 --dry-run`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 61 --problem-indices 60 --target-samples 4096 --batch-size 1 --workers 4 --request-delay 3 --progress-every 128 --timeout 90 --max-job-attempts 2`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 61 --problem-indices 60 --target-samples 4096 --batch-size 1 --workers 1 --request-delay 2 --progress-every 1 --timeout 60 --dry-run`
- `python experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 61 --problem-indices 60 --target-samples 4096 --progress-every 1`
- `python experiments\17_maxed_out_campaign.py analyze-heldout --models 3B`
- `python experiments\17_maxed_out_campaign.py gate-report`
- `python experiments\17_maxed_out_campaign.py status`

Most recent continuation for 3B/problem-59:
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 60 --problem-indices 59 --target-samples 512 --batch-size 1 --workers 4 --request-delay 3 --progress-every 64 --timeout 90 --max-job-attempts 2`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 60 --problem-indices 59 --target-samples 1024 --batch-size 1 --workers 4 --request-delay 3 --progress-every 128 --timeout 90 --max-job-attempts 2`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 60 --problem-indices 59 --target-samples 1024 --batch-size 1 --workers 1 --request-delay 3 --progress-every 1 --timeout 120 --max-job-attempts 4`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 60 --problem-indices 59 --target-samples 2048 --batch-size 1 --workers 4 --request-delay 3 --progress-every 128 --timeout 90 --max-job-attempts 2`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 60 --problem-indices 59 --target-samples 2048 --batch-size 1 --workers 1 --request-delay 3 --progress-every 1 --timeout 120 --max-job-attempts 4`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 60 --problem-indices 59 --target-samples 3072 --batch-size 1 --workers 4 --request-delay 3 --progress-every 128 --timeout 90 --max-job-attempts 2`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 60 --problem-indices 59 --target-samples 4096 --batch-size 1 --workers 4 --request-delay 3 --progress-every 128 --timeout 90 --max-job-attempts 2`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 60 --problem-indices 59 --target-samples 4096 --batch-size 1 --workers 1 --request-delay 3 --progress-every 1 --timeout 120 --max-job-attempts 4`
- `python experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 60 --problem-indices 59 --target-samples 4096 --progress-every 1`
- `python experiments\17_maxed_out_campaign.py analyze-heldout --models 3B`
- `python experiments\17_maxed_out_campaign.py gate-report`
- `python experiments\17_maxed_out_campaign.py status`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 59 --problem-indices 58 --target-samples 512 --batch-size 1 --workers 4 --request-delay 3 --progress-every 16 --timeout 90 --max-job-attempts 2`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 59 --problem-indices 58 --target-samples 1024 --batch-size 1 --workers 4 --request-delay 3 --progress-every 64 --timeout 90 --max-job-attempts 2`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 59 --problem-indices 58 --target-samples 1024 --batch-size 1 --workers 1 --request-delay 3 --progress-every 1 --timeout 120 --max-job-attempts 4`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 59 --problem-indices 58 --target-samples 2048 --batch-size 1 --workers 4 --request-delay 3 --progress-every 128 --timeout 90 --max-job-attempts 2`

Most recent continuation for 3B/problem-53:
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 54 --problem-indices 53 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96 --timeout 300 --dry-run`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 54 --problem-indices 53 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 64 --timeout 180`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 54 --problem-indices 53 --target-samples 4096 --batch-size 8 --workers 1 --request-delay 2 --progress-every 1 --timeout 60 --dry-run`
- `python experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 54 --problem-indices 53 --target-samples 4096`
- `python experiments\17_maxed_out_campaign.py analyze-heldout --models 3B`
- `python experiments\17_maxed_out_campaign.py gate-report`
- `python experiments\17_maxed_out_campaign.py status`

Most recent continuation for 3B/problem-54:
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 55 --problem-indices 54 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96 --timeout 300 --dry-run`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 55 --problem-indices 54 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 64 --timeout 180`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 55 --problem-indices 54 --target-samples 4096 --batch-size 8 --workers 1 --request-delay 2 --progress-every 1 --timeout 60 --dry-run`
- `python experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 55 --problem-indices 54 --target-samples 4096`
- `python experiments\17_maxed_out_campaign.py analyze-heldout --models 3B`
- `python experiments\17_maxed_out_campaign.py gate-report`
- `python experiments\17_maxed_out_campaign.py status`

Most recent continuation for 3B/problem-55:
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 56 --problem-indices 55 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96 --timeout 300 --dry-run`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 56 --problem-indices 55 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 64 --timeout 180`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 56 --problem-indices 55 --target-samples 4096 --batch-size 8 --workers 1 --request-delay 2 --progress-every 1 --timeout 60 --dry-run`
- `python experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 56 --problem-indices 55 --target-samples 4096`
- `python experiments\17_maxed_out_campaign.py analyze-heldout --models 3B`
- `python experiments\17_maxed_out_campaign.py gate-report`
- `python experiments\17_maxed_out_campaign.py status`

Most recent continuation for 3B/problem-56:
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 57 --problem-indices 56 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96 --timeout 300 --dry-run`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 57 --problem-indices 56 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 64 --timeout 180`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 57 --problem-indices 56 --target-samples 4096 --batch-size 8 --workers 1 --request-delay 2 --progress-every 1 --timeout 60 --dry-run`
- `python experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 57 --problem-indices 56 --target-samples 4096`
- `python experiments\17_maxed_out_campaign.py analyze-heldout --models 3B`
- `python experiments\17_maxed_out_campaign.py gate-report`
- `python experiments\17_maxed_out_campaign.py status`

Most recent continuation for 3B/problem-57 and 3B/problem-58:
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 58 --problem-indices 57 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96 --timeout 300 --dry-run`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 58 --problem-indices 57 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 64 --timeout 180`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 58 --problem-indices 57 --target-samples 4096 --batch-size 8 --workers 8 --request-delay 2 --progress-every 32 --timeout 60`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 58 --problem-indices 57 --target-samples 4096 --batch-size 1 --workers 16 --request-delay 2 --progress-every 128 --timeout 90`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 58 --problem-indices 57 --target-samples 4096 --batch-size 1 --workers 4 --request-delay 3 --progress-every 16 --timeout 300`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 58 --problem-indices 57 --target-samples 4096 --batch-size 1 --workers 1 --request-delay 2 --progress-every 1 --timeout 60 --dry-run`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 59 --problem-indices 58 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96 --timeout 300 --dry-run`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 59 --problem-indices 58 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 64 --timeout 180`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 59 --problem-indices 58 --target-samples 4096 --batch-size 8 --workers 1 --request-delay 2 --progress-every 1 --timeout 60 --dry-run`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 59 --problem-indices 58 --target-samples 512 --batch-size 1 --workers 4 --request-delay 3 --progress-every 16 --timeout 90 --max-job-attempts 2`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 59 --problem-indices 58 --target-samples 1024 --batch-size 1 --workers 4 --request-delay 3 --progress-every 64 --timeout 90 --max-job-attempts 2`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 59 --problem-indices 58 --target-samples 1024 --batch-size 1 --workers 1 --request-delay 3 --progress-every 1 --timeout 120 --max-job-attempts 4`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 59 --problem-indices 58 --target-samples 2048 --batch-size 1 --workers 4 --request-delay 3 --progress-every 128 --timeout 90 --max-job-attempts 2`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 59 --problem-indices 58 --target-samples 3072 --batch-size 1 --workers 4 --request-delay 3 --progress-every 128 --timeout 90 --max-job-attempts 2`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 59 --problem-indices 58 --target-samples 4096 --batch-size 1 --workers 4 --request-delay 3 --progress-every 128 --timeout 90 --max-job-attempts 2`
- `python experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 59 --problem-indices 58 --target-samples 4096 --progress-every 1`
- `python experiments\17_maxed_out_campaign.py analyze-heldout --models 3B`
- `python experiments\17_maxed_out_campaign.py gate-report`
- `python experiments\17_maxed_out_campaign.py status`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 58 --problem-indices 57 --target-samples 88 --batch-size 1 --workers 2 --request-delay 3 --progress-every 1 --timeout 60 --max-job-attempts 2`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 58 --problem-indices 57 --target-samples 88 --batch-size 1 --workers 1 --request-delay 3 --progress-every 1 --timeout 90 --max-job-attempts 4`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 58 --problem-indices 57 --target-samples 128 --batch-size 1 --workers 4 --request-delay 3 --progress-every 4 --timeout 90 --max-job-attempts 4`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 58 --problem-indices 57 --target-samples 256 --batch-size 1 --workers 4 --request-delay 3 --progress-every 16 --timeout 90 --max-job-attempts 4`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 58 --problem-indices 57 --target-samples 512 --batch-size 1 --workers 4 --request-delay 3 --progress-every 32 --timeout 90 --max-job-attempts 4`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 58 --problem-indices 57 --target-samples 1024 --batch-size 1 --workers 4 --request-delay 3 --progress-every 64 --timeout 90 --max-job-attempts 4`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 58 --problem-indices 57 --target-samples 2048 --batch-size 1 --workers 4 --request-delay 3 --progress-every 128 --timeout 90 --max-job-attempts 4`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 58 --problem-indices 57 --target-samples 1728 --batch-size 1 --workers 4 --request-delay 3 --progress-every 4 --timeout 90 --max-job-attempts 2`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 58 --problem-indices 57 --target-samples 1792 --batch-size 1 --workers 4 --request-delay 3 --progress-every 8 --timeout 90 --max-job-attempts 2`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 58 --problem-indices 57 --target-samples 2048 --batch-size 1 --workers 4 --request-delay 3 --progress-every 32 --timeout 90 --max-job-attempts 2`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 58 --problem-indices 57 --target-samples 3072 --batch-size 1 --workers 4 --request-delay 3 --progress-every 128 --timeout 90 --max-job-attempts 2`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 58 --problem-indices 57 --target-samples 3072 --batch-size 1 --workers 1 --request-delay 3 --progress-every 1 --timeout 120 --max-job-attempts 4`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 58 --problem-indices 57 --target-samples 4096 --batch-size 1 --workers 4 --request-delay 3 --progress-every 128 --timeout 90 --max-job-attempts 2`
- `python experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 58 --problem-indices 57 --target-samples 4096`
- `python experiments\17_maxed_out_campaign.py analyze-heldout --models 3B`
- `python experiments\17_maxed_out_campaign.py gate-report`
- `python experiments\17_maxed_out_campaign.py status`

Passed in the maxed-out campaign setup session:
- `python -m py_compile experiments\17_maxed_out_campaign.py`
- `python experiments\17_maxed_out_campaign.py prepare --smoke --force`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 3 --problem-indices 0-2 --target-samples 8 --dry-run`
- `python experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 3 --problem-indices 0-2 --target-samples 8`
- `python experiments\17_maxed_out_campaign.py measure-math --model 70B --n-problems 3 --problem-indices 0-2 --target-samples 8`
- `python experiments\17_maxed_out_campaign.py analyze-heldout --models 3B 70B --smoke`
- `python experiments\17_maxed_out_campaign.py prepare --force`
- `python experiments\17_maxed_out_campaign.py status`
- `python experiments\17_maxed_out_campaign.py status --smoke`
- `python experiments\17_maxed_out_campaign.py gate-report`
- `python experiments\17_maxed_out_campaign.py gate-report --smoke`
- `python experiments\17_maxed_out_campaign.py analyze-expanded-proxy --seeds 5`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 1 --problem-indices 0 --target-samples 128 --batch-size 8 --workers 1 --request-delay 2 --progress-every 1`
- `python experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 1 --problem-indices 0 --target-samples 128`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 1 --problem-indices 0 --target-samples 512 --batch-size 8 --workers 4 --request-delay 2 --progress-every 4`
- `python experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 1 --problem-indices 0 --target-samples 512`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 1 --problem-indices 0 --target-samples 1024 --batch-size 8 --workers 4 --request-delay 2 --progress-every 8`
- `python experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 1 --problem-indices 0 --target-samples 1024`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 1 --problem-indices 0 --target-samples 2048 --batch-size 8 --workers 4 --request-delay 2 --progress-every 16`
- `python experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 1 --problem-indices 0 --target-samples 2048`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 1 --problem-indices 0 --target-samples 4096 --batch-size 8 --workers 8 --request-delay 2 --progress-every 32`
- `python experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 1 --problem-indices 0 --target-samples 4096`
- `python experiments\17_maxed_out_campaign.py analyze-heldout --models 3B`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 2 --problem-indices 1 --target-samples 1024 --batch-size 8 --workers 8 --request-delay 2 --progress-every 16`
- `python experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 2 --problem-indices 1 --target-samples 1024`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 2 --problem-indices 1 --target-samples 2048 --batch-size 8 --workers 8 --request-delay 2 --progress-every 16`
- `python experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 2 --problem-indices 1 --target-samples 2048`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 2 --problem-indices 1 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 32`
- `python experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 2 --problem-indices 1 --target-samples 4096`
- `python experiments\17_maxed_out_campaign.py analyze-heldout --models 3B`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 3 --problem-indices 2 --target-samples 1024 --batch-size 8 --workers 16 --request-delay 2 --progress-every 16`
- `python experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 3 --problem-indices 2 --target-samples 1024`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 3 --problem-indices 2 --target-samples 2048 --batch-size 8 --workers 16 --request-delay 2 --progress-every 32`
- `python experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 3 --problem-indices 2 --target-samples 2048`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 3 --problem-indices 2 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 64`
- `python experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 3 --problem-indices 2 --target-samples 4096`
- `python experiments\17_maxed_out_campaign.py analyze-heldout --models 3B`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 4 --problem-indices 3 --target-samples 1024 --batch-size 8 --workers 16 --request-delay 2 --progress-every 32`
- `python experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 4 --problem-indices 3 --target-samples 1024`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 4 --problem-indices 3 --target-samples 2048 --batch-size 8 --workers 16 --request-delay 2 --progress-every 32`
- `python experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 4 --problem-indices 3 --target-samples 2048`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 4 --problem-indices 3 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 64`
- `python experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 4 --problem-indices 3 --target-samples 4096`
- `python experiments\17_maxed_out_campaign.py analyze-heldout --models 3B`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 5 --problem-indices 4 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96`
- `python experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 5 --problem-indices 4 --target-samples 4096`
- `python experiments\17_maxed_out_campaign.py analyze-heldout --models 3B`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 6 --problem-indices 5 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96`
- `python experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 6 --problem-indices 5 --target-samples 4096`
- `python experiments\17_maxed_out_campaign.py analyze-heldout --models 3B`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 7 --problem-indices 6 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96`
- `python experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 7 --problem-indices 6 --target-samples 4096`
- `python experiments\17_maxed_out_campaign.py analyze-heldout --models 3B`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 8 --problem-indices 7 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96`
- `python experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 8 --problem-indices 7 --target-samples 4096`
- `python experiments\17_maxed_out_campaign.py analyze-heldout --models 3B`
- `python experiments\17_maxed_out_campaign.py gate-report`
- `python experiments\17_maxed_out_campaign.py status`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 12 --problem-indices 11 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96 --dry-run`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 12 --problem-indices 11 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96`
- `python experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 12 --problem-indices 11 --target-samples 4096`
- `python experiments\17_maxed_out_campaign.py analyze-heldout --models 3B`
- `python experiments\17_maxed_out_campaign.py gate-report`
- `python experiments\17_maxed_out_campaign.py status`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 13 --problem-indices 12 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96 --dry-run`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 13 --problem-indices 12 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96`
- `python experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 13 --problem-indices 12 --target-samples 4096`
- `python experiments\17_maxed_out_campaign.py analyze-heldout --models 3B`
- `python experiments\17_maxed_out_campaign.py gate-report`
- `python experiments\17_maxed_out_campaign.py status`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 14 --problem-indices 13 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96 --dry-run`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 14 --problem-indices 13 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96`
- `python experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 14 --problem-indices 13 --target-samples 4096`
- `python experiments\17_maxed_out_campaign.py analyze-heldout --models 3B`
- `python experiments\17_maxed_out_campaign.py gate-report`
- `python experiments\17_maxed_out_campaign.py status`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 15 --problem-indices 14 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96 --dry-run`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 15 --problem-indices 14 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96`
- `python experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 15 --problem-indices 14 --target-samples 4096`
- `python experiments\17_maxed_out_campaign.py analyze-heldout --models 3B`
- `python experiments\17_maxed_out_campaign.py gate-report`
- `python experiments\17_maxed_out_campaign.py status`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 16 --problem-indices 15 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96 --dry-run`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 16 --problem-indices 15 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96`
- `python experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 16 --problem-indices 15 --target-samples 4096`
- `python experiments\17_maxed_out_campaign.py analyze-heldout --models 3B`
- `python experiments\17_maxed_out_campaign.py gate-report`
- `python experiments\17_maxed_out_campaign.py status`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 17 --problem-indices 16 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96 --dry-run`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 17 --problem-indices 16 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96`
- `python experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 17 --problem-indices 16 --target-samples 4096`
- `python experiments\17_maxed_out_campaign.py analyze-heldout --models 3B`
- `python experiments\17_maxed_out_campaign.py gate-report`
- `python experiments\17_maxed_out_campaign.py status`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 18 --problem-indices 17 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96 --dry-run`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 18 --problem-indices 17 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96`
- `python experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 18 --problem-indices 17 --target-samples 4096`
- `python experiments\17_maxed_out_campaign.py analyze-heldout --models 3B`
- `python experiments\17_maxed_out_campaign.py gate-report`
- `python experiments\17_maxed_out_campaign.py status`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 19 --problem-indices 18 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96 --dry-run`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 19 --problem-indices 18 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96`
- `python experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 19 --problem-indices 18 --target-samples 4096`
- `python experiments\17_maxed_out_campaign.py analyze-heldout --models 3B`
- `python experiments\17_maxed_out_campaign.py gate-report`
- `python experiments\17_maxed_out_campaign.py status`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 20 --problem-indices 19 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96 --dry-run`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 20 --problem-indices 19 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96`
- `python experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 20 --problem-indices 19 --target-samples 4096`
- `python experiments\17_maxed_out_campaign.py analyze-heldout --models 3B`
- `python experiments\17_maxed_out_campaign.py gate-report`
- `python experiments\17_maxed_out_campaign.py status`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 21 --problem-indices 20 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96 --dry-run`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 21 --problem-indices 20 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96`
- `python experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 21 --problem-indices 20 --target-samples 4096`
- `python experiments\17_maxed_out_campaign.py analyze-heldout --models 3B`
- `python experiments\17_maxed_out_campaign.py gate-report`
- `python experiments\17_maxed_out_campaign.py status`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 22 --problem-indices 21 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96 --dry-run`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 22 --problem-indices 21 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 22 --problem-indices 21 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96 --dry-run`
- `python experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 22 --problem-indices 21 --target-samples 4096`
- `python experiments\17_maxed_out_campaign.py analyze-heldout --models 3B`
- `python experiments\17_maxed_out_campaign.py gate-report`
- `python experiments\17_maxed_out_campaign.py status`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 23 --problem-indices 22 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96 --dry-run`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 23 --problem-indices 22 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 23 --problem-indices 22 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96 --dry-run`
- `python experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 23 --problem-indices 22 --target-samples 4096`
- `python experiments\17_maxed_out_campaign.py analyze-heldout --models 3B`
- `python experiments\17_maxed_out_campaign.py gate-report`
- `python experiments\17_maxed_out_campaign.py status`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 24 --problem-indices 23 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96 --dry-run`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 24 --problem-indices 23 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 24 --problem-indices 23 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96 --dry-run`
- `python experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 24 --problem-indices 23 --target-samples 4096`
- `python experiments\17_maxed_out_campaign.py analyze-heldout --models 3B`
- `python experiments\17_maxed_out_campaign.py gate-report`
- `python experiments\17_maxed_out_campaign.py status`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 25 --problem-indices 24 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96 --dry-run`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 25 --problem-indices 24 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 25 --problem-indices 24 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96 --dry-run`
- `python experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 25 --problem-indices 24 --target-samples 4096`
- `python experiments\17_maxed_out_campaign.py analyze-heldout --models 3B`
- `python experiments\17_maxed_out_campaign.py gate-report`
- `python experiments\17_maxed_out_campaign.py status`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 26 --problem-indices 25 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96 --dry-run`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 26 --problem-indices 25 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 26 --problem-indices 25 --target-samples 4096 --batch-size 8 --workers 8 --request-delay 2 --progress-every 1 --dry-run`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 26 --problem-indices 25 --target-samples 4096 --batch-size 8 --workers 8 --request-delay 2 --progress-every 1`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 26 --problem-indices 25 --target-samples 4096 --batch-size 8 --workers 8 --request-delay 2 --progress-every 1 --dry-run`
- `python experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 26 --problem-indices 25 --target-samples 4096`
- `python experiments\17_maxed_out_campaign.py analyze-heldout --models 3B`
- `python experiments\17_maxed_out_campaign.py gate-report`
- `python experiments\17_maxed_out_campaign.py status`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 27 --problem-indices 26 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96 --dry-run`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 27 --problem-indices 26 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 27 --problem-indices 26 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96 --dry-run`
- `python experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 27 --problem-indices 26 --target-samples 4096`
- `python experiments\17_maxed_out_campaign.py analyze-heldout --models 3B`
- `python experiments\17_maxed_out_campaign.py gate-report`
- `python experiments\17_maxed_out_campaign.py status`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 28 --problem-indices 27 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96 --dry-run`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 28 --problem-indices 27 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 28 --problem-indices 27 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96 --dry-run`
- `python experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 28 --problem-indices 27 --target-samples 4096`
- `python experiments\17_maxed_out_campaign.py analyze-heldout --models 3B`
- `python experiments\17_maxed_out_campaign.py gate-report`
- `python experiments\17_maxed_out_campaign.py status`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 29 --problem-indices 28 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96 --dry-run`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 29 --problem-indices 28 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 29 --problem-indices 28 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96 --dry-run`
- `python experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 29 --problem-indices 28 --target-samples 4096`
- `python experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 29 --problem-indices 28 --target-samples 4096`
- `python experiments\17_maxed_out_campaign.py analyze-heldout --models 3B`
- `python experiments\17_maxed_out_campaign.py gate-report`
- `python experiments\17_maxed_out_campaign.py status`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 30 --problem-indices 29 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96 --dry-run`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 30 --problem-indices 29 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96`
- `python -m py_compile experiments\17_maxed_out_campaign.py`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 30 --problem-indices 29 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 16 --timeout 300 --dry-run`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 30 --problem-indices 29 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 16 --timeout 300`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 30 --problem-indices 29 --target-samples 4096 --batch-size 1 --workers 8 --request-delay 2 --progress-every 1 --timeout 180`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 30 --problem-indices 29 --target-samples 4096 --batch-size 8 --workers 1 --request-delay 2 --progress-every 1 --timeout 180 --dry-run`
- `python experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 30 --problem-indices 29 --target-samples 4096`
- `python experiments\17_maxed_out_campaign.py analyze-heldout --models 3B`
- `python experiments\17_maxed_out_campaign.py gate-report`
- `python experiments\17_maxed_out_campaign.py status`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 31 --problem-indices 30 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96 --timeout 300 --dry-run`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 31 --problem-indices 30 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96 --timeout 300`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 31 --problem-indices 30 --target-samples 4096 --batch-size 8 --workers 1 --request-delay 2 --progress-every 1 --timeout 300 --dry-run`
- `python experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 31 --problem-indices 30 --target-samples 4096`
- `python experiments\17_maxed_out_campaign.py analyze-heldout --models 3B`
- `python experiments\17_maxed_out_campaign.py gate-report`
- `python experiments\17_maxed_out_campaign.py status`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 32 --problem-indices 31 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96 --timeout 300 --dry-run`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 32 --problem-indices 31 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96 --timeout 300`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 32 --problem-indices 31 --target-samples 4096 --batch-size 8 --workers 1 --request-delay 2 --progress-every 1 --timeout 300 --dry-run`
- `python experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 32 --problem-indices 31 --target-samples 4096`
- `python experiments\17_maxed_out_campaign.py analyze-heldout --models 3B`
- `python experiments\17_maxed_out_campaign.py gate-report`
- `python experiments\17_maxed_out_campaign.py status`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 33 --problem-indices 32 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96 --timeout 300 --dry-run`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 33 --problem-indices 32 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96 --timeout 300`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 33 --problem-indices 32 --target-samples 4096 --batch-size 8 --workers 1 --request-delay 2 --progress-every 1 --timeout 300 --dry-run`
- `python experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 33 --problem-indices 32 --target-samples 4096`
- `python experiments\17_maxed_out_campaign.py analyze-heldout --models 3B`
- `python experiments\17_maxed_out_campaign.py gate-report`
- `python experiments\17_maxed_out_campaign.py status`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 34 --problem-indices 33 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96 --timeout 300 --dry-run`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 34 --problem-indices 33 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96 --timeout 300`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 34 --problem-indices 33 --target-samples 4096 --batch-size 8 --workers 1 --request-delay 2 --progress-every 1 --timeout 300 --dry-run`
- `python experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 34 --problem-indices 33 --target-samples 4096`
- `python experiments\17_maxed_out_campaign.py analyze-heldout --models 3B`
- `python experiments\17_maxed_out_campaign.py gate-report`
- `python experiments\17_maxed_out_campaign.py status`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 35 --problem-indices 34 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96 --timeout 300 --dry-run`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 35 --problem-indices 34 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96 --timeout 300`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 35 --problem-indices 34 --target-samples 4096 --batch-size 8 --workers 1 --request-delay 2 --progress-every 1 --timeout 300 --dry-run`
- `python experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 35 --problem-indices 34 --target-samples 4096`
- `python experiments\17_maxed_out_campaign.py analyze-heldout --models 3B`
- `python experiments\17_maxed_out_campaign.py gate-report`
- `python experiments\17_maxed_out_campaign.py status`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 36 --problem-indices 35 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96 --timeout 300 --dry-run`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 36 --problem-indices 35 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96 --timeout 300`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 36 --problem-indices 35 --target-samples 4096 --batch-size 8 --workers 1 --request-delay 2 --progress-every 1 --timeout 300 --dry-run`
- `python experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 36 --problem-indices 35 --target-samples 4096`
- `python experiments\17_maxed_out_campaign.py analyze-heldout --models 3B`
- `python experiments\17_maxed_out_campaign.py gate-report`
- `python experiments\17_maxed_out_campaign.py status`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 37 --problem-indices 36 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96 --timeout 300 --dry-run`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 37 --problem-indices 36 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96 --timeout 300`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 37 --problem-indices 36 --target-samples 4096 --batch-size 8 --workers 1 --request-delay 2 --progress-every 1 --timeout 300 --dry-run`
- `python experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 37 --problem-indices 36 --target-samples 4096`
- `python experiments\17_maxed_out_campaign.py analyze-heldout --models 3B`
- `python experiments\17_maxed_out_campaign.py gate-report`
- `python experiments\17_maxed_out_campaign.py status`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 38 --problem-indices 37 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96 --timeout 300 --dry-run`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 38 --problem-indices 37 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96 --timeout 300`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 38 --problem-indices 37 --target-samples 4096 --batch-size 8 --workers 1 --request-delay 2 --progress-every 1 --timeout 300 --dry-run`
- `python experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 38 --problem-indices 37 --target-samples 4096`
- `python experiments\17_maxed_out_campaign.py analyze-heldout --models 3B`
- `python experiments\17_maxed_out_campaign.py gate-report`
- `python experiments\17_maxed_out_campaign.py status`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 39 --problem-indices 38 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96 --timeout 300 --dry-run`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 39 --problem-indices 38 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96 --timeout 300`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 39 --problem-indices 38 --target-samples 4096 --batch-size 8 --workers 1 --request-delay 2 --progress-every 1 --timeout 300 --dry-run`
- `python experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 39 --problem-indices 38 --target-samples 4096`
- `python experiments\17_maxed_out_campaign.py analyze-heldout --models 3B`
- `python experiments\17_maxed_out_campaign.py gate-report`
- `python experiments\17_maxed_out_campaign.py status`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 40 --problem-indices 39 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96 --timeout 300 --dry-run`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 40 --problem-indices 39 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96 --timeout 300`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 40 --problem-indices 39 --target-samples 4096 --batch-size 8 --workers 1 --request-delay 2 --progress-every 1 --timeout 300 --dry-run`
- `python experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 40 --problem-indices 39 --target-samples 4096`
- `python experiments\17_maxed_out_campaign.py analyze-heldout --models 3B`
- `python experiments\17_maxed_out_campaign.py gate-report`
- `python experiments\17_maxed_out_campaign.py status`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 41 --problem-indices 40 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96 --timeout 300 --dry-run`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 41 --problem-indices 40 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96 --timeout 300`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 41 --problem-indices 40 --target-samples 4096 --batch-size 8 --workers 1 --request-delay 2 --progress-every 1 --timeout 300 --dry-run`
- `python experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 41 --problem-indices 40 --target-samples 4096`
- `python experiments\17_maxed_out_campaign.py analyze-heldout --models 3B`
- `python experiments\17_maxed_out_campaign.py gate-report`
- `python experiments\17_maxed_out_campaign.py status`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 42 --problem-indices 41 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96 --timeout 300 --dry-run`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 42 --problem-indices 41 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96 --timeout 300`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 42 --problem-indices 41 --target-samples 4096 --batch-size 8 --workers 1 --request-delay 2 --progress-every 1 --timeout 300 --dry-run`
- `python experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 42 --problem-indices 41 --target-samples 4096`
- `python experiments\17_maxed_out_campaign.py analyze-heldout --models 3B`
- `python experiments\17_maxed_out_campaign.py gate-report`
- `python experiments\17_maxed_out_campaign.py status`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 43 --problem-indices 42 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96 --timeout 300 --dry-run`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 43 --problem-indices 42 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96 --timeout 300`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 43 --problem-indices 42 --target-samples 4096 --batch-size 8 --workers 1 --request-delay 2 --progress-every 1 --timeout 300 --dry-run`
- `python experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 43 --problem-indices 42 --target-samples 4096`
- `python experiments\17_maxed_out_campaign.py analyze-heldout --models 3B`
- `python experiments\17_maxed_out_campaign.py gate-report`
- `python experiments\17_maxed_out_campaign.py status`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 44 --problem-indices 43 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96 --timeout 300 --dry-run`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 44 --problem-indices 43 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96 --timeout 300`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 44 --problem-indices 43 --target-samples 4096 --batch-size 8 --workers 1 --request-delay 2 --progress-every 1 --timeout 300 --dry-run`
- `python experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 44 --problem-indices 43 --target-samples 4096`
- `python experiments\17_maxed_out_campaign.py analyze-heldout --models 3B`
- `python experiments\17_maxed_out_campaign.py gate-report`
- `python experiments\17_maxed_out_campaign.py status`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 45 --problem-indices 44 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96 --timeout 300 --dry-run`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 45 --problem-indices 44 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96 --timeout 300`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 45 --problem-indices 44 --target-samples 4096 --batch-size 8 --workers 1 --request-delay 2 --progress-every 1 --timeout 300 --dry-run`
- `python experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 45 --problem-indices 44 --target-samples 4096`
- `python experiments\17_maxed_out_campaign.py analyze-heldout --models 3B`
- `python experiments\17_maxed_out_campaign.py gate-report`
- `python experiments\17_maxed_out_campaign.py status`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 46 --problem-indices 45 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96 --timeout 300 --dry-run`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 46 --problem-indices 45 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96 --timeout 300`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 46 --problem-indices 45 --target-samples 4096 --batch-size 8 --workers 1 --request-delay 2 --progress-every 1 --timeout 300 --dry-run`
- `python experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 46 --problem-indices 45 --target-samples 4096`
- `python experiments\17_maxed_out_campaign.py analyze-heldout --models 3B`
- `python experiments\17_maxed_out_campaign.py gate-report`
- `python experiments\17_maxed_out_campaign.py status`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 47 --problem-indices 46 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96 --timeout 300 --dry-run`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 47 --problem-indices 46 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96 --timeout 300`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 47 --problem-indices 46 --target-samples 4096 --batch-size 8 --workers 1 --request-delay 2 --progress-every 1 --timeout 300 --dry-run`
- `python experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 47 --problem-indices 46 --target-samples 4096`
- `python experiments\17_maxed_out_campaign.py analyze-heldout --models 3B`
- `python experiments\17_maxed_out_campaign.py gate-report`
- `python experiments\17_maxed_out_campaign.py status`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 48 --problem-indices 47 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96 --timeout 300 --dry-run`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 48 --problem-indices 47 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96 --timeout 300`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 48 --problem-indices 47 --target-samples 4096 --batch-size 8 --workers 1 --request-delay 2 --progress-every 1 --timeout 300 --dry-run`
- `python experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 48 --problem-indices 47 --target-samples 4096`
- `python experiments\17_maxed_out_campaign.py analyze-heldout --models 3B`
- `python experiments\17_maxed_out_campaign.py gate-report`
- `python experiments\17_maxed_out_campaign.py status`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 49 --problem-indices 48 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96 --timeout 300 --dry-run`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 49 --problem-indices 48 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96 --timeout 300`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 49 --problem-indices 48 --target-samples 4096 --batch-size 8 --workers 1 --request-delay 2 --progress-every 1 --timeout 300 --dry-run`
- `python experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 49 --problem-indices 48 --target-samples 4096`
- `python experiments\17_maxed_out_campaign.py analyze-heldout --models 3B`
- `python experiments\17_maxed_out_campaign.py gate-report`
- `python experiments\17_maxed_out_campaign.py status`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 50 --problem-indices 49 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96 --timeout 300 --dry-run`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 50 --problem-indices 49 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96 --timeout 300`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 50 --problem-indices 49 --target-samples 4096 --batch-size 8 --workers 1 --request-delay 2 --progress-every 1 --timeout 300 --dry-run`
- `python experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 50 --problem-indices 49 --target-samples 4096`
- `python experiments\17_maxed_out_campaign.py analyze-heldout --models 3B`
- `python experiments\17_maxed_out_campaign.py gate-report`
- `python experiments\17_maxed_out_campaign.py status`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 51 --problem-indices 50 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96 --timeout 300 --dry-run`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 51 --problem-indices 50 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96 --timeout 300`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 51 --problem-indices 50 --target-samples 4096 --batch-size 8 --workers 1 --request-delay 2 --progress-every 1 --timeout 300 --dry-run`
- `python experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 51 --problem-indices 50 --target-samples 4096`
- `python experiments\17_maxed_out_campaign.py analyze-heldout --models 3B`
- `python experiments\17_maxed_out_campaign.py gate-report`
- `python experiments\17_maxed_out_campaign.py status`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 11 --problem-indices 10 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96 --dry-run`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 11 --problem-indices 10 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96`
- `python experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 11 --problem-indices 10 --target-samples 4096`
- `python experiments\17_maxed_out_campaign.py analyze-heldout --models 3B`
- `python experiments\17_maxed_out_campaign.py gate-report`
- `python experiments\17_maxed_out_campaign.py status`
- `python -m py_compile experiments\17_maxed_out_campaign.py`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 9 --problem-indices 8 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96 --dry-run`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 9 --problem-indices 8 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96`
- `python experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 9 --problem-indices 8 --target-samples 4096`
- `python experiments\17_maxed_out_campaign.py analyze-heldout --models 3B`
- `python experiments\17_maxed_out_campaign.py gate-report`
- `python experiments\17_maxed_out_campaign.py status`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 10 --problem-indices 9 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96 --dry-run`
- `python experiments\17_maxed_out_campaign.py collect-math --model 3B --n-problems 10 --problem-indices 9 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 96`
- `python experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 10 --problem-indices 9 --target-samples 4096`
- `python experiments\17_maxed_out_campaign.py analyze-heldout --models 3B`
- `python experiments\17_maxed_out_campaign.py gate-report`
- `python experiments\17_maxed_out_campaign.py status`
- `python -m py_compile config.py experiments\15_paper_claims_status.py experiments\16_cross_benchmark.py experiments\17_maxed_out_campaign.py`
- `git diff --check`
- import smoke for `experiments\16_cross_benchmark.py` MATH500 loader and boxed-answer grader
- import smoke for MBPP-backed `humaneval_mbpp` loader and canonical-code public-test grader

Previous continuation for 3B/problem-80:
- Initial cache audit found 256 valid samples; checkpoints to 512, 1024, and 2048 completed cleanly with `--batch-size 1 --workers 4 --request-delay 3 --timeout 300 --max-job-attempts 5`.
- The 3072 checkpoint initially wrote 976/1024 new samples and left 48 failed slots after provider `DEGRADED function cannot be invoked` errors. `experiments\17_maxed_out_campaign.py` now treats `degraded` as retryable; a patched top-off filled the exact 48 missing slots with zero failures.
- The 4096 checkpoint added the final 1024 samples with zero failed or requeued batches.
- Verified cache completeness: `4096/4096` valid samples and zero missing slots for 3B/problem-80.
- `python -u experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 81 --problem-indices 80 --target-samples 4096 --progress-every 1`
- `python experiments\17_maxed_out_campaign.py analyze-heldout --models 3B`
- `python experiments\17_maxed_out_campaign.py gate-report`
- `python experiments\17_maxed_out_campaign.py status`
- Notes: final measured record has `p=0.644775390625` (2641 correct out of 4096), `kappa=0.5069161816504474`, level `5`, and ground truth `501`. The locked `K=128, N=8` calibration-selected estimator row now has MAE `0.006464931290937331` over 81 rows.

Previous continuation for 3B/problem-81:
- Initial cache audit found 64 valid samples. Checkpoints to 512, 1024, 2048, and 3072 completed cleanly with `--batch-size 1 --workers 4 --request-delay 3 --timeout 300 --max-job-attempts 5`.
- The 4096 checkpoint wrote 1023/1024 new samples and left one failed slot, sample `4055`, after a provider `500` inference-connection error. `experiments\17_maxed_out_campaign.py` now treats provider 500/internal-server/inference-connection errors as retryable; a one-worker top-off filled the exact missing slot with zero failures.
- Verified cache completeness: `4096/4096` valid samples and zero missing slots for 3B/problem-81.
- `python -u experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 82 --problem-indices 81 --target-samples 4096 --progress-every 1`
- `python experiments\17_maxed_out_campaign.py analyze-heldout --models 3B`
- `python experiments\17_maxed_out_campaign.py gate-report`
- `python experiments\17_maxed_out_campaign.py status`
- Notes: final measured record has `p=0.525146484375` (2151 correct out of 4096), `kappa=0.624971227587097`, level `3`, and ground truth `3`. The locked `K=128, N=8` calibration-selected estimator row now has MAE `0.006420352336630348` over 82 rows.

Most recent continuation for 3B/problem-82:
- Initial cache audit found 64 valid samples. Checkpoints to 512, 1024, 2048, and 4096 completed cleanly with `--batch-size 1 --workers 4 --request-delay 3 --timeout 300 --max-job-attempts 5`.
- The 3072 checkpoint completed with one internal recovered requeue and zero failed batches; the final cache audit verified `4096/4096` valid samples and zero missing slots for 3B/problem-82.
- `python -u experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 83 --problem-indices 82 --target-samples 4096 --progress-every 1`
- `python experiments\17_maxed_out_campaign.py analyze-heldout --models 3B`
- `python experiments\17_maxed_out_campaign.py gate-report`
- `python experiments\17_maxed_out_campaign.py status`
- Notes: final measured record is a low-p hard-tail case with `p=0.024169921875` (99 correct out of 4096), `kappa=0.5942386082491161`, level `5`, and ground truth `\frac{3}{2}`. The locked `K=128, N=8` calibration-selected estimator row now has MAE `0.006401411840366082` over 83 rows.

Most recent continuation for 3B/problem-83:
- Initial cache audit found 95 valid samples. Checkpoints to 512, 1024, and 2048 completed cleanly with `--batch-size 1 --workers 4 --request-delay 3 --timeout 300 --max-job-attempts 5`.
- The 3072 checkpoint completed with one recovered requeue and zero failed batches; the 4096 checkpoint completed with zero requeues and zero failed batches. The final cache audit verified `4096/4096` valid samples and zero missing slots for 3B/problem-83.
- `python -u experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 84 --problem-indices 83 --target-samples 4096 --progress-every 1`
- `python experiments\17_maxed_out_campaign.py analyze-heldout --models 3B`
- `python experiments\17_maxed_out_campaign.py gate-report`
- `python experiments\17_maxed_out_campaign.py status`
- Notes: final measured record has `p=0.146728515625` (601 correct out of 4096), `kappa=0.39237846317177616`, level `2`, and ground truth `2`. The locked `K=128, N=8` calibration-selected estimator row now has MAE `0.0064441492498823446` over 84 rows.

Most recent continuation for 3B/problem-84:
- Initial cache audit found 64 valid samples. Checkpoints to 512, 2048, and 3072 completed with zero requeues and zero failed batches; the 1024 and 4096 checkpoints each had one recovered requeue and zero failed batches. The final cache audit verified `4096/4096` valid samples and zero missing slots for 3B/problem-84.
- `python -u experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 85 --problem-indices 84 --target-samples 4096 --progress-every 1`
- `python experiments\17_maxed_out_campaign.py analyze-heldout --models 3B`
- `python experiments\17_maxed_out_campaign.py gate-report`
- `python experiments\17_maxed_out_campaign.py status`
- Notes: final measured record has `p=0.436767578125` (1789 correct out of 4096), `kappa=0.5998764786879701`, level `4`, and ground truth `-1`. The locked `K=128, N=8` calibration-selected estimator row now has MAE `0.006404007330684015` over 85 rows.

Most recent continuation for 3B/problem-85:
- Initial cache audit found 64 valid samples. Checkpoints to 512, 1024, 2048, 3072, and 4096 completed with zero requeues and zero failed batches using the long-timeout 4-worker profile. The final cache audit verified `4096/4096` valid samples and zero missing slots for 3B/problem-85.
- `python -u experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 86 --problem-indices 85 --target-samples 4096 --progress-every 1`
- `python experiments\17_maxed_out_campaign.py analyze-heldout --models 3B`
- `python experiments\17_maxed_out_campaign.py gate-report`
- `python experiments\17_maxed_out_campaign.py status`
- Notes: final measured record has `p=0.82861328125` (3394 correct out of 4096), `kappa=0.5009603003120976`, level `4`, and ground truth `\sqrt{5}`. The locked `K=128, N=8` calibration-selected estimator row now has MAE `0.006344296445705742` over 86 rows.

Most recent continuation for 3B/problem-86:
- Initial cache audit found 64 valid samples. Checkpoints to 512, 1024, 2048, 3072, and 4096 completed with zero requeues and zero failed batches using the long-timeout 4-worker profile. The final cache audit verified `4096/4096` valid samples and zero missing slots for 3B/problem-86.
- `python -u experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 87 --problem-indices 86 --target-samples 4096 --progress-every 1`
- `python experiments\17_maxed_out_campaign.py analyze-heldout --models 3B`
- `python experiments\17_maxed_out_campaign.py gate-report`
- `python experiments\17_maxed_out_campaign.py status`
- Notes: final measured record has `p=0.9541015625` (3908 correct out of 4096), `kappa=0.5963204229186175`, level `1`, and ground truth `240`. The locked `K=128, N=8` calibration-selected estimator row now has MAE `0.0062751419627296405` over 87 rows.

Most recent continuation for 3B/problem-87:
- Initial cache audit found 80 valid samples. The 512, 1024, and 4096 checkpoints completed with zero requeues and zero failed batches; the 2048 checkpoint had two recovered internal requeues and zero failed batches; the 3072 checkpoint had one recovered internal requeue and zero failed batches. The final cache audit verified `4096/4096` valid samples and zero missing slots for 3B/problem-87.
- `python -u experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 88 --problem-indices 87 --target-samples 4096 --progress-every 1`
- `python experiments\17_maxed_out_campaign.py analyze-heldout --models 3B`
- `python experiments\17_maxed_out_campaign.py gate-report`
- `python experiments\17_maxed_out_campaign.py status`
- Notes: final measured record has `p=0.96923828125` (3970 correct out of 4096), `kappa=0.5190596137699413`, level `1`, and ground truth `1`. The locked `K=128, N=8` calibration-selected estimator row now has MAE `0.006219258412983198` over 88 rows.

Most recent continuation for 3B/problem-88:
- Initial cache audit found 64 valid samples. The 512 and 1024 checkpoints completed cleanly; the first 2048 attempt stalled at `1258/2048` valid samples and repeated provider 504/timeouts around slot `1256`.
- After 3B/problem-89 confirmed provider health, a one-slot retry filled slot `1256`; the resumed 2048 checkpoint then completed cleanly with zero failed batches. The 3072 checkpoint completed with one recovered requeue and zero failed batches, and the 4096 checkpoint completed with one recovered requeue and zero failed batches.
- Final cache audit verified `4096/4096` valid samples and zero missing slots for 3B/problem-88.
- `python -u experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 90 --problem-indices 88 --target-samples 4096 --progress-every 1`
- `python experiments\17_maxed_out_campaign.py analyze-heldout --models 3B`
- `python experiments\17_maxed_out_campaign.py gate-report`
- `python experiments\17_maxed_out_campaign.py status`
- Notes: final measured record has `p=0.34814453125` (1426 correct out of 4096), `kappa=0.5272457464634844`, level `5`, and ground truth `2`. After measuring through 3B/problem-91, the locked `K=128, N=8` calibration-selected estimator row has MAE `0.006197719634165873` over 92 rows.

Most recent continuation for 3B/problem-89:
- Initial cache audit found 64 valid samples. The 512, 1024, 3072, and 4096 checkpoints completed cleanly with zero failed batches; the 2048 checkpoint had one recovered internal requeue and zero failed batches. Final cache audit verified `4096/4096` valid samples and zero missing slots for 3B/problem-89.
- `python -u experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 90 --problem-indices 89 --target-samples 4096 --progress-every 1`
- `python experiments\17_maxed_out_campaign.py analyze-heldout --models 3B`
- `python experiments\17_maxed_out_campaign.py gate-report`
- `python experiments\17_maxed_out_campaign.py status`
- Notes: final measured record has `p=0.058349609375` (239 correct out of 4096), `kappa=0.3917899640169534`, level `4`, and ground truth `21`. After measuring through 3B/problem-91, the locked `K=128, N=8` calibration-selected estimator row has MAE `0.006197719634165873` over 92 rows. The full gate report remains `PASS=8`, `WARN=1`, `INFO=1`, `MISSING=4`.

Most recent continuation for 3B/problem-90:
- Initial cache audit found 256 valid samples. Checkpoints to 512 and 1024 completed with zero requeues and zero failed batches; the 2048 checkpoint completed with one recovered requeue and zero failed batches; the 3072 and 4096 checkpoints completed with zero requeues and zero failed batches.
- Final cache audit verified `4096/4096` valid samples and zero missing slots for 3B/problem-90.
- `python -u experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 91 --problem-indices 90 --target-samples 4096 --progress-every 1`
- `python experiments\17_maxed_out_campaign.py analyze-heldout --models 3B`
- `python experiments\17_maxed_out_campaign.py gate-report`
- `python experiments\17_maxed_out_campaign.py status`
- Notes: final measured record is a low-p tail case with `p=0.048828125` (200 correct out of 4096), `kappa=0.30111139630390144`, level `3`, and ground truth `\frac{3}{2}`. After measuring through 3B/problem-91, the locked `K=128, N=8` calibration-selected estimator row has MAE `0.006197719634165873` over 92 rows. The full gate report remains `PASS=8`, `WARN=1`, `INFO=1`, `MISSING=4`.

Most recent continuation for 3B/problem-91:
- Initial cache audit found 64 valid samples. The 512, 1024, and 4096 checkpoints completed with zero requeues and zero failed batches; the 2048 and 3072 checkpoints each had one recovered requeue and zero failed batches.
- Final cache audit verified `4096/4096` valid samples and zero missing slots for 3B/problem-91.
- `python -u experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 92 --problem-indices 91 --target-samples 4096 --progress-every 1`
- `python experiments\17_maxed_out_campaign.py analyze-heldout --models 3B`
- `python experiments\17_maxed_out_campaign.py gate-report`
- `python experiments\17_maxed_out_campaign.py status`
- Notes: final measured record has `p=0.078125` (320 correct out of 4096), `kappa=0.3488868842690678`, level `3`, and ground truth `1`. The locked `K=128, N=8` calibration-selected estimator row now has MAE `0.006197719634165873` over 92 rows. The full gate report remains `PASS=8`, `WARN=1`, `INFO=1`, `MISSING=4`.

Most recent continuation for 3B/problem-92:
- Initial continuation audit found `537/4096` valid raw-cache samples and no measurement file. Low-rate top-offs then completed 1024, 2048, 3072, and 4096 checkpoints with zero failed batches; the 1024 checkpoint had one recovered requeue near the end, and the 3072/4096 checkpoints completed cleanly with `--workers 4`, `--request-delay 3`, `--timeout 300`, and `--max-job-attempts 5`.
- Final cache audit verified `4096/4096` valid samples and zero missing slots for 3B/problem-92.
- `python -u experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 93 --problem-indices 92 --target-samples 4096 --progress-every 1`
- `python experiments\17_maxed_out_campaign.py analyze-heldout --models 3B`
- `python experiments\17_maxed_out_campaign.py gate-report`
- `python experiments\17_maxed_out_campaign.py status`
- Notes: final measured record has `p=0.538330078125` (2205 correct out of 4096), `kappa=0.6020517285003195`, level `4`, and ground truth `\frac{448}{15625}`. The locked `K=128, N=8` calibration-selected estimator row now has MAE `0.006167350732892268` over 93 rows. The full gate report remains `PASS=8`, `WARN=1`, `INFO=1`, `MISSING=4`.

Most recent continuation for 3B/problem-93:
- Initial continuation audit found `78/4096` valid raw-cache samples and no measurement file. Low-rate top-offs then completed 512, 1024, 2048, 3072, and 4096 checkpoints with zero failed batches and zero requeues using `--workers 4`, `--request-delay 3`, `--timeout 300`, and `--max-job-attempts 5`.
- Final cache audit verified `4096/4096` valid samples and zero missing slots for 3B/problem-93.
- `python -u experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 94 --problem-indices 93 --target-samples 4096 --progress-every 1`
- `python experiments\17_maxed_out_campaign.py analyze-heldout --models 3B`
- `python experiments\17_maxed_out_campaign.py gate-report`
- `python experiments\17_maxed_out_campaign.py status`
- Notes: final measured record has `p=0.964111328125` (3949 correct out of 4096), `kappa=0.6372128998472015`, level `4`, and ground truth `33`. The locked `K=128, N=8` calibration-selected estimator row now has MAE `0.006115284391454071` over 94 rows. The full gate report remains `PASS=8`, `WARN=1`, `INFO=1`, `MISSING=4`.
- No campaign process was left running after the most recent checks.

Most recent continuation for 3B/problem-94:
- Initial continuation audit found `64/4096` valid raw-cache samples and no measurement file. Low-rate top-offs completed 512, 1024, 2048, 3072, and 4096 checkpoints using `--workers 4`, `--request-delay 3`, `--timeout 300`, and `--max-job-attempts 5`. The 2048 checkpoint had two recovered requeues and the 4096 checkpoint had one recovered requeue; all checkpoints ended with zero failed batches.
- Final cache audit verified `4096/4096` valid samples and zero missing slots for 3B/problem-94.
- `python -u experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 95 --problem-indices 94 --target-samples 4096 --progress-every 1`
- `python experiments\17_maxed_out_campaign.py analyze-heldout --models 3B`
- `python experiments\17_maxed_out_campaign.py gate-report`
- `python experiments\17_maxed_out_campaign.py status`
- Notes: final measured record is a low-p tail case with `p=0.02685546875` (110 correct out of 4096), `kappa=0.5133763627240797`, level `5`, and ground truth `80`. The locked `K=128, N=8` calibration-selected estimator row now has MAE `0.0060844377524898975` over 95 rows. The full gate report remains `PASS=8`, `WARN=1`, `INFO=1`, `MISSING=4`.
- No campaign process was left running after the most recent checks.

Most recent continuation for 3B/problem-95:
- Initial continuation audit found `64/4096` valid raw-cache samples and no measurement file. Low-rate top-offs completed 512, 1024, 2048, 3072, and 4096 checkpoints. The 1024 checkpoint had one recovered requeue; the first 3072 attempt stalled near `2828/3072` valid samples and was stopped, then a shorter-timeout retry filled the remaining 244 slots with 13 recovered requeues and zero failed batches. The final 4096 checkpoint used `--timeout 120`, completed cleanly, and had zero failed batches.
- Final cache audit verified `4096/4096` valid samples and zero missing slots for 3B/problem-95.
- `python -u experiments\17_maxed_out_campaign.py measure-math --model 3B --n-problems 96 --problem-indices 95 --target-samples 4096 --progress-every 1`
- `python experiments\17_maxed_out_campaign.py analyze-heldout --models 3B`
- `python experiments\17_maxed_out_campaign.py gate-report`
- `python experiments\17_maxed_out_campaign.py status`
- Notes: final measured record has `p=0.932861328125` (3821 correct out of 4096), `kappa=0.6242030881968071`, level `2`, and ground truth `-4`. The locked `K=128, N=8` calibration-selected estimator row now has MAE `0.0060458468069552185` over 96 rows. The full gate report remains `PASS=8`, `WARN=1`, `INFO=1`, `MISSING=4`.
- No campaign process was left running after the most recent checks.

Current cross-benchmark summary:

| Benchmark | Records | Coverage | Exact-law MAE | Nondegenerate | N=48 |
| --- | ---: | ---: | ---: | ---: | ---: |
| GPQA Diamond | 320/320 | 1.0 | 0.0 | 251 | 0.5890 |
| IFEval | 320/320 | 1.0 | 0.0 | 164 | 0.7962 |
| LiveBench selected | 320/320 | 1.0 | 0.0 | 141 | 0.2546 |
| LiveCodeBench | 320/320 | 1.0 | 0.0 | 262 | 0.5246 |

Other completed artifacts:
- Core Du-aligned bundle validates the exact finite empirical selector law across 23 models with overall MAE `0.000622`.
- Maxed-out held-out MATH collection has 119 real records measured at 4096 samples: 3B/problem-0 through 3B/problem-118. The newest 4096-depth measured record, 3B/problem-118, is a level-5 hard-side record with `p=0.0732421875` (300 correct out of 4096), `kappa=0.5267746750965929`, and ground truth `81`. The locked-estimator gates for `K=128/256/512, N=8` pass over the completed 4096-depth records with locked `K=128, N=8` MAE `0.005639141465438472`, but this is not enough for the maximum claim because coverage is only 119/11,500 manifest target records.
- The next raw-cache frontier beyond measured records is 3B/problem-119 at `3152/4096` valid raw-cache samples after a stopped partial top-off run. Do not count it until its valid cache count reaches 4096 and `measure-math` has written the 4096-depth measurement file.
- Expanded pilot sample-complexity artifact covers 5 models, 100 problems, 256 samples/problem, and K values through 192.
- Nested-calibration proxy over the 256-sample expanded pilot pool is written under `results/maxed_out/expanded_pilot_proxy/`. Its `K=128, N=8` aggregate MAE is `0.04014`, which fails the ideal `0.03` gate, so it should be treated as diagnostic evidence that the full 4096-sample split is needed.
- Live LLM judge artifact covers 135/135 model/problem pairs and 6480/6480 judgments on a 45-problem manifest.
- Adaptive allocation artifact compares uniform, p-based, AUC/kappa, moment-law, and oracle policies.
- Current full gate report summary is `PASS=8`, `WARN=1`, `INFO=1`, `MISSING=4`; claim summary is `PASS=8`, `MISSING=4`; diagnostic summary is `WARN=1`, `INFO=1`. The diagnostic warning is the expanded-pilot proxy at `K=128, N=8`. Claim-blocking missing gates are maxed held-out coverage, maxed live-judge coverage, six-family cross-benchmark presence, and manifest-scale cross-benchmark coverage.
- Cross-benchmark summaries now record requested task limit, requested tasks, requested samples/task, observed task/model counts, and sample-depth fields. The current four completed families are pilot-scale only: 20 tasks, 48 samples/task, and 16 live models. The maxed-out cross-benchmark gate requires six families at 500 requested tasks, 128 samples/task, 16 live models, and 100% measured coverage.

## 5. Known Bugs Or Open Questions
- The actual inference-value theorem manuscript is not written in this repo.
- UNKNOWN whether the final manuscript should live in this repo, in `C:\Users\wangz\MIRROR\paper`, or in a new paper directory.
- Full maxed-out collection has started but is far from complete. The generated full manifest targets 47,104,000 held-out MATH responses, 8,832,000 verifier judgments across all target models, 6,144,000 live-runnable verifier judgments, and 6,144,000 cross-benchmark responses through the now-supported six-family runner.
- `experiments/16_cross_benchmark.py` now supports MATH500 and MBPP-backed `humaneval_mbpp`, but those two new benchmark families have not been collected/measured/analyzed yet. Current evidence still has four completed cross-benchmark families, all at pilot scale, until the six-family manifest-scale command queue is run.
- Recent 3B/problem-57 through 3B/problem-118 collection attempts showed that high-concurrency runs can stall or leave a small failed tail. Low-rate single-sample chunks plus targeted top-offs worked through 4096 valid samples for problems 57-95; problem-96 eventually completed cleanly at 4096 after a resumed 16-worker single-sample top-off from 3270/4096 valid samples; problems 97-99 completed cleanly through all checkpoints with the 16-worker single-sample profile; problem-100 needed one 3072 top-off after an early process exit, then completed 4096 cleanly; problem-101 completed cleanly through all checkpoints with zero failed batches but slow 16-worker throughput; problem-102 completed cleanly through all checkpoints with zero failed batches and many recovered requeues; problem-103 completed cleanly with zero failed batches but slow 2048/3072 throughput; problem-104 completed from 64/4096 to 4096/4096 in one 16-worker single-sample run with four recovered requeues, one retired auth key, and zero failed batches; problem-105 completed from 64/4096 to 4096/4096 in one 16-worker batch-8 run with three recovered requeues, one retired auth key, and zero failed batches; problem-106 completed from 64/4096 to 4096/4096 in one 16-worker batch-8 run with four recovered requeues, one retired auth key, and zero failed batches; problem-107 completed from 64/4096 to 4096/4096 in one 16-worker batch-8 run with 20 recovered requeues, one retired auth key, and zero failed batches; problem-108 needed an eight-sample top-off after a provider `DEGRADED` tail, then audited cleanly at 4096/4096; problem-109 completed from 64/4096 to 4096/4096 in one 16-worker batch-8 run with five recovered requeues, one retired auth key, and zero failed batches; problem-110 completed from 256/4096 to 4096/4096 in one 16-worker batch-8 run with 10 recovered requeues, one retired auth key, and zero failed batches; problem-111 completed from 64/4096 to 4096/4096 in one 16-worker batch-8 run with three recovered requeues, one retired auth key, and zero failed batches; problem-112 completed from 256/4096 to 4096/4096 in one 16-worker batch-8 run with three recovered requeues, one retired auth key, and zero failed batches; problem-113 completed from 80/4096 to 4096/4096 in one 16-worker batch-8 run with 31 recovered requeues, one retired auth key, and zero failed batches; problem-114 completed from 64/4096 to 4096/4096 in one 16-worker batch-8 run with four recovered requeues, one retired auth key, and zero failed batches; problem-115 completed from 64/4096 to 4096/4096 in one 16-worker batch-8 run with 11 recovered requeues, one retired auth key, and zero failed batches; problem-116 completed from 64/4096 to 4096/4096 in one 16-worker batch-8 run with 30 recovered requeues, one retired auth key, and zero failed batches; problem-117 completed from 80/4096 to 4096/4096 in one 16-worker batch-8 run with 23 recovered requeues, one retired auth key, and zero failed batches; problem-118 completed from 64/4096 to 4096/4096 in one 16-worker batch-8 run with three recovered requeues, one retired auth key, and zero failed batches; problem-119 was stopped at 3152/4096 valid raw-cache samples after 352 completed batches, one recovered requeue, one retired auth key, and zero failed batches. Continue using cache audits as source of truth and avoid counting partial records as manifest-complete.
- Optional: add a second external verifier if broader judge-generalization evidence is needed.
- Optional: add hidden/private LiveCodeBench scoring if a stronger code appendix is needed.
- Do not claim all 23 models for the cross-benchmark live panel; claim the 16 stable high-volume live NIM endpoints only.
- Do not claim global optimal allocation, broad judge superiority, or tiny-pilot sufficiency.

## 6. Next Recommended Steps
1. Start writing from `docs/MANUSCRIPT_DRAFT.md` and use `docs/PAPER_WRITING_PLAN.md` as the claim-boundary checklist.
2. Review `results/maxed_out/manifest.json` and `results/maxed_out/command_queue.jsonl` only if resuming the maxed-out campaign.
3. Continue the full 4096-sample held-out forecasting campaign only if more data is desired. The next direct increment is 3B/problem-119 from `3152/4096` valid raw-cache samples to 4096, then rerun `measure-math --target-samples 4096`, `analyze-heldout`, `gate-report`, and `status`.
4. If choosing to publish before manifest completion, scope the current evidence package explicitly: 119 full 4096-depth 3B held-out records, four completed pilot-scale cross-benchmark families, and the current gate report (`PASS=8`, `WARN=1`, `INFO=1`, `MISSING=4`). Do not present this as full maxed-out manifest completion.
5. Run the six-family cross-benchmark command queue at manifest scale: MATH500, GPQA Diamond, IFEval, LiveBench selected, LiveCodeBench, and MBPP-backed `humaneval_mbpp`, with 500 requested tasks and 128 samples/task.
6. Use `python experiments\17_maxed_out_campaign.py gate-report` as the paper-claim gate before upgrading any wording to maximum-strength claims.
7. Add second/third independent live judge providers before claiming broad multi-judge superiority.
8. After the maxed-out campaign or if choosing to publish now, decide the manuscript location and write the paper from `results/du_aligned/`, `results/benchmarks/`, and `results/maxed_out/`.
9. Before `/clear`, update this file with current status, changed files, checks run, known unknowns, and next steps.
