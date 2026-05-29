# Agent Instructions

## Goal
Implement changes conservatively, verify claims from repo files or command output, and keep reusable context in files instead of long chat history.

## Repo Map
- `src/`: shared implementation code.
- `experiments/`: experiment runners and analyses.
- `results/`: generated experiment artifacts, tables, figures, and status docs.
- `docs/CODEX_HANDOFF.md`: current working memory for fresh Codex sessions.

## Commands
- Compile key scripts: `python -m py_compile config.py experiments\15_paper_claims_status.py experiments\16_cross_benchmark.py`
- Check whitespace before commits: `git diff --check`
- Check sync after pushes: `git status --short --branch` and `git rev-list --left-right --count origin/main...main`

## Rules
- Search and inspect the repo before claiming a function, file, result, or manuscript exists.
- Trust code, committed artifacts, and command output over previous chat context or handoff notes.
- When unsure, write `UNKNOWN` instead of guessing.
- Never print, commit, or summarize API keys. This repo may read NIM keys from `NIM_API_KEYS` or `C:\Users\wangz\MIRROR\.env`.
- Scope paper claims to the evidence in `results/`; do not claim unavailable, EOL, degraded, or rate-limited endpoints were benchmarked live.
- After changing experiment logic, run the smallest relevant check first, then broader checks when needed.

## Context Files
- `AGENTS.md` is repo law: stable rules, commands, conventions, and done criteria.
- `docs/CODEX_HANDOFF.md` is current working memory and should be updated before `/clear`.
- `docs/EXPERIMENT_LOG.md` is for empirical results, if created.
- `docs/ARCHITECTURE.md` is for stable design facts, if created.

## AGENTS.md Update Policy
- Change `AGENTS.md` rarely and deliberately, not every session.
- Add a rule only after repeated friction or after the same type of mistake happens 2-3 times.
- Keep temporary bugs, session notes, current next steps, and experiment logs out of `AGENTS.md`.
- At the end of a session, ask: "Did I make any repeated mistake that should become a permanent repo rule?" If yes, propose the smallest possible `AGENTS.md` edit. If no, update only `docs/CODEX_HANDOFF.md`.

## Definition Of Done
- Relevant checks have run or any skipped checks are named with the reason.
- Claims in final responses cite repo artifacts, command output, or explicitly marked unknowns.
- When asked to commit or push, finish with a clean worktree and verify local `main`, `origin/main`, and the GitHub remote head match.
