# Inference Value Theorem Experiments

This repository contains the empirical artifacts and experiment runners for
the corrected Theorem 1 best-of-N selector audit.

## Du-aligned experiments

The Du-aligned extension is implemented in:

```bash
python experiments/09_du_aligned.py --pilot-seeds 20
```

It writes the complete bundle to `results/du_aligned/`, including the summary
report, JSON results, CSV tables, and PDF figures for:

- moment hierarchy vs AUC,
- pilot sample complexity,
- score-function comparison,
- verifier-style reranking diagnostics,
- posterior monotonicity.

API keys are intentionally not committed. Set `NIM_API_KEYS` in the
environment before running any collection scripts.
