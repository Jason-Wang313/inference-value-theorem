# Current Quick-Wrap Evidence Snapshot

This file records the scoped quick-wrap package after completing the
4096-sample `3B/problem_118` measurement.

## Scope

- This is not the full maxed-out campaign.
- The full manifest still targets 11,500 held-out MATH records at 4096 samples each, plus large verifier and cross-benchmark runs.
- The current held-out evidence has moved beyond this quick-wrap snapshot.

## Held-Out Forecasting

- `3B` now has 119 manifest-depth held-out records: `problem_0` through `problem_118`, each measured at 4096 samples.
- `3B/problem_106` is the hardest completed tail record so far: `p=0.0`, 0 correct out of 4096, `kappa=null`.
- `3B/problem_103` remains the one-correct extreme-tail record: `p=0.000244140625`, 1 correct out of 4096, `kappa=0.3641025641025641`.
- `3B/problem_104` is now measured at full depth: `p=0.050537109375`, 207 correct out of 4096, `kappa=0.41121558017597015`.
- `3B/problem_105` is now measured at full depth: `p=0.1064453125`, 436 correct out of 4096, `kappa=0.4066708026269614`.
- `3B/problem_107` is now measured at full depth: `p=0.9140625`, 3744 correct out of 4096, `kappa=0.6872291120337995`.
- `3B/problem_108` is now measured at full depth: `p=0.106201171875`, 435 correct out of 4096, `kappa=0.5235194202953153`.
- `3B/problem_109` is now measured at full depth: `p=0.0`, 0 correct out of 4096, `kappa=null`.
- `3B/problem_110` is now measured at full depth: `p=0.02734375`, 112 correct out of 4096, `kappa=0.4159271012621916`.
- `3B/problem_111` is now measured at full depth: `p=0.89990234375`, 3686 correct out of 4096, `kappa=0.4981015841086246`.
- `3B/problem_112` is now measured at full depth: `p=0.919189453125`, 3765 correct out of 4096, `kappa=0.7292032273724839`.
- `3B/problem_113` is now measured at full depth: `p=0.936767578125`, 3837 correct out of 4096, `kappa=0.6478617565404118`.
- `3B/problem_114` is now measured at full depth: `p=0.6591796875`, 2700 correct out of 4096, `kappa=0.6245211185397432`.
- `3B/problem_115` is now measured at full depth: `p=0.281982421875`, 1155 correct out of 4096, `kappa=0.4952095394121916`.
- `3B/problem_116` is now measured at full depth: `p=0.955078125`, 3912 correct out of 4096, `kappa=0.6493273206188317`.
- `3B/problem_117` is now measured at full depth: `p=0.965576171875`, 3955 correct out of 4096, `kappa=0.7107154064789162`.
- `3B/problem_118` is now measured at full depth: `p=0.0732421875`, 300 correct out of 4096, `kappa=0.5267746750965929`.
- The next raw-cache frontier beyond measured records is `3B/problem_119` at `3152/4096` valid raw-cache samples after a stopped partial top-off run.
- The locked held-out `K=128, N=8` MAE is `0.005639141465438472` over `num_rows=119`.

## Gates

- Full gate summary: `PASS=8`, `WARN=1`, `INFO=1`, `MISSING=4`.
- Claim-blocking missing gates remain: maxed held-out coverage, maxed live-judge coverage, six-family cross-benchmark presence, and manifest-scale cross-benchmark coverage.

## Claim Wording

Use the current package for scoped claims about exact-law validation, pilot-scale cross-benchmark behavior, live-judge subset improvement, adaptive allocation, and 119 full-depth 3B held-out records. Do not claim full maxed-out coverage or manifest-scale generality.
