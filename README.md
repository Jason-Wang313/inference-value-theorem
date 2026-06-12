# Score-Rank Evaluation for Multi-Sample Language Models

This repository contains the empirical artifacts, experiment runners, and
ICLR-targeted manuscript package for the LLM response-pool score-rank
evaluation paper. The paper-facing draft is in `paper/iclr2027/` and is
framed as language-model selection evaluation, not as an architecture paper.

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

Additional best-paper closure experiments:

```bash
python experiments/13_adaptive_allocation.py --pilot-k 16 --seeds 20
python experiments/12_expanded_pilot.py --collect --measure --target-samples 256 --workers 4
python experiments/11_model_judge.py --live --models 3B 8B 70B --n-problems 50 --max-samples 48 --workers 12 --request-delay 3 --rate-limit-cooldown 90 --max-task-attempts 16 --live-batch-size 8
python experiments/14_live_judge_analysis.py
python experiments/15_paper_claims_status.py
```

Cross-benchmark expansion:

```bash
python experiments/16_cross_benchmark.py --benchmarks gpqa_diamond ifeval livebench_selected livecodebench --model-set all --n-tasks 20 --n-samples 48 --batch-size 1 --workers 192 --collect --allow-unsafe-code-exec --code-test-scope public
```

```bash
python experiments/16_cross_benchmark.py --benchmarks gpqa_diamond ifeval livebench_selected livecodebench --model-set all --n-tasks 20 --n-samples 48 --measure --analyze --allow-unsafe-code-exec --code-test-scope public --measure-workers 12
```

The current cross-benchmark expansion covers GPQA Diamond, IFEval, LiveBench
selected subsets, and LiveCodeBench on the 16-model stable live NIM panel with
20 tasks/benchmark and 48 samples/model/task. All four benchmarks have
320/320 measured model/task records, 100% grading coverage, exact-law MAE
0.0, and non-degenerate counts of 251, 164, 141, and 262 respectively.
Unavailable/deprecated/rate-limited provider endpoints are excluded from this
live panel and are commented in `config.py`.

Current closure artifacts include a 5-model, 100-problem expanded pilot with
256 samples/problem for all five models, enabling five-model K=128/192 pilot
curves; adaptive allocation on the same 5-model set; and a completed live LLM
judge subset with 135/135 model/problem pairs and 6480/6480 judgments.

Maxed-out campaign orchestration:

```bash
python experiments/17_maxed_out_campaign.py prepare --force
python experiments/17_maxed_out_campaign.py status
python experiments/17_maxed_out_campaign.py analyze-expanded-proxy --seeds 5
python experiments/17_maxed_out_campaign.py gate-report
```

This writes a locked manifest and resumable command queue to
`results/maxed_out/` for the 4096-sample held-out forecasting campaign,
large real-verifier scaling, and cross-benchmark expansion. The gate report
marks which ideal paper-claim metrics are passed, missing, or failed, and
requires any estimator/policy adjustment to use pilot or calibration data only.
The current nested-calibration proxy over the existing 256-sample expanded
pilot pool is diagnostic only: `K=128, N=8` has MAE `0.04014` against the
`0.03` ideal target, so the maximum-strength held-out forecasting claim still
requires the full 4096-sample locked split. 119 true maxed-out
held-out records now exist: `3B` problems `0` through `118` each have 4096
measured samples. The measured 4096-depth set includes the all-incorrect stress
records `problem_11`, `problem_60`, `problem_71`, `problem_96`, and
`problem_98` with `p=0.0`,
the hard stress records `problem_21` with `p=0.000244140625`,
`problem_106` with `p=0.0`, `problem_109` with `p=0.0`,
`problem_103` with `p=0.000244140625`, `problem_23`
with `p=0.001708984375`, `problem_43` with `p=0.001708984375`,
`problem_31` with `p=0.0029296875`,
`problem_25` with `p=0.00048828125`, `problem_7`
with `p=0.001953125`, `problem_64` with `p=0.002197265625`,
`problem_100` with `p=0.007080078125`,
`problem_101` with `p=0.01171875`,
`problem_15` with `p=0.01318359375`,
`problem_62` with `p=0.015869140625`, `problem_82` with
`p=0.024169921875`, `problem_94` with `p=0.02685546875`,
`problem_110` with `p=0.02734375`, and `problem_9` with
`p=0.0302734375`, `problem_26` with `p=0.032470703125`,
`problem_48` with `p=0.039306640625`,
`problem_78` with `p=0.048583984375`,
`problem_90` with `p=0.048828125`,
`problem_104` with `p=0.050537109375`,
`problem_46` with `p=0.056396484375`,
`problem_89` with `p=0.058349609375`,
`problem_99` with `p=0.064453125`,
`problem_63` with `p=0.068603515625`,
`problem_118` with `p=0.0732421875`,
`problem_91` with `p=0.078125`,
`problem_68` with `p=0.0810546875`,
`problem_36` with `p=0.044921875`, `problem_33` with `p=0.078369140625`,
`problem_50` with `p=0.0830078125`,
`problem_108` with `p=0.106201171875`,
`problem_105` with `p=0.1064453125`,
the medium
records `problem_22` with
`p=0.23193359375`, `problem_102` with `p=0.235107421875`,
`problem_37` with `p=0.2470703125`, `problem_29` with
`p=0.287353515625`, `problem_75` with `p=0.273193359375`,
`problem_115` with `p=0.281982421875`,
`problem_97` with `p=0.29345703125`,
`problem_10` with `p=0.33984375`, `problem_34` with
`p=0.346923828125`, `problem_88` with `p=0.34814453125`, `problem_32` with
`p=0.352294921875`, `problem_12` with `p=0.085693359375`, `problem_14` with
`p=0.13232421875`, `problem_24` with `p=0.134765625`, `problem_41` with
`p=0.14306640625`, `problem_83` with `p=0.146728515625`,
the nondegenerate `problem_17` with
`p=0.151611328125`, `problem_66` with `p=0.16650390625`,
`problem_74` with `p=0.160888671875`,
the mid-range `problem_18` with `p=0.354736328125`,
`problem_69` with `p=0.392822265625`, and
`problem_84` with `p=0.436767578125`,
`problem_20` with `p=0.485107421875`, `problem_55` with
`p=0.533935546875`, `problem_92` with `p=0.538330078125`,
`problem_19` with `p=0.5986328125`,
`problem_28` with `p=0.6005859375`, `problem_35` with `p=0.604248046875`,
`problem_44` with `p=0.606689453125`, `problem_51` with
`p=0.575439453125`, `problem_80` with `p=0.644775390625`,
`problem_81` with `p=0.525146484375`,
`problem_114` with `p=0.6591796875`,
`problem_58` with `p=0.68212890625`,
`problem_53` with `p=0.719970703125`, the easy-side record `problem_27`
with `p=0.729736328125`, `problem_61` with `p=0.775146484375`,
`problem_59` with `p=0.783203125`, and the easy-side
records `problem_85` with `p=0.82861328125`, `problem_16` with
`p=0.82958984375`, `problem_47` with
`p=0.8359375`, `problem_57` with `p=0.836669921875`, `problem_42` with
`p=0.849609375`, `problem_45` with `p=0.855224609375`,
`problem_76` with `p=0.862548828125`, `problem_39` with
`p=0.88427734375`, `problem_111` with `p=0.89990234375`,
`problem_73` with `p=0.903564453125`,
`problem_70` with `p=0.908935546875`, `problem_72` with
`p=0.918212890625`, `problem_112` with `p=0.919189453125`,
`problem_107` with `p=0.9140625`,
`problem_8` with `p=0.92236328125`,
`problem_52` with `p=0.91015625`,
`problem_56` with `p=0.92138671875`,
`problem_95` with `p=0.932861328125`,
`problem_113` with `p=0.936767578125`,
`problem_30` with `p=0.94775390625`, `problem_86` with
`p=0.9541015625`, `problem_54` with
`p=0.954345703125`, `problem_65` with `p=0.955078125`,
`problem_116` with `p=0.955078125`,
`problem_38` with `p=0.958740234375`, `problem_67` with
`p=0.970947265625`, `problem_93` with `p=0.964111328125`,
`problem_117` with `p=0.965576171875`,
`problem_87` with `p=0.96923828125`,
`problem_77` with `p=0.975830078125`, `problem_40` with
`p=0.97998046875`, `problem_79` with `p=0.98095703125`,
`problem_49` with `p=0.989501953125`, and the near-solved `problem_13` with
`p=0.9951171875`.
The locked estimator gates for `K=128/256/512, N=8` pass over the completed
4096-depth records with MAE `0.00563914` and `num_rows=119`. The coverage gate
remains missing at 119/11,500 manifest target records until all model/problem
records are measured at 4096 samples. The next raw-cache frontier beyond the
measured set is `problem_119` at `3152/4096` valid samples after a stopped
partial top-off run; it is not counted in held-out metrics until it reaches the
4096-sample manifest depth and is measured.
The gate report also distinguishes family presence from maxed-out scale:
the current four completed cross-benchmark families are complete at the pilot
scale of 20 tasks, 48 samples/task, and 16 live models, but the maximum claim
requires the manifest-scale run with six families, 500 requested tasks, and
128 samples/task.
The cross-benchmark runner now supports the six-family command queue:
MATH500, GPQA Diamond, IFEval, LiveBench selected, LiveCodeBench, and an
MBPP-backed `humaneval_mbpp` executable-code family. For a local smoke check
without API spend:

```bash
python experiments/17_maxed_out_campaign.py prepare --smoke --force
python experiments/17_maxed_out_campaign.py collect-math --model 3B --problem-indices 0-2 --target-samples 8 --dry-run
python experiments/17_maxed_out_campaign.py measure-math --model 3B --problem-indices 0-2 --target-samples 8
python experiments/17_maxed_out_campaign.py analyze-heldout --models 3B --smoke
python experiments/17_maxed_out_campaign.py status --smoke
python experiments/17_maxed_out_campaign.py gate-report --smoke
```

The project reads NIM keys from `NIM_API_KEYS` when set. If that variable is
absent on this machine, `config.py` falls back to `C:\Users\wangz\MIRROR\.env`
and composes the `NVIDIA_NIM_API_KEY*` entries without printing key material.
When this checkout's `data/raw` directory is empty, `src/nim_client.py` falls
back to the sibling `C:\Users\wangz\Downloads\inference-value-theorem\data\raw`
cache; set `INFERENCE_VALUE_RAW_DIR` to override that behavior.

API keys are intentionally not committed. Set `NIM_API_KEYS` in the
environment before running any collection scripts.
