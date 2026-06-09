# Maxed-Out Campaign Status

- Manifest: `C:\Users\wangz\Downloads\inference-value-theorem-github\results\maxed_out\manifest.smoke.json`
- Smoke mode: `True`
- Created UTC: `2026-05-29T11:57:05+00:00`
- Repo commit: `b1fc0594692d13304c9619bfa21c7697f18fed46`
- Target models: 2
- Live collection models: 2
- Inactive target models: []

## Planned Scale

- Held-out MATH responses: 48
- Held-out split: {'pilot': 4, 'calibration': 2, 'evaluation': 2}
- Real-verifier judgments, all target models: 72
- Real-verifier judgments, live-runnable panel: 72
- Cross-benchmark responses, desired families: 48
- Cross-benchmark responses, existing runner support: 48

## Current Maxed-Out Artifacts

- Command queue: `C:\Users\wangz\Downloads\inference-value-theorem-github\results\maxed_out\command_queue.smoke.jsonl`
- Held-out detail table exists: `True`
- Held-out summary table exists: `True`

| Model | Live endpoint | Any measurement records | Records at manifest n | Last measured |
| --- | ---: | ---: | ---: | --- |
| 3B | True | 4 | 3 | 2026-05-29T11:52:18+00:00 |
| 70B | True | 4 | 3 | 2026-05-29T11:52:17+00:00 |

## Existing Closure Evidence

- Du-aligned bundle: present
- Existing expanded pilot max samples/problem: 256
- Existing expanded pilot models: ['3B', '70B', 'Super49B', 'Super120B', 'MistralSmall119B']
- Existing live judge status: completed (135 pairs)
- Existing cross benchmarks: ['gpqa_diamond', 'ifeval', 'livebench_selected', 'livecodebench']

## Next Commands

Run one model/problem slice first:

```bash
python experiments/17_maxed_out_campaign.py collect-math --model 3B --problem-indices 0-2 --target-samples 8 --dry-run
python experiments/17_maxed_out_campaign.py measure-math --model 3B --problem-indices 0-2 --target-samples 8
python experiments/17_maxed_out_campaign.py analyze-heldout --models 3B --smoke
```
