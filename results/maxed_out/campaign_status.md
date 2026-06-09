# Maxed-Out Campaign Status

- Manifest: `C:\Users\wangz\Downloads\inference-value-theorem-github\results\maxed_out\manifest.json`
- Smoke mode: `False`
- Created UTC: `2026-05-29T12:07:04+00:00`
- Repo commit: `b1fc0594692d13304c9619bfa21c7697f18fed46`
- Target models: 23
- Live collection models: 16
- Inactive target models: ['GLM5', 'KimiK2', 'MiniMax', 'MistralLarge675B', 'MistralNemotron', 'Mixtral8x22B', 'Ultra253B']

## Planned Scale

- Held-out MATH responses: 47,104,000
- Held-out split: {'pilot': 1024, 'calibration': 1024, 'evaluation': 2048}
- Real-verifier judgments, all target models: 8,832,000
- Real-verifier judgments, live-runnable panel: 6,144,000
- Cross-benchmark responses, desired families: 6,144,000
- Cross-benchmark responses, existing runner support: 6,144,000

## Current Maxed-Out Artifacts

- Command queue: `C:\Users\wangz\Downloads\inference-value-theorem-github\results\maxed_out\command_queue.jsonl`
- Held-out detail table exists: `True`
- Held-out summary table exists: `True`

| Model | Live endpoint | Any measurement records | Records >= 1024 | Records >= 2048 | Max samples/record | Records at manifest n | Last measured |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 3B | True | 119 | 119 | 119 | 4096 | 119 | 2026-06-05T12:28:19+00:00 |
| 70B | True | 3 | 0 | 0 | 8 | 0 | 2026-05-29T11:52:17+00:00 |
| 8B | True | 0 | 0 | 0 | 0 | 0 | N/A |
| Dracarys70B | True | 0 | 0 | 0 | 0 | 0 | N/A |
| GLM5 | False | 0 | 0 | 0 | 0 | 0 | N/A |
| Gemma3n2B | True | 0 | 0 | 0 | 0 | 0 | N/A |
| Gemma3n4B | True | 0 | 0 | 0 | 0 | 0 | N/A |
| KimiK2 | False | 0 | 0 | 0 | 0 | 0 | N/A |
| Maverick17B | True | 0 | 0 | 0 | 0 | 0 | N/A |
| MiniMax | False | 0 | 0 | 0 | 0 | 0 | N/A |
| Ministral14B | True | 0 | 0 | 0 | 0 | 0 | N/A |
| MistralLarge675B | False | 0 | 0 | 0 | 0 | 0 | N/A |
| MistralNemotron | False | 0 | 0 | 0 | 0 | 0 | N/A |
| MistralSmall119B | True | 0 | 0 | 0 | 0 | 0 | N/A |
| Mixtral8x22B | False | 0 | 0 | 0 | 0 | 0 | N/A |
| Mixtral8x7B | True | 0 | 0 | 0 | 0 | 0 | N/A |
| Qwen122B | True | 0 | 0 | 0 | 0 | 0 | N/A |
| Qwen397B | True | 0 | 0 | 0 | 0 | 0 | N/A |
| QwenNext80B | True | 0 | 0 | 0 | 0 | 0 | N/A |
| Stockmark100B | True | 0 | 0 | 0 | 0 | 0 | N/A |
| Super120B | True | 0 | 0 | 0 | 0 | 0 | N/A |
| Super49B | True | 0 | 0 | 0 | 0 | 0 | N/A |
| Ultra253B | False | 0 | 0 | 0 | 0 | 0 | N/A |

## Existing Closure Evidence

- Du-aligned bundle: present
- Existing expanded pilot max samples/problem: 256
- Existing expanded pilot models: ['3B', '70B', 'Super49B', 'Super120B', 'MistralSmall119B']
- Existing live judge status: completed (135 pairs)
- Existing cross benchmarks: ['gpqa_diamond', 'ifeval', 'livebench_selected', 'livecodebench']

## Next Commands

Continue with the next real 4096-sample held-out slice:

```bash
python experiments/17_maxed_out_campaign.py collect-math --model 3B --n-problems 120 --problem-indices 119 --target-samples 4096 --batch-size 8 --workers 16 --request-delay 2 --progress-every 32
python experiments/17_maxed_out_campaign.py measure-math --model 3B --n-problems 120 --problem-indices 119 --target-samples 4096
python experiments/17_maxed_out_campaign.py analyze-heldout --models 3B
```
