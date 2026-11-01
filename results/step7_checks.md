# Step 7: base vs fine-tuned checks

Base: Qwen/Qwen3-VL-2B-Instruct (bfloat16). Fine-tuned: /content/fvs_local/models/qwen3-vl-2b-foveated-search (bfloat16). Same val samples, prompts, schemas and parsers (Step 6 scorer).

## Val samples (300 per type; success %)

| skill | subset | n | base | fine-tuned | change |
|---|---|---|---|---|---|
| where (first glance) | all | 300 | 42.7 | 59.7 | +17.0 pp |
| where (first glance) | positives | 240 | 42.1 | 52.1 | +10.0 pp |
| where (first glance) | negatives (absent) | 60 | 45.0 | 90.0 | +45.0 pp |
| look again | all | 300 | 15.0 | 57.3 | +42.3 pp |
| look again | positives | 255 | 0.0 | 50.2 | +50.2 pp |
| look again | negatives (absent) | 45 | 100.0 | 97.8 | -2.2 pp |
| confirm | all | 300 | 90.0 | 93.3 | +3.3 pp |
| confirm | yes crops | 147 | 84.4 | 90.5 | +6.1 pp |
| confirm | no crops | 153 | 95.4 | 96.1 | +0.7 pp |
| confirm | no: near_miss | 27 | 85.2 | 85.2 | +0.0 pp |
| confirm | no: distractor | 78 | 97.4 | 98.7 | +1.3 pp |
| confirm | no: background | 48 | 97.9 | 97.9 | +0.0 pp |

**Negatives check:** no negative subset dropped by more than 5 pp.

## Dev split sanity runs (accuracy runs, concurrency 8; R5 on test is Step 8)

| strategy | model | found % | FP % (neg) | accuracy % | wrong loc. % | calls / query | fixations / query | searches with a rejected crop | took a 2nd fixation after it |
|---|---|---|---|---|---|---|---|---|---|
| eye search (dev, 50) | base | 72.5 | 0.0 | 78.0 | 7.5 | 2.08 | 0.86 | 11 | 0 / 11 |
| eye search (dev, 50) | fine-tuned | 85.0 | 0.0 | 88.0 | 5.0 | 1.82 | 0.76 | 2 | 1 / 2 |
| full frame (dev, 50) | base | 92.5 | 30.0 | 88.0 | 7.5 | 1.00 | 0.00 | - | - |
| full frame (dev, 50) | fine-tuned | 97.5 | 0.0 | 98.0 | 2.5 | 1.00 | 0.00 | - | - |
