# Val samples: base vs fine-tuned vs FP8 (same 900 samples, Step 6 scorer)

| skill | subset | n | base BF16 | fine-tuned BF16 | fine-tuned FP8 |
|---|---|---|---|---|---|
| first glance | positives | 240 | 42.1 | 52.1 | 50.8 |
| first glance | negatives (absent) | 60 | 45.0 | 90.0 | 90.0 |
| look again | positives | 255 | 0.0 | 50.2 | 49.8 |
| look again | negatives (absent) | 45 | 100.0 | 97.8 | 91.1 |
| confirm | yes crops | 147 | 84.4 | 90.5 | 91.8 |
| confirm | no crops (all) | 153 | 95.4 | 96.1 | 96.1 |
| confirm | no: near miss | 27 | 85.2 | 85.2 | 85.2 |
| confirm | no: distractor | 78 | 97.4 | 98.7 | 98.7 |
| confirm | no: background | 48 | 97.9 | 97.9 | 97.9 |

Unparsed answers: base BF16: 0, fine-tuned BF16: 0, fine-tuned FP8: 0
