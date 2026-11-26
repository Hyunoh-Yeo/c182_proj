# V2 learning-rate check

Short runs of `train/train_delims_v2.py` (all four template tokens, `--init template`) on Llama-3.1-8B-Instruct in 4-bit, before the full V2 run. Each folder holds the run's `config.json` and `log.jsonl` (validation loss, share of pairs beating the reference margin, norms, and distance moved from the original embeddings). The cosine schedule decays to 0 within each short run.

| Run | Val. loss, step 0 | Last val. step | Val. loss | beats_ref, step 0 → last | Distance moved (header, input, end_header, eot) |
|---|---|---|---|---|---|
| `lr0.0001_ga16_30steps` | 6.56 | 29 | 6.34 | 0.06 → 0.07 | 0.037, 0.047, 0.029, 0.032 |
| `lr0.001_ga16_30steps` | 6.56 | 29 | 3.09 | 0.06 → 0.31 | 0.321, 0.336, 0.293, 0.286 |
| `lr0.1_ga4_3steps` | 5.68 | 3 | 156.66 | 0.00 → 0.50 | 8.829, 8.240, 8.197, 8.617 |

`lr0.1_ga4_3steps` is the 3-step smoke test (32 training rows, 4 validation pairs, so its step-0 loss differs from the other two, which use 480 training rows and 100 validation pairs): lr 0.1 (V1's value) moves every vector by about 6.4 in the first step, 100 to 200 times the norm of the special tokens, and the validation loss jumps to 156.7. `lr0.001` learns steadily; `lr0.0001` barely moves in 30 steps. We use lr 1e-3 for the full V2 run.
