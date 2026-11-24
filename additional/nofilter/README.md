# AlpacaFarm without the data filter

The same runs as `results/attacks/`, but with `--no-filter`: the Llama-3 control tokens and our delimiter
tokens in the untrusted data reach ours and undefended unchanged. SecPO and DefensiveToken are not here
because the flag does not change them (they always use their released filters; see `results/attacks/`).

This isolates one change, the filter, and shows that the defense comes from the delimiters rather than the
filter. Any-attack ASR over the 10 shared attacks (208 samples, Wilson CI):

| Defense | ASR, filter off | 95% CI | ASR, filter on (`results/`) |
|---|---|---|---|
| Undefended | 96.6% (201/208) | [93.2, 98.4] | 95.7% (199/208) |
| Undefended, input-role template (`--no-delims`) | 98.1% (204/208) | [95.2, 99.2] | 94.2% (196/208) |
| Ours, step 300 | 13.5% (28/208) | [9.5, 18.8] | 0.5% (1/208) |
| Ours, step 600 | 1.9% (4/208) | [0.8, 4.8] | 0.0% (0/208) |
| Ours, step 1163 | 1.9% (4/208) | [0.8, 4.8] | 0.0% (0/208) |

All 4 failures at step 1163 are `completion_chat_ignore`, which writes real chat-template tokens into the
data. `delim_escape` (closing our delimiters from inside the data) is 0.0% at steps 600 and 1163 and 3.4% at
step 300. The `delim_escape` row of `no-delims` is not meaningful: the trained vectors are not loaded there.
These runs predate the filter (commit `add8e2d`); the current code reproduces them with `--no-filter`.

```
python attacks/summarize_results.py undefended=additional/nofilter/undefended.jsonl no-delims=additional/nofilter/no-delims.jsonl ours_s300=additional/nofilter/ours_s300.jsonl ours_s600=additional/nofilter/ours_s600.jsonl ours_s1163=additional/nofilter/ours_s1163.jsonl --ref ours_s1163 --out additional/nofilter/asr.csv
```
