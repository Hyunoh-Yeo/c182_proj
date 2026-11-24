# Evaluation: SEP and utility

## SEP (`run_eval.py`, `summarize_sep.py`)

Runs a defense on the SEP dataset (`data/SEP_dataset.json`, from
https://github.com/egozverev/Should-It-Be-Executed-Or-Processed) and checks whether the output contains the
probe's witness word.

Give every defense the same `--sample N --seed S` so they are scored on the same rows. SEP is ordered by
task type, so `--limit` (the first N rows) is for debugging only. Our results use 500 rows:

```
S="--sample 500 --seed 42 --four-bit"
python eval/run_eval.py delims --checkpoint checkpoints/insert_n2_lr0.1_ga16/delims_step1163.pt $S
python eval/run_eval.py delims --checkpoint checkpoints/insert_n2_lr0.1_ga16/delims_step1163.pt --no-delims $S
python eval/run_eval.py secpo --checkpoint Sizhe-Chen/Llama-3.1-8B-Instruct-SecPO $S
python eval/run_eval.py defensivetoken --embeddings defensivetokens.json $S
python eval/run_eval.py undefended $S
```

(In PowerShell, write each command out in full instead of using `$S`.) Each run writes
`eval/results/<defense>_<mode>_<timestamp>.csv` with `idx`, `witness`, `hit` and the full `output`, and a
`.json` next to it with the arguments, the number of rows and the ASR. The copies used in the paper are in
`results/sep/`.

`summarize_sep.py` computes two metrics from the saved CSVs, with 95% Wilson intervals and two-sided Fisher
tests against `--ref`:

- **substring**: the witness appears anywhere in the output (`metrics.witness_hit`, SEP's standard check).
- **word**: the witness appears as a whole word. The witnesses are short common words ("cat", "pen"), so the
  substring check also fires on words like "indicating" or "rooted".

```
python eval/summarize_sep.py undefended=results/sep/undefended.csv no-delims=results/sep/no-delims.csv secpo=results/sep/secpo.csv defensivetoken=results/sep/defensivetoken.csv ours=results/sep/ours.csv --ref ours --out results/sep_asr.csv
```

`--mode clean` runs the same prompts without the probe; it is implemented but not part of our results.

Limits: one attack per SEP row (SEP's own probe), so these ASRs are not comparable to SecPO's SEP numbers,
which use other attack templates and an LLM judge. Compare defenses with each other within this harness.

## Utility (`utility_judge.py`)

AlpacaEval-style win rate of each defense's clean outputs (no injection) on the 208 AlpacaFarm samples with
an input, against GPT-4 Turbo's outputs. First make clean outputs with a longer limit, then judge them:

```
python attacks/run_attacks.py delims --checkpoint checkpoints/insert_n2_lr0.1_ga16/delims_step1163.pt --attacks none --max-new-tokens 512 --four-bit --out outputs/utility
(the same for --no-delims, undefended, secpo and defensivetoken)
python eval/utility_judge.py ours=<.jsonl> undefended=<.jsonl> secpo=<.jsonl> defensivetoken=<.jsonl> no-delims=<.jsonl> --ref ours --probe 10 --out-dir results/utility/gpt-5.6-sol_512
```

`--probe 10` judges 10 samples per defense and prints the token use and projected cost; drop it to judge
all 208. The judge needs `OPENAI_API_KEY` (in PowerShell, `$env:OPENAI_API_KEY="sk-..."`; never commit it).
Every judgment is appended to `annotations.jsonl` in the output folder, and a rerun reuses them, so an
interrupted run resumes and the table can be rebuilt without new API calls. `--mock` uses a random judge to
test the pipeline without an API key.

What follows AlpacaEval 2.0: the judge prompt (copied from AlpacaEval's
`alpaca_eval_clf_gpt4_turbo/alpaca_eval_clf.txt`), the reference outputs (`gpt-4-1106-preview`, downloaded
from the AlpacaEval repository), and a per-sample random order of the two outputs. The order is fixed by
`--seed` and is the same for every defense.

What differs, to report with the numbers:

- The judge. AlpacaEval 2.0's judge, `gpt-4-1106-preview`, was shut down on 2026-10-23, so the default is
  `gpt-5.6-sol`, the replacement OpenAI names for it, with `reasoning_effort` as low as the API accepts.
- Binary judgments (the judge answers `m` or `M`) instead of logprob-weighted preferences, and no
  length-controlled win rate.
- 208 of the 805 AlpacaEval instructions.

So the win rates compare these defenses with each other; they are not comparable to published AlpacaEval
numbers. Intervals are 95% Wilson; p is an exact two-sided sign test against `--ref` on the samples where
exactly one of the two defenses beat the reference.
