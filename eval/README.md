# Eval harness

Runs a baseline defense against the SEP dataset, computes ASR (attack success rate)
via witness string match.

## Setup

SEP dataset at `data/SEP_dataset.json`. Get it from:
https://github.com/egozverev/Should-It-Be-Executed-Or-Processed

## Quick debug (Mac, Llama-3.2-1B)

```bash
python eval/run_eval.py undefended --model meta-llama/Llama-3.2-1B-Instruct --sample 12
```

## Baseline eval (CUDA, Llama-3.1-8B)

Give every defense the **same `--sample N --seed S`** so they are scored on the same rows. SEP is
ordered by task type, so `--limit` (first N rows) is for debugging only. The full 9,160 rows at
256 new tokens take on the order of a day per defense on a 12 GB GPU, so use a subset (SecPO's
adaptive evals use 1,024 rows).

```bash
S="--sample 1024 --seed 0 --four-bit"
python eval/run_eval.py undefended $S
python eval/run_eval.py secpo --checkpoint Sizhe-Chen/Llama-3.1-8B-Instruct-SecPO $S
curl -LO https://raw.githubusercontent.com/Sizhe-Chen/DefensiveToken/main/defensivetokens.json
python eval/run_eval.py defensivetoken --embeddings defensivetokens.json $S
python eval/run_eval.py delims --checkpoint checkpoints/insert_n2_lr0.1_ga16/delims_step1163.pt $S
python eval/run_eval.py delims --checkpoint checkpoints/insert_n2_lr0.1_ga16/delims_step1163.pt --no-delims $S
```

## Utility mode (same prompts, no probe)

```bash
python eval/run_eval.py secpo --checkpoint <path> --mode clean
```

Hit rate in clean mode is a utility floor: witness answers should almost never appear
in the clean output (if they do, the data already contained the answer or the model
is hallucinating).

## Output

CSV at `eval/results/<baseline>_<mode>_<timestamp>.csv` with per-row `idx`, `witness`,
`hit` (bool), and truncated `output`. Aggregate ASR printed at end.

## Known limits (v1)

- One attack per SEP row (SEP's baked-in probe). SecPO's paper max-over-6-templates
  is not implemented yet, so **our SEP ASRs are not comparable to the paper's numbers**
  (e.g. undefended 98.1%). Compare defenses against each other within this harness only.
- Witness check only, no LLM judge. May overcount successes when the witness string
  appears coincidentally. Can bolt on judge later.