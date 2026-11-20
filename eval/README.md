# Eval harness

Runs a baseline defense against the SEP dataset, computes ASR (attack success rate)
via witness string match.

## Setup

SEP dataset at `data/SEP_dataset.json`. Get it from:
https://github.com/egozverev/Should-It-Be-Executed-Or-Processed

## Quick debug (Mac, Llama-3.2-1B, 20 rows)

```bash
python eval/run_eval.py undefended --model meta-llama/Llama-3.2-1B-Instruct --limit 20
```

## Full baseline eval (CUDA, Llama-3.1-8B)

```bash
# Undefended baseline
python eval/run_eval.py undefended

# SecPO released checkpoint
python eval/run_eval.py secpo --checkpoint Sizhe-Chen/Llama-3.1-8B-Instruct-SecPO

# DefensiveToken
curl -LO https://raw.githubusercontent.com/Sizhe-Chen/DefensiveToken/main/defensivetokens.json
python eval/run_eval.py defensivetoken --embeddings defensivetokens.json
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
  is not implemented yet.
- Witness check only, no LLM judge. May overcount successes when the witness string
  appears coincidentally. Can bolt on judge later.