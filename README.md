# Parameter-Efficient Prompt Injection Defense via Learnable Delimiter Embeddings

CS 182 final project. We freeze Llama-3.1-8B-Instruct and train only two delimiter embeddings (8,192
parameters) that wrap the untrusted data, using SecPO's training data and loss unchanged. We compare against
the released SecPO checkpoint, DefensiveToken, the undefended model, and a no-delimiter ablation on
AlpacaFarm prompt injections and SEP.

The paper source is in `paper/` (`paper/main.tex`). The results in it come from files in `results/`,
`additional/` and `checkpoints/`, and every table and figure can be rebuilt from those files without a GPU.

Status: this is the V1 draft. For the final submission we plan a variant that inserts no new tokens and
instead trains the template tokens around the data (see the discussion section of the paper).

**Where to start.** Read the results below, then [Setup](#setup) and step 1 of [Reproduce](#reproduce),
which rebuilds every number in the paper from the saved outputs in about a minute without a GPU. Steps 2 to
6 rerun the data, training and evaluation from scratch.

## Results

All runs use Llama-3.1-8B-Instruct in 4-bit NF4 with greedy decoding, the same attacks and the same samples
for every defense, and a single training seed. Brackets are 95% Wilson intervals.

| Defense | Trainable parameters | AlpacaFarm, any of 10 attacks | SEP, whole-word witness | Utility win rate vs GPT-4 Turbo |
|---|---|---|---|---|
| Undefended | 0 | 95.7% (199/208) | 42.4% (212/500) | 23.1% [17.9, 29.3] |
| No delimiters (our template, no vectors) | 0 | 94.2% (196/208) | 28.0% (140/500) | 21.6% [16.6, 27.7] |
| DefensiveToken (5 vectors) | 20,480 | 9.6% (20/208) | 2.2% (11/500) | 21.2% [16.2, 27.2] |
| SecPO (LoRA) | about 140M | 0.0% (0/208) | 1.8% (9/500) | 14.4% [10.3, 19.8] |
| **Ours (2 vectors)** | **8,192** | **0.0% (0/208)** [0, 1.8] | **0.6% (3/500)** [0.2, 1.7] | **10.6%** [7.1, 15.5] |

- On AlpacaFarm, our two vectors match SecPO (both 0/208); DefensiveToken is broken on 9.6% of samples.
- On SEP, ours is the lowest, but its interval overlaps those of SecPO and DefensiveToken.
- With the filter that removes chat-template control tokens from the data turned off, ours rises only to
  1.9% on AlpacaFarm (`additional/nofilter/`), so the robustness comes mainly from the delimiters.
- The delimiters cost utility on benign inputs: 10.6% against 21.6% for the same prompt without them.

The numbers come from `results/asr.csv`, `results/sep_asr.csv` and
`results/utility/gpt-5.6-sol_512/win_rate.csv`.

## From the paper to the files

Every number in the paper comes from a file in this repository and can be rebuilt with step 1 of
[Reproduce](#reproduce) (no GPU). The table says where each result lives and which README explains it.

| In the paper | Rebuilt by | Raw files | Details in |
|---|---|---|---|
| Figure 2 (left), Table 2 (App. C): AlpacaFarm ASR, any attack and per attack | `attacks/summarize_results.py`, `paper/make_figures.py` | `results/attacks/` | `results/README.md` |
| Figure 2 (right), Table 3 (App. C): SEP | `eval/summarize_sep.py`, `paper/make_figures.py` | `results/sep/` | `results/README.md`, `eval/README.md` |
| Table 1: utility win rates | `eval/utility_judge.py` (reuses the saved judgments) | `results/utility/` | `eval/README.md` |
| Table 5 (App. D), Section 5.3: data filter off vs on | `attacks/summarize_results.py` | `additional/nofilter/` | `additional/nofilter/README.md` |
| Figure 3, Section 5.2: training loss and checkpoint ASR | `paper/make_figures.py` | `checkpoints/insert_n2_lr0.1_ga16/log.jsonl`, `additional/nofilter/` | `paper/README.md` |
| Figure 4, Table 7 (App. H): learning-rate and batch-size sweep | `paper/make_sweep_figure.py` | `results/sweep/` | `paper/README.md` |
| Figure 5 (App. J): delimiter norms | `paper/make_figures.py` | `checkpoints/insert_n2_lr0.1_ga16/log.jsonl` | `paper/README.md` |
| Appendix A: prompt templates of every defense | `baselines/` | | `baselines/README.md` |

Which README to open for what:

| README | What it covers |
|---|---|
| `results/README.md` | All result tables with counts, 95% Wilson intervals and p-values, and which run produced each file. |
| `eval/README.md` | SEP and utility evaluation commands, and how our utility judge differs from AlpacaEval 2.0. |
| `baselines/README.md` | How each of the five defenses is loaded (template, data filter, checkpoint) and the choices we made. |
| `additional/nofilter/README.md` | The ablation with the data filter turned off, with its numbers and command. |
| `paper/README.md` | How to rebuild the figures and tables and compile the paper. |

## Layout

| Path | What it holds |
|---|---|
| `data/make_data.py` | Builds our training set from SecPO's released training data (`Sizhe-Chen/Llama-3.1-8B-Instruct-SecPO-trainset` on Hugging Face). |
| `train/train_delims.py` | Trains the delimiter embeddings with the SecPO loss on the frozen model. |
| `train/sweep.py` | Learning-rate and batch-size sweep on a 2,000-row subset. |
| `baselines/` | One `load(name, ...)` interface for all five defenses: `undefended`, `secpo`, `defensivetoken`, `delims` (ours) and `delims --no-delims`. See `baselines/README.md`. |
| `attacks/` | AlpacaFarm injection attacks (`run_attacks.py`) and the ASR summary (`summarize_results.py`). |
| `eval/` | SEP evaluation (`run_eval.py`, `summarize_sep.py`) and the utility judge (`utility_judge.py`). See `eval/README.md`. |
| `checkpoints/insert_n2_lr0.1_ga16/` | Our trained delimiters at steps 300, 600 and 1163 (final), the training config and the training log. |
| `results/` | Raw outputs and summaries behind every table in the paper. See `results/README.md`. |
| `additional/nofilter/` | The same AlpacaFarm runs with the data filter turned off (an ablation). |
| `paper/` | LaTeX source, figures, tables and the scripts that make them. See `paper/README.md`. |

## Setup

All our runs used Windows with one RTX 5070 Ti laptop GPU (12 GB) and the CUDA 12.8 build of PyTorch.
Every run loads the model in 4-bit NF4 (`--four-bit` / `--load_in_4bit`).

```
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
hf auth login
```

Llama-3.1-8B-Instruct is gated: request access on its Hugging Face page before `hf auth login`. The SecPO
checkpoint (`Sizhe-Chen/Llama-3.1-8B-Instruct-SecPO`) is downloaded automatically. DefensiveToken needs its
released vectors, which we do not redistribute:

```
curl -LO https://raw.githubusercontent.com/Sizhe-Chen/DefensiveToken/main/defensivetokens.json
```

## Reproduce

Each step only needs the files from the steps before it, and the repo already contains the outputs of every
step, so you can start at any step. Times are for the GPU above.

**1. Tables and figures from saved outputs (no GPU, about a minute).**

```
python attacks/summarize_results.py undefended=results/attacks/undefended.jsonl no-delims=results/attacks/no-delims.jsonl secpo=results/attacks/secpo.jsonl defensivetoken=results/attacks/defensivetoken.jsonl ours_s300=results/attacks/ours_s300.jsonl ours_s600=results/attacks/ours_s600.jsonl ours_s1163=results/attacks/ours_s1163.jsonl --ref ours_s1163 --out results/asr.csv
python eval/summarize_sep.py undefended=results/sep/undefended.csv no-delims=results/sep/no-delims.csv secpo=results/sep/secpo.csv defensivetoken=results/sep/defensivetoken.csv ours=results/sep/ours.csv --ref ours --out results/sep_asr.csv
python paper/make_figures.py
python paper/make_sweep_figure.py
python eval/utility_judge.py ours=results/utility/clean_512/ours.jsonl undefended=results/utility/clean_512/undefended.jsonl secpo=results/utility/clean_512/secpo.jsonl defensivetoken=results/utility/clean_512/defensivetoken.jsonl no-delims=results/utility/clean_512/no-delims.jsonl --ref ours --out-dir results/utility/gpt-5.6-sol_512
```

The last command rebuilds the utility table from the saved judgments in `results/utility/gpt-5.6-sol_512/annotations.jsonl`, so it needs neither a GPU nor an OpenAI API key. It downloads the AlpacaFarm samples and GPT-4 Turbo's reference outputs on the first run.

**2. Training data (downloads SecPO's training set).**

```
python data/make_data.py --n 2
```

**3. Training (about 10.8 h, 1,163 steps).** These are the settings of the reported run, as recorded in
`checkpoints/insert_n2_lr0.1_ga16/config.json`. Several of them differ from the script's defaults, so pass
all of them.

```
python train/train_delims.py --model meta-llama/Llama-3.1-8B-Instruct --load_in_4bit --n 2 --lr 0.1 --grad_accum 16 --init std_normal --seed 42 --out "outputs/final_n2_lr{lr}_ga{ga}_{init}"
```

The run writes `delims_step<N>.pt` and `log.jsonl` to `outputs/final_n2_lr0.1_ga16_std_normal/`. The
checkpoints in `checkpoints/insert_n2_lr0.1_ga16/` are copies of that run's steps 300, 600 and 1163.

The sweep behind the learning rate and batch size (`results/sweep/`) is `python train/sweep.py lr` followed by
`python train/sweep.py ga 1e-1`.

**4. AlpacaFarm attacks (1.2 to 2.7 h per defense, 7.3 h for DefensiveToken).** Outputs go to
`outputs/attacks/`; the copies in `results/attacks/` are what the paper uses.

```
python attacks/run_attacks.py delims --checkpoint checkpoints/insert_n2_lr0.1_ga16/delims_step1163.pt --attacks all --four-bit
python attacks/run_attacks.py delims --checkpoint checkpoints/insert_n2_lr0.1_ga16/delims_step1163.pt --no-delims --attacks all --four-bit
python attacks/run_attacks.py secpo --checkpoint Sizhe-Chen/Llama-3.1-8B-Instruct-SecPO --attacks all --four-bit
python attacks/run_attacks.py defensivetoken --embeddings defensivetokens.json --attacks all --four-bit
python attacks/run_attacks.py undefended --attacks all --four-bit
```

For the step-300 and step-600 rows, run the first command with `delims_step300.pt` and `delims_step600.pt`.
For `additional/nofilter/`, add `--no-filter` to the `delims` and `undefended` commands.

**5. SEP (about 1 h per defense).** The same 500 rows for every defense (`--sample 500 --seed 42`). Outputs
go to `eval/results/`.

```
python eval/run_eval.py delims --checkpoint checkpoints/insert_n2_lr0.1_ga16/delims_step1163.pt --four-bit --sample 500 --seed 42
python eval/run_eval.py delims --checkpoint checkpoints/insert_n2_lr0.1_ga16/delims_step1163.pt --no-delims --four-bit --sample 500 --seed 42
python eval/run_eval.py secpo --checkpoint Sizhe-Chen/Llama-3.1-8B-Instruct-SecPO --four-bit --sample 500 --seed 42
python eval/run_eval.py defensivetoken --embeddings defensivetokens.json --four-bit --sample 500 --seed 42
python eval/run_eval.py undefended --four-bit --sample 500 --seed 42
```

**6. Utility.** Clean (no injection) outputs with a longer generation limit, then an AlpacaEval-style judge.
The judge calls the OpenAI API and needs `OPENAI_API_KEY`; judging all five defenses costs a few US dollars.
See `eval/README.md` for how this differs from AlpacaEval 2.0.

```
python attacks/run_attacks.py delims --checkpoint checkpoints/insert_n2_lr0.1_ga16/delims_step1163.pt --attacks none --max-new-tokens 512 --four-bit --out outputs/utility
(the same for the other four defenses, with the flags from step 4)
python eval/utility_judge.py ours=<ours .jsonl> undefended=<...> secpo=<...> defensivetoken=<...> no-delims=<...> --ref ours --out-dir results/utility/gpt-5.6-sol_512
```

The judge saves every judgment in `annotations.jsonl`, so the win-rate table can be rebuilt from the saved
file without calling the API again (rerun the same command; cached judgments are reused).

## V2: template-token variant (in progress)

V1 (the paper) inserts two new delimiter vectors around the data. V2, suggested by our TA, inserts nothing:
the chat-template tokens around the data (`<|start_header_id|>`, `input`, `<|end_header_id|>` before it and
`<|eot_id|>` after it) get trainable embeddings at those positions only, starting from their original
embeddings, so training starts exactly at the undefended model on SecPO's prompt. The same tokens elsewhere
in the prompt stay frozen. V1 files are unchanged; V2 lives in files ending in `_v2`.

| Step | V1 | V2 |
|---|---|---|
| Training data | `data/make_data.py` | `data/make_data_v2.py` |
| Training | `train/train_delims.py` | `train/train_delims_v2.py` |
| Learning-rate sweep | `train/sweep.py` → `results/sweep/` | `train/sweep_v2.py` → `results/sweep_v2/` |
| Loading for evaluation | `baselines/delims.py` (`delims`) | `baselines/delims_v2.py` (`delims_v2`) |

```
python data/make_data_v2.py
python train/train_delims_v2.py --model meta-llama/Llama-3.1-8B-Instruct --load_in_4bit --lr 1e-3 --grad_accum 16 --seed 42
python attacks/run_attacks.py delims_v2 --checkpoint outputs/v2_all_lr0.001_ga16_template/delims_v2_step<N>.pt --attacks all --four-bit
python eval/run_eval.py delims_v2 --checkpoint outputs/v2_all_lr0.001_ga16_template/delims_v2_step<N>.pt --four-bit --sample 500 --seed 42
```

`--tokens header,eot` (in both `make_data_v2.py` and `train_delims_v2.py`) trains only some of the four
template tokens, and `--init std_normal` starts from N(0, I) as V1 does. The full run uses lr 1e-3, not V1's 0.1:
at 0.1 the validation loss blows up within 3 steps (`results/sweep_v2/`). Results are not in yet.

## Notes for reproducing

- Greedy decoding throughout, so reruns of steps 4 to 6 should give the same outputs on the same hardware
  and library versions. 4-bit kernels can differ across GPUs, so small differences elsewhere are possible.
- The undefended model, the no-delimiter ablation and ours remove the eight Llama-3 control tokens from the
  untrusted data, the same filter as SecPO's released chat template. SecPO and DefensiveToken use their
  released filters.
- The judge model of step 6 is an API model and may change or be retired; the saved judgments are the
  record of what we measured.
