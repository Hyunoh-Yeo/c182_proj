"""Run SEP-based prompt-injection eval against any baseline.

Usage (Mac debug):
    python eval/run_eval.py undefended --model meta-llama/Llama-3.2-1B-Instruct --limit 20

Usage (CUDA, full):
    python eval/run_eval.py secpo --checkpoint Sizhe-Chen/Llama-3.1-8B-Instruct-SecPO
    python eval/run_eval.py defensivetoken --embeddings defensivetokens.json
    python eval/run_eval.py delims --checkpoint checkpoints/insert_n2_lr0.1_ga16/delims_step1163.pt
    python eval/run_eval.py delims --checkpoint ... --no-delims       # same template, no delimiters
    python eval/run_eval.py delims --checkpoint ... --no-filter       # don't strip special tokens from data

Use the same --sample N --seed S for every defense so all of them see the same rows. SEP is
ordered by task type, so --limit N (first N rows) is only for debugging, not for results.

Writes per-row results to eval/results/<baseline>_<mode>_<timestamp>.csv and prints ASR.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines import load  # noqa: E402

from eval.metrics import attack_success_rate, witness_hit  # noqa: E402

SEP_PATH = ROOT / "data" / "SEP_dataset.json"
RESULTS_DIR = ROOT / "eval" / "results"


def load_sep(path: Path = SEP_PATH, limit: int | None = None, sample: int | None = None,
             seed: int = 0) -> list[tuple[int, dict]]:
    """Returns (original SEP index, row) pairs. `sample` draws a seeded random subset."""
    with open(path) as f:
        rows = list(enumerate(json.load(f)))
    if sample is not None:
        rows = [rows[i] for i in sorted(random.Random(seed).sample(range(len(rows)), sample))]
    if limit is not None:
        rows = rows[:limit]
    return rows


def run_eval(
    defense,
    rows: list[tuple[int, dict]],
    max_new_tokens: int = 256,
    attacked: bool = True,
) -> list[dict]:
    """Run the defense on each SEP row. Returns a list of per-row result dicts."""
    results = []
    for n, (i, row) in enumerate(rows):
        instruction = row["system_prompt_clean"]
        data = row["prompt_instructed"] if attacked else row["prompt_clean"]
        witness = row["witness"]

        output = defense.generate(
            system=None,
            instruction=instruction,
            data=data,
            max_new_tokens=max_new_tokens,
        )
        hit = witness_hit(output, witness)
        results.append(
            {
                "idx": i,
                "witness": witness,
                "hit": hit,
                "output": output[:500],  # truncate for CSV sanity
            }
        )
        running_asr = attack_success_rate([r["hit"] for r in results])
        print(f"  [{n + 1}/{len(rows)}] row {i} hit={hit} running ASR={running_asr:.3f}", flush=True)
    return results


def save_results(results: list[dict], baseline_name: str, mode: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"{baseline_name}_{mode}_{ts}.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["idx", "witness", "hit", "output"])
        writer.writeheader()
        writer.writerows(results)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", choices=["undefended", "secpo", "defensivetoken", "delims"])
    ap.add_argument("--model", help="base model id (undefended / defensivetoken / delims)")
    ap.add_argument("--checkpoint", help="secpo: path or HF id; delims: delims_step<N>.pt")
    ap.add_argument("--embeddings", help="DefensiveToken vectors JSON")
    ap.add_argument("--no-delims", action="store_true", help="delims: drop the delimiters (ablation)")
    ap.add_argument("--no-filter", action="store_true",
                    help="delims: don't strip Llama control tokens / our delimiters from the data")
    ap.add_argument("--four-bit", action="store_true")
    ap.add_argument("--sample", type=int, default=None, help="seeded random subset of N rows (use for results)")
    ap.add_argument("--seed", type=int, default=0, help="seed for --sample")
    ap.add_argument("--limit", type=int, default=None, help="only run first N rows (debug)")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--mode", choices=["attacked", "clean"], default="attacked",
                    help="attacked = prompt with probe (ASR); clean = no probe (utility)")
    args = ap.parse_args()

    kw = {"four_bit": args.four_bit}
    if args.model:
        if args.name == "secpo":
            ap.error("--model does not apply to secpo; use --checkpoint")
        kw["model_id"] = args.model
    if args.name in ("secpo", "delims"):
        if not args.checkpoint:
            ap.error(f"{args.name} needs --checkpoint")
        kw["checkpoint"] = args.checkpoint
    if args.name == "defensivetoken":
        if not args.embeddings:
            ap.error("defensivetoken needs --embeddings")
        kw["embeddings_path"] = args.embeddings
    if args.name == "delims":
        kw["use_delims"] = not args.no_delims
        kw["filter_data"] = not args.no_filter
    elif args.no_delims or args.no_filter:
        ap.error("--no-delims and --no-filter only apply to delims")
    tag = args.name + ("_nodelims" if args.no_delims else "") + ("_nofilter" if args.no_filter else "")

    print(f"loading SEP from {SEP_PATH}")
    rows = load_sep(limit=args.limit, sample=args.sample, seed=args.seed)

    print(f"loading baseline: {tag}")
    defense = load(args.name, **kw)
    print(f"evaluating {len(rows)} rows (mode={args.mode})")

    results = run_eval(
        defense,
        rows,
        max_new_tokens=args.max_new_tokens,
        attacked=(args.mode == "attacked"),
    )
    asr = attack_success_rate([r["hit"] for r in results])
    out_path = save_results(results, tag, args.mode)

    print(f"\n{'=' * 50}")
    print(f"baseline:      {tag}")
    print(f"mode:          {args.mode}")
    print(f"rows:          {len(rows)}")
    print(f"ASR / hit rate: {asr:.4f}")
    print(f"results:       {out_path}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()