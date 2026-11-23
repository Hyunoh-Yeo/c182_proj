"""Run SEP-based prompt-injection eval against any baseline.

Usage (Mac debug):
    python eval/run_eval.py undefended --model meta-llama/Llama-3.2-1B-Instruct --limit 20

Usage (CUDA, full):
    python eval/run_eval.py secpo --checkpoint Sizhe-Chen/Llama-3.1-8B-Instruct-SecPO
    python eval/run_eval.py defensivetoken --embeddings defensivetokens.json

Writes per-row results to eval/results/<baseline>_<timestamp>.csv and prints ASR.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baselines import load  # noqa: E402

from eval.metrics import attack_success_rate, witness_hit  # noqa: E402

SEP_PATH = Path("data/SEP_dataset.json")
RESULTS_DIR = Path("eval/results")


def load_sep(path: Path = SEP_PATH, limit: int | None = None) -> list[dict]:
    with open(path) as f:
        rows = json.load(f)
    if limit is not None:
        rows = rows[:limit]
    return rows


def run_eval(
    defense,
    rows: list[dict],
    max_new_tokens: int = 256,
    attacked: bool = True,
) -> list[dict]:
    """Run the defense on each SEP row. Returns a list of per-row result dicts."""
    results = []
    for i, row in enumerate(rows):
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
        print(f"  [{i + 1}/{len(rows)}] hit={hit} running ASR={running_asr:.3f}")
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
    ap.add_argument("name", choices=["undefended", "secpo", "defensivetoken"])
    ap.add_argument("--model", help="base model id (undefended / defensivetoken)")
    ap.add_argument("--checkpoint", help="SecPO checkpoint path or HF id")
    ap.add_argument("--embeddings", help="DefensiveToken vectors JSON")
    ap.add_argument("--four-bit", action="store_true")
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
    if args.name == "secpo":
        if not args.checkpoint:
            ap.error("secpo needs --checkpoint")
        kw["checkpoint"] = args.checkpoint
    if args.name == "defensivetoken":
        if not args.embeddings:
            ap.error("defensivetoken needs --embeddings")
        kw["embeddings_path"] = args.embeddings

    print(f"loading baseline: {args.name}")
    defense = load(args.name, **kw)

    print(f"loading SEP from {SEP_PATH}")
    rows = load_sep(limit=args.limit)
    print(f"evaluating {len(rows)} rows (mode={args.mode})")

    results = run_eval(
        defense,
        rows,
        max_new_tokens=args.max_new_tokens,
        attacked=(args.mode == "attacked"),
    )
    asr = attack_success_rate([r["hit"] for r in results])
    out_path = save_results(results, args.name, args.mode)

    print(f"\n{'=' * 50}")
    print(f"baseline:      {args.name}")
    print(f"mode:          {args.mode}")
    print(f"rows:          {len(rows)}")
    print(f"ASR / hit rate: {asr:.4f}")
    print(f"results:       {out_path}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()