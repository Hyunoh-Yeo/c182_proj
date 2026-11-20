# data/inspect_bad_rows.py
"""
Inspect rows rejected by make_data.py: per error type, print counts and simple patterns,
and write every bad row to a text file for reading.
"""
import json
from collections import Counter
from datasets import load_dataset

DATASET = "Sizhe-Chen/Llama-3.1-8B-Instruct-SecPO-trainset"
BAD = "data/processed/secpo_n2_bad_rows.json"
OUT = "data/processed/secpo_n2_bad_rows.txt"
SPECIALS = ["<|eot_id|>", "<|start_header_id|>", "<|end_header_id|>", "<|begin_of_text|>"]

ds = load_dataset(DATASET, split="train")
bad = json.load(open(BAD))

with open(OUT, "w") as f:
    for err, idx in bad.items():
        pattern = Counter()
        f.write(f"\n\n######## {err}: {len(idx)} rows ########\n")
        for i in idx:
            ex = ds[i]
            inj, p = ex["rejected_input_whole"], ex["prompt"]
            pattern["injection has special tokens"] += any(s in inj for s in SPECIALS)
            pattern["no input role"] += "<|start_header_id|>input<|end_header_id|>" not in p
            pattern["inject_at_the_end"] += ex["inject_at_the_end"]
            f.write(f"\n=== row {i} ===\n--- prompt ---\n{p}\n"
                    f"--- rejected_input_whole ---\n{inj}\n"
                    f"--- chosen_data ---\n{ex['chosen_data']}\n")

        print(f"\n######## {err}: {len(idx)} rows ########")
        for k, v in pattern.items():
            print(f"  {k}: {v}/{len(idx)}")

print(f"\nall bad rows written to {OUT}")