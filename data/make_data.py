"""
Create training data by adding learnable delimiters to SecPO training data.

Each row of data comprises:
1. attacked: prompt with n delimiters inserted around the input data
    <|start_header_id|>input<|end_header_id|>\n\n{token 1} ... {token n/2} {data_with_injection} {token n/2 + 1} ... {token n}<|eot_id|>
2. clean: clean prompt restored the same way as SecPO (no delimiters)
    prompt.replace(rejected_input_whole, "", 1)
3. chosen (copied)
4. rejected (copied)

Usage:
  python data/make_data.py --n 2

== prompt template == 
<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|>
<|start_header_id|>input<|end_header_id|>\n\n{data_with_injection}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>\n\n

<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|>
<|special_delimiter1|><|special_delimiter2|><|special_delimiter3|>\n\n{data_with_injection}<|special_delimiter4|>
<|start_header_id|>assistant<|end_header_id|>\n\n

1. initialize <|special_delimiter1|> with <|start_header_id|>, initialize <|special_delimiter2|> with input, ..., <|special_delimiter4|> with <|eot_id|> 
2. train the three new embeddings

# low priority: tuning hyper parameters, tuning #special delimiters
"""

import json
import os
import argparse
from datasets import load_dataset
from functools import cache

DATASET = "Sizhe-Chen/Llama-3.1-8B-Instruct-SecPO-trainset"

PREFIX = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
MIDDLE = "<|eot_id|><|start_header_id|>input<|end_header_id|>\n\n"
END = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"


RESERVED_BASE = 200

@cache
def make_delims(n):
    """Delimiters are unused Llama 3 reserved special tokens, so each is a single token id."""
    # n == 0 keeps the SecPO prompt unchanged (baseline)
    assert n >= 0 and n % 2 == 0, "n must be a non-negative even number"
    names = tuple(f"<|reserved_special_token_{RESERVED_BASE + k}|>" for k in range(n))
    return names[: n // 2], names[n // 2 :]


def split_input(prompt):
    """Split prompt at the first MIDDLE and the final END to insert delimiters"""
    assert MIDDLE in prompt, "no input header"
    assert prompt.endswith(END), "prompt does not end with the assistant header"

    head, rest = prompt.split(MIDDLE, 1)
    data = rest[: -len(END)]
    # delimiters are placed around {data}
    return head + MIDDLE, data, END


def add_delims(prompt, n):
    before, data, after = split_input(prompt)
    open_delims, close_delims = make_delims(n)
    assert not any(d in prompt for d in open_delims + close_delims), "delimiter token already in prompt"
    return before + "".join(open_delims) + data + "".join(close_delims) + after


def clean_prompt(prompt, injection):
    """Same as SecPO's derive_clean_prompt (src/secpo/data.py)."""
    assert isinstance(prompt, str) and prompt.strip(), "prompt must be a non-empty string"
    assert isinstance(injection, str) and injection, "injection must be a non-empty string"
    assert prompt.count(injection) == 1, "injection must occur exactly once"

    clean = prompt.replace(injection, "", 1)
    assert clean.strip(), "removing the injection left an empty prompt"
    return clean


def check_row(ex):
    """
    Return an error message, or None if the row is usable.
    Template: PREFIX {instruction} MIDDLE {data} END
    """
    p, inj = ex["prompt"], ex["rejected_input_whole"]
    if not p.startswith(PREFIX):
        return "prompt does not start with the user header"
    if not p.endswith(END):
        return "prompt does not end with the assistant header"
    if MIDDLE not in p[len(PREFIX): -len(END)]:
        return "no input header after the instruction"
    if p.count(inj) != 1:
        return "injection count != 1"
    _, data, _ = split_input(p)
    if inj not in data:
        return "injection not inside input data"
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2)
    parser.add_argument("--out", default="data/processed/secpo_n{n}")
    args = parser.parse_args()
    path = args.out.format(n=args.n)

    ds = load_dataset(DATASET, split="train")

    errors = {}
    for i, ex in enumerate(ds):
        err = check_row(ex)
        if err:
            errors.setdefault(err, []).append(i)
    n_bad = sum(len(v) for v in errors.values())
    print(f"format check: {len(ds) - n_bad}/{len(ds)} ok")
    for err, idx in errors.items():
        print(f"  {err}: {len(idx)} rows")
    if errors:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(f"{path}_bad_rows.json", "w") as f:
            json.dump(errors, f, indent=2)
        print(f"  bad row indices saved to {path}_bad_rows.json")

    bad = {i for idx in errors.values() for i in idx}
    ds = ds.filter(lambda _, i: i not in bad, with_indices=True)

    def convert(ex):
        return {
            "attacked": add_delims(ex["prompt"], args.n),
            "clean": clean_prompt(ex["prompt"], ex["rejected_input_whole"]),
            "chosen": ex["chosen"],
            "rejected": ex["rejected"],
        }

    out = ds.map(convert, remove_columns=ds.column_names)
    out.save_to_disk(path)
    print(f"saved {len(out)} rows to {path}")

    ex = out[0]
    print("\n===== attacked (with delimiters) =====")
    print(ex["attacked"])
    print("\n===== clean (reference input) =====")
    print(ex["clean"])


if __name__ == "__main__":
    main()