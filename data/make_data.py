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

"""

import argparse
from datasets import load_dataset
from functools import cache

DATASET = "Sizhe-Chen/Llama-3.1-8B-Instruct-SecPO-trainset"
INPUT_HEADER = "<|start_header_id|>input<|end_header_id|>\n\n"
EOT = "<|eot_id|>"


@cache
def make_delims(n):
    # n == 0 keeps the SecPO prompt unchanged (baseline)
    assert n >= 0 and n % 2 == 0, "n must be a non-negative even number"
    
    opens = tuple(f"<|data_open_{i}|>" for i in range(n // 2))
    closes = tuple(f"<|data_close_{i}|>" for i in range(n // 2))
    return opens, closes


def split_input(prompt):
    """Split prompt to insert delimeters"""
    assert prompt.count(INPUT_HEADER) == 1, "expected exactly one input header"

    head, rest = prompt.split(INPUT_HEADER)
    data, tail = rest.split(EOT, 1)
    # delimiters are placed around {data}
    return head + INPUT_HEADER, data, EOT + tail


def add_delims(prompt, n):
    before, data, after = split_input(prompt)
    open_delims, close_delims = make_delims(n)
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
    """Return an error message, or None if the row is usable."""
    p, inj = ex["prompt"], ex["rejected_input_whole"]
    if p.count(INPUT_HEADER) != 1:
        return "input header count != 1"
    if p.count(inj) != 1:
        return "injection count != 1"
    _, data, _ = split_input(p)
    if inj not in data:
        return "injection not inside input data"
    if ex["chosen_data"] not in data:
        return "chosen_data not inside input data"
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2)
    parser.add_argument("--out", default="data/processed/secpo_n{n}")
    args = parser.parse_args()

    ds = load_dataset(DATASET, split="train")
    opens, closes = make_delims(args.n)

    errors = {}
    for i, ex in enumerate(ds):
        err = check_row(ex)
        if err:
            errors.setdefault(err, []).append(i)
    n_bad = sum(len(v) for v in errors.values())
    print(f"format check: {len(ds) - n_bad}/{len(ds)} ok")
    for err, idx in errors.items():
        print(f"  {err}: {len(idx)} rows (first: {idx[:5]})")

    bad = {i for idx in errors.values() for i in idx}
    ds = ds.filter(lambda _, i: i not in bad, with_indices=True)

    def convert(ex):
        return {
            "attacked": add_delims(ex["prompt"], opens, closes),
            "clean": clean_prompt(ex["prompt"], ex["rejected_input_whole"]),
            "chosen": ex["chosen"],
            "rejected": ex["rejected"],
        }

    out = ds.map(convert, remove_columns=ds.column_names)
    path = args.out.format(n=args.n)
    out.save_to_disk(path)
    print(f"saved {len(out)} rows to {path}")

    ex = out[0]
    print("\n===== attacked (with delimiters) =====")
    print(ex["attacked"])
    print("\n===== clean (reference input) =====")
    print(ex["clean"])


if __name__ == "__main__":
    main()