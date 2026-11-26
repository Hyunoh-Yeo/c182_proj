"""
V2 training data (the template-token variant suggested by our TA): insert nothing, and instead make the
chat-template tokens around the untrusted data trainable at those positions.

V1 (data/make_data.py) inserts n new delimiter tokens around the data:
    <|start_header_id|>input<|end_header_id|>\n\n<d_1>{data}<d_2><|eot_id|>
V2 keeps the SecPO prompt token for token, but swaps the template tokens around the data for placeholders:
    <t_1><t_2><t_3>\n\n{data}<t_4>
where t_1..t_4 stand for <|start_header_id|>, input, <|end_header_id|> (before the data) and <|eot_id|>
(after it). Each placeholder is an unused Llama 3 reserved special token, so it is a single token id, and
train/train_delims_v2.py starts its embedding from the embedding of the token it replaces. Before training,
the attacked prompt is therefore exactly the undefended model reading SecPO's prompt. The same tokens
elsewhere in the prompt (the user and assistant headers, the <|eot_id|> that ends the user turn) are not
replaced and stay frozen, and an attacker who writes <|eot_id|> into the data gets its original embedding.

Which of the four tokens become trainable is set with --tokens (default: all four). The placeholders use
reserved tokens 210-213, so they never collide with V1's delimiters (200, 201).

Each output row has
    attacked          the SecPO prompt with the chosen template tokens around the data replaced
    clean             the SecPO prompt with the injection removed (same as V1, no placeholders)
    chosen, rejected  copied from SecPO

Usage:
  python data/make_data_v2.py                       # all four template tokens
  python data/make_data_v2.py --tokens header,eot   # only <|start_header_id|> and <|eot_id|>
"""

import argparse
import json
import os
import sys

from datasets import load_dataset

sys.path.insert(0, os.path.dirname(__file__))
from make_data import DATASET, END, MIDDLE, check_row, clean_prompt, split_input  # noqa: E402

# The template tokens around the data, in prompt order. key -> original token text.
TEMPLATE = {
    "header": "<|start_header_id|>",
    "role": "input",
    "end_header": "<|end_header_id|>",
    "eot": "<|eot_id|>",
}
ALL_TOKENS = tuple(TEMPLATE)
RESERVED_BASE_V2 = 210


def placeholder(key):
    """Placeholder token for one template token; fixed per key so checkpoints and data always agree."""
    return f"<|reserved_special_token_{RESERVED_BASE_V2 + ALL_TOKENS.index(key)}|>"


def parse_tokens(spec):
    keys = tuple(k for k in ALL_TOKENS if k in spec.split(",")) if spec != "all" else ALL_TOKENS
    unknown = set(spec.split(",")) - set(ALL_TOKENS) - {"all"}
    assert not unknown, f"unknown template tokens {unknown}; choose from {ALL_TOKENS}"
    assert keys, "choose at least one template token"
    return keys


def tag(keys):
    return "all" if keys == ALL_TOKENS else "-".join(keys)


def template_parts(keys):
    """(input header, closing token) with the chosen template tokens replaced by placeholders.

    MIDDLE in make_data.py is "<|eot_id|>" + "<|start_header_id|>input<|end_header_id|>\\n\\n": its first
    <|eot_id|> ends the user turn and is not part of the data's template, so it is never replaced.
    """
    sub = {k: (placeholder(k) if k in keys else TEMPLATE[k]) for k in ALL_TOKENS}
    header = "<|eot_id|>" + sub["header"] + sub["role"] + sub["end_header"] + "\n\n"
    assert MIDDLE == "<|eot_id|>" + TEMPLATE["header"] + TEMPLATE["role"] + TEMPLATE["end_header"] + "\n\n"
    close = sub["eot"]
    return header, close


def to_v2(prompt, keys):
    """SecPO prompt -> the same prompt with the chosen template tokens around the data replaced."""
    before, data, after = split_input(prompt)
    head = before[: -len(MIDDLE)]
    header, close = template_parts(keys)
    assert after == END and END.startswith(TEMPLATE["eot"])
    rest = END[len(TEMPLATE["eot"]):]
    names = [placeholder(k) for k in keys]
    assert not any(n in prompt for n in names), "placeholder token already in prompt"
    return head + header + data + close + rest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", default="all", help=f"comma list from {ALL_TOKENS}, or all")
    parser.add_argument("--out", default="data/processed/secpo_v2_{tag}")
    args = parser.parse_args()
    keys = parse_tokens(args.tokens)
    path = args.out.format(tag=tag(keys))

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

    bad = {i for idx in errors.values() for i in idx}
    ds = ds.filter(lambda _, i: i not in bad, with_indices=True)

    def convert(ex):
        return {
            "attacked": to_v2(ex["prompt"], keys),
            "clean": clean_prompt(ex["prompt"], ex["rejected_input_whole"]),
            "chosen": ex["chosen"],
            "rejected": ex["rejected"],
        }

    out = ds.map(convert, remove_columns=ds.column_names)
    out.save_to_disk(path)
    with open(os.path.join(path, "v2_tokens.json"), "w") as f:
        json.dump({"tokens": list(keys), "placeholders": [placeholder(k) for k in keys],
                   "replaces": [TEMPLATE[k] for k in keys]}, f, indent=2)
    print(f"saved {len(out)} rows to {path} (trainable template tokens: {', '.join(keys)})")
    print("\n===== attacked (V2) =====")
    print(out[0]["attacked"])


if __name__ == "__main__":
    main()
