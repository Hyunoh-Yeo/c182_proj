"""V2 of our model: frozen Llama with the chat-template tokens around the data made trainable
(train/train_delims_v2.py). V1 is baselines/delims.py.

Prompt = SecPO's training template with the chosen template tokens around the untrusted data swapped for
placeholder tokens (data/make_data_v2.py); with all four:

    <|begin_of_text|><|start_header_id|>user<|end_header_id|>\\n\\n{instruction}<|eot_id|>
    <t_header><t_role><t_end_header>\\n\\n{data}<t_eot>
    <|start_header_id|>assistant<|end_header_id|>\\n\\n

The placeholders are unused reserved special tokens, so the trained vectors are written into their rows of
the input embedding matrix, as in V1. Every other occurrence of the same template tokens keeps its original
embedding. With use_delims=False the placeholders are not used, which gives SecPO's template unchanged,
the same prompt as V1's no-delimiter ablation.

Untrusted data is filtered as in V1: the 8 Llama-3 control tokens plus our placeholder tokens are stripped
recursively (filter_data=True). With filter_data=False, an attacker who writes <|eot_id|> into the data gets
its original, untrained embedding; delim_escape writes our trained closing placeholder instead.
"""
from __future__ import annotations

import os
import sys

import torch

from .common import Defense, load_hf, recursive_filter, LLAMA3_CONTROL_TOKENS
from .delims import SYSTEM_BLOCK

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
from make_data import END, MIDDLE, PREFIX  # noqa: E402
from make_data_v2 import ALL_TOKENS, TEMPLATE, placeholder, template_parts  # noqa: E402


class DelimsV2(Defense):
    name = "delims_v2"

    def __init__(self, model, tokenizer, tokens, use_delims: bool = True, filter_data: bool = True):
        super().__init__(model, tokenizer)
        self.tokens = tuple(tokens)
        self.use_delims = use_delims
        self.filter_data = filter_data
        names = tuple(placeholder(k) for k in self.tokens)
        self.filters = LLAMA3_CONTROL_TOKENS + names  # same filtering with and without the placeholders
        if use_delims:
            self.header, close = template_parts(self.tokens)
            self.end = close + END[len(TEMPLATE["eot"]):]
        else:
            self.header, self.end = MIDDLE, END
        # run_attacks.py adds delim_escape when the defense has a trained closing token
        self.close = placeholder("eot") if (use_delims and "eot" in self.tokens) else ""

    def build_input_ids(self, system, instruction, data):
        head = PREFIX
        if system:
            head = PREFIX.replace("<|begin_of_text|>", "<|begin_of_text|>" + SYSTEM_BLOCK.format(system=system), 1)
        if data:
            if self.filter_data:
                data = recursive_filter(data, self.filters)
            text = head + instruction + self.header + data + self.end
        else:
            text = head + instruction + END
        return self.tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids


def load_delims_v2(
    checkpoint: str,
    model_id: str | None = None,
    four_bit: bool = False,
    use_delims: bool = True,
    filter_data: bool = True,
) -> DelimsV2:
    """checkpoint: a delims_v2_step{N}.pt saved by train/train_delims_v2.py."""
    import json

    ckpt = torch.load(checkpoint, map_location="cpu")
    assert ckpt.get("version") == 2, f"{checkpoint} is not a V2 checkpoint; use load_delims for V1"
    if model_id is None:
        cfg = os.path.join(os.path.dirname(checkpoint), "config.json")
        model_id = json.load(open(cfg))["model"]
    model, tok = load_hf(model_id, four_bit=four_bit)
    tokens = tuple(ckpt["tokens"])
    assert set(tokens) <= set(ALL_TOKENS)
    ids = [tok.convert_tokens_to_ids(placeholder(k)) for k in tokens]
    if ids != list(ckpt["delim_ids"]):
        raise RuntimeError(f"placeholder ids {ids} != checkpoint ids {ckpt['delim_ids']}; wrong tokenizer?")

    emb = model.get_input_embeddings()
    out = model.get_output_embeddings()
    if out is not None and out.weight is emb.weight:
        out.weight = torch.nn.Parameter(emb.weight.detach().clone(), requires_grad=False)
        model.config.tie_word_embeddings = False
    if use_delims:
        with torch.no_grad():
            for vec, tid in zip(ckpt["delims"], ids):
                emb.weight[tid] = vec.to(emb.weight.device, emb.weight.dtype)
    model.eval()
    return DelimsV2(model, tok, tokens, use_delims=use_delims, filter_data=filter_data)
