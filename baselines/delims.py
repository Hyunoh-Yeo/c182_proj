"""Our model: frozen Llama + learnable delimiter embeddings (train/train_delims.py).

Prompt = the SecPO training template from data/make_data.py, with the delimiter tokens
wrapped around the untrusted data:

    <|begin_of_text|><|start_header_id|>user<|end_header_id|>\\n\\n{instruction}<|eot_id|>
    <|start_header_id|>input<|end_header_id|>\\n\\n{open delims}{data}{close delims}<|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>\\n\\n

The delimiters are unused reserved special tokens, so the trained vectors are written into
their rows of the input embedding matrix. That is the same as embed_with_delims() in training
(which swaps the embedding at those ids) and lets us use the normal Defense.generate().
"""
from __future__ import annotations

import json
import os
import sys

import torch

from .common import BASE_MODEL, Defense, load_hf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
from make_data import END, MIDDLE, PREFIX, make_delims  # noqa: E402

SYSTEM_BLOCK = "<|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>"


class Delims(Defense):
    name = "delims"

    def __init__(self, model, tokenizer, n: int, use_delims: bool = True):
        super().__init__(model, tokenizer)
        self.n = n
        self.use_delims = use_delims  # False = same template without delimiters (ablation)
        opens, closes = make_delims(n)
        self.open, self.close = "".join(opens), "".join(closes)

    def build_input_ids(self, system, instruction, data):
        head = PREFIX
        if system:  # training data has no system block; this is only for benchmarks that pass one
            head = PREFIX.replace("<|begin_of_text|>", "<|begin_of_text|>" + SYSTEM_BLOCK.format(system=system), 1)
        if data:
            if self.use_delims:
                data = self.open + data + self.close
            text = head + instruction + MIDDLE + data + END
        else:
            text = head + instruction + END
        return self.tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids


def load_delims(
    checkpoint: str,
    model_id: str | None = None,
    four_bit: bool = False,
    use_delims: bool = True,
) -> Delims:
    """checkpoint: a delims_step{N}.pt saved by train/train_delims.py.

    The base model is read from config.json next to the checkpoint unless model_id is given.
    """
    ckpt = torch.load(checkpoint, map_location="cpu")
    if model_id is None:
        cfg = os.path.join(os.path.dirname(checkpoint), "config.json")
        model_id = json.load(open(cfg))["model"] if os.path.exists(cfg) else BASE_MODEL

    model, tok = load_hf(model_id, four_bit=four_bit)
    opens, closes = make_delims(ckpt["n"])
    ids = [tok.convert_tokens_to_ids(name) for name in opens + closes]
    if ids != list(ckpt["delim_ids"]):
        raise RuntimeError(f"delimiter ids {ids} != checkpoint ids {ckpt['delim_ids']}; wrong tokenizer?")

    emb = model.get_input_embeddings()
    out = model.get_output_embeddings()
    if out is not None and out.weight is emb.weight:
        # Tied embeddings (e.g. Llama-3.2-1B): untie so the lm_head stays frozen, as in training.
        out.weight = torch.nn.Parameter(emb.weight.detach().clone(), requires_grad=False)
        model.config.tie_word_embeddings = False
    if use_delims:
        with torch.no_grad():
            for vec, tid in zip(ckpt["delims"], ids):
                emb.weight[tid] = vec.to(emb.weight.device, emb.weight.dtype)
    model.eval()
    return Delims(model, tok, n=ckpt["n"], use_delims=use_delims)
