"""DefensiveToken baseline (Chen et al. 2025, github.com/Sizhe-Chen/DefensiveToken).

What it is: Llama-3.1-8B-Instruct with its weights UNCHANGED, plus 5 extra "tokens" whose
embedding vectors were optimized with a SecAlign-style loss. At test time the 5 tokens are
written in front of the prompt. Leaving them out gives back the original model.

This file mirrors the repo's setup.py + demo.py exactly:
  * 5 tokens named [DefensiveToken0] .. [DefensiveToken4], vectors from defensivetokens.json
    (a dict: base model id -> list of 5 vectors).
  * Placement: the tokens come BEFORE <|begin_of_text|>, not after it.
  * Roles: trusted instruction -> `system` turn, untrusted data -> `user` turn.
  * A custom chat template (below): one "\\n" after the header, "\\n\\n" after the content.
  * Untrusted data is stripped of every special token (recursively) before templating.

Get the vectors:
    curl -LO https://raw.githubusercontent.com/Sizhe-Chen/DefensiveToken/main/defensivetokens.json
"""
from __future__ import annotations

import json
import os

import torch

from .common import BASE_MODEL, Defense, load_hf, recursive_filter

N_TOKENS = 5

# Copied verbatim from the repo's setup.py (Llama-3.1-8B-Instruct entry). Do not "clean up" the
# whitespace or the \n escapes: this exact string is what the vectors were trained with.
LLAMA31_DT_TEMPLATE = """{%- if add_defensive_tokens %}\n{{- '[DefensiveToken0][DefensiveToken1][DefensiveToken2][DefensiveToken3][DefensiveToken4]' }}\n{%- endif %}\n
    {{- bos_token }}\n
    {%- for message in messages %}\n
        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n'+ message['content'] | trim + '\\n\\n' + '<|eot_id|>' }}\n
    {%- endfor %}\n
    {%- if add_generation_prompt %}\n{{- '<|start_header_id|>assistant<|end_header_id|>\\n' }}\n{%- endif %}\n"""


class DefensiveToken(Defense):
    name = "defensivetoken"

    def __init__(self, model, tokenizer, use_tokens: bool = True):
        super().__init__(model, tokenizer)
        self.use_tokens = use_tokens  # False = same model/template without the tokens (ablation)

    def build_input_ids(self, system, instruction, data):
        # Repo convention: trusted instruction in `system`, untrusted data in `user`.
        # The benchmark's own `system` prompt (if any) is merged into the trusted turn.
        trusted = f"{system}\n\n{instruction}" if system else instruction
        if data:
            # As in the released demo.py. For Llama-3.1 all_special_tokens is only BOS, EOT and the
            # 5 DefensiveTokens, so header tokens get through; we keep this to match the release.
            data = recursive_filter(data, self.tokenizer.all_special_tokens)
            msgs = [{"role": "system", "content": trusted}, {"role": "user", "content": data}]
        else:
            msgs = [{"role": "user", "content": trusted}]
        text = self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            add_defensive_tokens=self.use_tokens,
        )
        # The template already writes <bos>; do not let the tokenizer add a second one.
        return self.tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids


def load_defensivetoken(
    embeddings_path: str,
    model_id: str = BASE_MODEL,
    four_bit: bool = False,
) -> DefensiveToken:
    """embeddings_path is one of:
      * defensivetokens.json from the repo (we build the modified model in memory), or
      * a directory produced by the repo's `python setup.py`, e.g.
        meta-llama/Llama-3.1-8B-Instruct-5DefensiveTokens (already has tokens + template).
    """
    if os.path.isdir(embeddings_path):
        model, tok = load_hf(embeddings_path, four_bit=four_bit)
        return DefensiveToken(model, tok)

    with open(embeddings_path) as f:
        table = json.load(f)
    if model_id not in table:
        raise KeyError(f"{model_id} not in {embeddings_path}; it has vectors for: {sorted(table)}")
    vecs = torch.tensor(table[model_id])  # [5, d]
    if vecs.shape[0] != N_TOKENS:
        raise ValueError(f"expected {N_TOKENS} vectors, got {vecs.shape[0]}")

    model, tok = load_hf(model_id, four_bit=four_bit)
    names = [f"[DefensiveToken{i}]" for i in range(N_TOKENS)]
    tok.add_special_tokens({"additional_special_tokens": names})
    model.resize_token_embeddings(len(tok))
    w = model.get_input_embeddings().weight
    ids = tok.convert_tokens_to_ids(names)
    if ids != list(range(len(tok) - N_TOKENS, len(tok))):  # setup.py writes to the last 5 rows
        raise RuntimeError(f"new token ids {ids} are not the last {N_TOKENS} rows")
    with torch.no_grad():
        for i, tid in enumerate(ids):
            w[tid] = vecs[i].to(w.device, w.dtype)
    tok.chat_template = LLAMA31_DT_TEMPLATE
    model.eval()
    return DefensiveToken(model, tok)
