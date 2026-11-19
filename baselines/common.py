"""Shared interface for all defenses (undefended, SecPO, DefensiveToken, and our delimiter model).

Everyone else (eval harness, attack script) should only call:

    model = load("secpo")            # or "undefended", "defensivetoken"
    text = model.generate(system, instruction, data)
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"


def pick_device() -> str:
    """cuda (RTX laptop) > mps (Apple Silicon Mac) > cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_hf(model_id: str, tokenizer_id: str | None = None, four_bit: bool = False):
    """Load model + tokenizer. Use the SAME precision for every baseline you compare."""
    device = pick_device()
    tok = AutoTokenizer.from_pretrained(tokenizer_id or model_id)
    if four_bit:
        if device != "cuda":
            raise RuntimeError(
                f"--four-bit uses bitsandbytes and needs an NVIDIA GPU; this machine has {device}. "
                "Drop --four-bit (bf16) here, and keep precision identical across compared baselines."
            )
        kwargs = dict(
            device_map="auto",
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            ),
        )
    else:
        kwargs = dict(device_map={"": device}, dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    return model, tok


class Defense(ABC):
    """One defense = one way of turning (system, instruction, untrusted data) into model input."""

    name: str = "base"

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @abstractmethod
    def build_input_ids(self, system: str | None, instruction: str, data: str) -> torch.Tensor:
        """Return input_ids of shape [1, T] ending with the assistant generation prompt."""

    @torch.no_grad()
    def generate(
        self,
        system: str | None,
        instruction: str,
        data: str,
        max_new_tokens: int = 512,
        do_sample: bool = False,
        temperature: float = 1.0,
    ) -> str:
        ids = self.build_input_ids(system, instruction, data).to(self.model.device)
        out = self.model.generate(
            ids,
            attention_mask=torch.ones_like(ids),
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=None,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def chat_ids(tokenizer, msgs: list[dict]) -> torch.Tensor:
    """Render with the chat template, then tokenize. Returns a [1, T] id tensor.

    Don't use apply_chat_template(return_tensors="pt") directly: newer transformers return a
    BatchEncoding instead of a tensor. add_special_tokens=False because the template already
    writes <|begin_of_text|>.
    """
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids


def _messages(system: str | None, *rest: dict) -> list[dict]:
    msgs = [{"role": "system", "content": system}] if system else []
    return msgs + list(rest)
