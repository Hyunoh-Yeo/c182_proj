from __future__ import annotations

from .common import BASE_MODEL, Defense, _messages, chat_ids, load_hf


class Undefended(Defense):
    """Llama-3.1-8B-Instruct, original template. Untrusted data is appended to the user turn.

    ASSUMPTION: this matches how the undefended reference was queried in SecPO training
    (instruction + data in one user message). The SecPO docs do not state this for Llama
    (they only describe Qwen, where data goes in a `tool` role), so it stays an assumption.
    """

    name = "undefended"

    def build_input_ids(self, system, instruction, data):
        content = f"{instruction}\n\n{data}" if data else instruction
        return chat_ids(self.tokenizer, _messages(system, {"role": "user", "content": content}))


def load_undefended(model_id: str = BASE_MODEL, four_bit: bool = False) -> Undefended:
    model, tok = load_hf(model_id, four_bit=four_bit)
    return Undefended(model, tok)
