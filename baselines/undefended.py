from __future__ import annotations

from .common import BASE_MODEL, LLAMA3_CONTROL_TOKENS, Defense, _messages, chat_ids, load_hf, recursive_filter


class Undefended(Defense):
    """Llama-3.1-8B-Instruct, original template. Untrusted data is appended to the user turn.

    ASSUMPTION: this matches how the undefended reference was queried in SecPO training
    (instruction + data in one user message). The SecPO docs do not state this for Llama
    (they only describe Qwen, where data goes in a `tool` role), so it stays an assumption.

    Untrusted data is filtered before templating (filter_data=True, the default): the 8 Llama-3
    control tokens SecPO's template removes are stripped recursively, so this baseline sees the same
    data as SecPO and ours. filter_data=False is the unfiltered setting of the earlier results.
    """

    name = "undefended"

    def __init__(self, model, tokenizer, filter_data: bool = True):
        super().__init__(model, tokenizer)
        self.filter_data = filter_data

    def build_input_ids(self, system, instruction, data):
        if data and self.filter_data:
            data = recursive_filter(data, LLAMA3_CONTROL_TOKENS)
        content = f"{instruction}\n\n{data}" if data else instruction
        return chat_ids(self.tokenizer, _messages(system, {"role": "user", "content": content}))


def load_undefended(model_id: str = BASE_MODEL, four_bit: bool = False, filter_data: bool = True) -> Undefended:
    model, tok = load_hf(model_id, four_bit=four_bit)
    return Undefended(model, tok, filter_data=filter_data)