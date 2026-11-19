from __future__ import annotations

from .common import BASE_MODEL, Defense, _messages, chat_ids, load_hf


class SecPO(Defense):
    """Released SecPO Llama-3.1-8B checkpoint.

    Uses the extra `input` role for untrusted data: [user: instruction, input: data].
    The checkpoint's tokenizer/chat template must define the `input` role; we do NOT
    fall back to Llama's default template because that would silently change behavior.
    """

    name = "secpo"

    def __init__(self, model, tokenizer):
        super().__init__(model, tokenizer)
        probe = tokenizer.apply_chat_template(
            [{"role": "user", "content": "u"}, {"role": "input", "content": "INPUT_PROBE"}],
            tokenize=False,
            add_generation_prompt=True,
        )
        if "INPUT_PROBE" not in probe:
            raise RuntimeError("Chat template does not render the `input` role; wrong tokenizer?")

    def build_input_ids(self, system, instruction, data):
        msgs = _messages(system, {"role": "user", "content": instruction})
        if data:
            msgs.append({"role": "input", "content": data})
        return chat_ids(self.tokenizer, msgs)


def load_secpo(
    checkpoint: str,
    adapter: bool | None = None,
    four_bit: bool = False,
) -> SecPO:
    """checkpoint: Sizhe-Chen/Llama-3.1-8B-Instruct-SecPO (or a local copy).

    The Llama release is a merged model that ships its own tokenizer and `input`-role chat
    template, so the default (adapter=False) path is the right one. The adapter branch is only
    for your own LoRA checkpoints: their tokenizer must come from the SecPO release.
    """
    import json, os

    if adapter is None:
        adapter = os.path.exists(os.path.join(checkpoint, "adapter_config.json"))

    if adapter:
        from peft import PeftModel

        model, tok = load_hf(BASE_MODEL, tokenizer_id=checkpoint, four_bit=four_bit)
        model.resize_token_embeddings(len(tok))
        model = PeftModel.from_pretrained(model, checkpoint)
    else:
        model, tok = load_hf(checkpoint, four_bit=four_bit)
    model.eval()
    return SecPO(model, tok)
