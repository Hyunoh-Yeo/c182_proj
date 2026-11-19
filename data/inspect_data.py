from datasets import load_dataset

ds = load_dataset("Sizhe-Chen/Llama-3.1-8B-Instruct-SecPO-trainset", split="train")
print(ds)
print("columns:", ds.column_names)

ex = ds[0]
for k, v in ex.items():
    print(f"\n===== {k} ({type(v).__name__}) =====")
    print(repr(v)[:1500])

"""
SecPO trainset format (Sizhe-Chen/Llama-3.1-8B-Instruct-SecPO-trainset, 19,157 rows)

== prompt (attacked input; line breaks after <|eot_id|> are for readability only) ==
<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|>
<|start_header_id|>input<|end_header_id|>\n\n{data_with_injection}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>\n\n

data_with_injection:
    inject_at_the_end == False -> {injection}{sep}{data}
    inject_at_the_end == True  -> {data}{sep}{injection}

    {injection} = rejected_input_whole column
    {data}      = chosen_data column
    {sep}       = separator between them (e.g. " ")

== columns ==
prompt: str               = Meta-SecAlign templated attacked prompt (no default Llama system block).
chosen_data: str          = Clean untrusted data without the injection.
rejected_input_whole: str = Full injected text inserted into the data.
inject_at_the_end: bool   = True if the injection comes after the data, False otherwise.
chosen: str               = Response that follows the trusted instruction.
rejected: str             = Response that follows the injection.
chosen_input: str         = Clean input, "{instruction}\n\n{chosen_data}" (not used in training).
rejected_input: str       = Injected attacking message (not used in training).

# Note: <|eot_id|> is already included in the responses.

Training uses:
attacked prompt  = rejected_input_whole
clean prompt     = chosen_data
SecPO loss       = increase log p(chosen | attacked) and decrease log p(rejected | attacked),
                   with reference log-probs from the undefended model on the clean prompt
delimiters       = inserted right after the input header and right before its <|eot_id|>,
                   in both the attacked and clean prompts
"""