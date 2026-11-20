"""
Sanity checks for train/train_delims.py on a few real examples.
Usage: python train/check_train.py --model meta-llama/Llama-3.1-8B-Instruct --load_in_4bit
"""
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer

from train_delims import (IGNORE, build_examples, delimiter_ids, embed_with_delims, init_delims,
                          load_model, pad, pair_logps, parse_args, pick_device, secpo_loss, seq_logps)

args = parse_args()
device = pick_device()
tok = AutoTokenizer.from_pretrained(args.model)
pad_id = tok.convert_tokens_to_ids("<|eot_id|>")
model = load_model(args, device)
delim_ids, n_open = delimiter_ids(tok, args.n)
examples, _ = build_examples(tok, load_from_disk(args.data.format(n=args.n)).select(range(20)), args.max_length)
ex = min(examples, key=lambda e: len(e[0][0]))  # shortest example, to keep the full-logits check small

# 1. each attacked sequence has every delimiter exactly once; clean sequences have none
for seq, _ in ex[:2]:
    assert all(seq.count(t) == 1 for t in delim_ids)
for seq, _ in ex[2:]:
    assert not any(t in seq for t in delim_ids)
print("1 ok: delimiters appear once in attacked, never in clean")

# 2. our summed log-prob equals a plain full-logits computation
with torch.no_grad():
    ids, lab, att = pad(ex[2:], pad_id, device)
    ours = seq_logps(model, model.get_input_embeddings()(ids), lab, att)
    logp = torch.log_softmax(model(input_ids=ids, attention_mask=att).logits.float(), dim=-1)
    y = lab[:, 1:]
    tok_lp = logp[:, :-1].gather(-1, y.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    full = (tok_lp * y.ne(IGNORE)).sum(-1)
print(f"2 ours {ours.tolist()} full {full.tolist()}")
assert torch.allclose(ours, full, atol=0.5)
print("2 ok: seq_logps matches the full computation")

# 3. putting the original embedding rows back in changes nothing
emb = model.get_input_embeddings().weight
same = emb[delim_ids].float()
ids, _, _ = pad(ex[:2], pad_id, device)
assert torch.equal(embed_with_delims(model, ids, delim_ids, same), model.get_input_embeddings()(ids))
print("3 ok: the swap only touches delimiter positions")

# 4. gradient reaches the delimiters and nothing else
delims = init_delims(model, tok, delim_ids, n_open, "header").to(device)
pc, pr = pair_logps(model, ex[:2], pad_id, device, delim_ids, delims)
with torch.no_grad():
    rc, rr = pair_logps(model, ex[2:], pad_id, device)
loss, _ = secpo_loss(pc, pr, rc, rr, args.beta)
loss.backward()
assert delims.grad is not None and delims.grad.abs().sum() > 0
assert all(p.grad is None for p in model.parameters())
print(f"4 ok: delimiter grad norms {delims.grad.norm(dim=-1).tolist()}, model has no grads")
print(f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GB" if device == "cuda" else "")