"""
Train learnable delimiter embeddings on a frozen LLM with the SecPO loss.

data/make_data.py marks the data with unused Llama 3 reserved special tokens
(<|reserved_special_token_200+k|>), each a single token id.

SecPO loss (Sizhe-Chen/SecPO, src/secpo/loss.py):
  policy    log p(y | attacked prompt with delimiters)    ours, trainable delimiters
  reference log p(y | clean prompt without delimiters)    frozen base model
  loss = -logsigmoid(beta * [(pol_c - pol_r) - (ref_c - ref_r)])
Since the model is frozen and the clean prompt has no delimiters, the reference is exactly
the undefended model, so it is computed on the fly under no_grad.

Usage (Mac, debug):
  python train/train_delims.py --model unsloth/Llama-3.2-1B-Instruct --max_steps 20 --log_every 1
Usage (Windows, 8B, the settings of the reported run; several differ from the defaults below):
  python train/train_delims.py --model meta-llama/Llama-3.1-8B-Instruct --load_in_4bit --n 2 --lr 0.1 --grad_accum 16 --init std_normal --seed 42 --out "outputs/final_n2_lr{lr}_ga{ga}_{init}"
"""

import argparse
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn.functional as F
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
from make_data import make_delims  # noqa: E402

IGNORE = -100


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="unsloth/Llama-3.2-1B-Instruct")
    p.add_argument("--data", default="data/processed/secpo_n{n}")
    p.add_argument("--n", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--grad_accum", type=int, default=32)
    p.add_argument("--max_steps", type=int, default=-1, help="optimizer steps; -1 = one epoch")
    p.add_argument("--val_size", type=int, default=500)
    p.add_argument("--val_every", type=int, default=50)
    p.add_argument("--val_limit", type=int, default=100, help="val rows used per check (speed)")
    p.add_argument("--log_every", type=int, default=5)
    p.add_argument("--save_every", type=int, default=50)
    p.add_argument("--out", default="outputs/delims_n{n}_lr{lr}_ga{ga}_{init}")
    p.add_argument("--load_in_4bit", action="store_true")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_limit", type=int, default=-1, help="use only the first N train rows (sweeps)")
    p.add_argument("--init", default="header", choices=["header", "mean", "normal", "std_normal", "zero"])
    return p.parse_args()


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(args, device):
    dtype = getattr(torch, args.dtype)
    kwargs = {"dtype": dtype}
    if args.load_in_4bit: # for quantization
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype
        )
        kwargs["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    if not args.load_in_4bit:
        model.to(device)
    for prm in model.parameters():
        prm.requires_grad_(False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.train()  # gradient checkpointing only runs in train mode; Llama has no dropout
    return model


def delimiter_ids(tok, n):
    """Token ids of the delimiters from make_data.py. Each must be a single token."""
    opens, closes = make_delims(n)
    ids = []
    for name in opens + closes:
        enc = tok.encode(name, add_special_tokens=False)
        assert len(enc) == 1, f"{name} is not a single token: {enc}"
        ids.append(enc[0])
    return ids, len(opens)


def init_delims(model, tok, ids, n_open, how="header"):
    """
    header: open delimiters start from <|start_header_id|>, close delimiters from <|eot_id|>.
    mean:   mean of all vocabulary embeddings (Hewitt 2021; default of HF resize_token_embeddings).
    normal: N(0, initializer_range^2), how Llama initializes its own embeddings before pretraining.
    std_normal: N(0, 1), as in DefensiveToken and LTBD.
    zero:   all-zero vectors, an uninformative start.
    """
    emb = model.get_input_embeddings().weight
    shape = (len(ids), emb.shape[1])
    if how == "header":
        hdr = tok.convert_tokens_to_ids("<|start_header_id|>")
        eot = tok.convert_tokens_to_ids("<|eot_id|>")
        rows = torch.stack([emb[hdr] if i < n_open else emb[eot] for i in range(len(ids))]).float()
    elif how == "mean":
        rows = emb.float().mean(dim=0, keepdim=True).repeat(len(ids), 1)
    elif how == "normal":
        rows = torch.randn(shape, device=emb.device) * model.config.initializer_range
    elif how == "std_normal":
        rows = torch.randn(shape, device=emb.device)
    else:
        rows = torch.zeros(shape, device=emb.device)
    return torch.nn.Parameter(rows.detach().clone())


def encode(tok, prompt, response, max_length):
    """Prompt already contains <|begin_of_text|> and the template; response ends with <|eot_id|>."""
    p = tok.encode(prompt, add_special_tokens=False)
    r = tok.encode(response, add_special_tokens=False)
    if len(p) + len(r) > max_length:
        return None
    return p + r, [IGNORE] * len(p) + r


def build_examples(tok, ds, max_length):
    out, skipped = [], 0
    for ex in ds:
        parts = [
            encode(tok, ex["attacked"], ex["chosen"], max_length),
            encode(tok, ex["attacked"], ex["rejected"], max_length),
            encode(tok, ex["clean"], ex["chosen"], max_length),
            encode(tok, ex["clean"], ex["rejected"], max_length),
        ]
        if any(x is None for x in parts):  # SecPO also drops over-length rows
            skipped += 1
            continue
        out.append(parts)
    return out, skipped


def pad(seqs, pad_id, device):
    L = max(len(s[0]) for s in seqs)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    lab = torch.full((len(seqs), L), IGNORE, dtype=torch.long)
    att = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, (x, y) in enumerate(seqs):
        ids[i, : len(x)] = torch.tensor(x)
        lab[i, : len(y)] = torch.tensor(y)
        att[i, : len(x)] = 1
    return ids.to(device), lab.to(device), att.to(device)


def seq_logps(model, embeds, labels, attn):
    """Summed log-prob of response tokens; lm_head is applied only at scored positions."""
    hidden = model.model(inputs_embeds=embeds, attention_mask=attn).last_hidden_state
    h, y = hidden[:, :-1], labels[:, 1:]
    mask = y.ne(IGNORE)
    logits = model.lm_head(h[mask]).float()
    lp = torch.log_softmax(logits, dim=-1).gather(-1, y[mask].unsqueeze(-1)).squeeze(-1)
    out = torch.zeros(labels.size(0), device=lp.device, dtype=lp.dtype)
    out.index_add_(0, mask.nonzero()[:, 0], lp)
    return out


def embed_with_delims(model, ids, delim_ids, delims):
    """Replace the embeddings at delimiter ids with the trainable vectors; the embedding matrix stays frozen."""
    emb = model.get_input_embeddings()(ids)
    for k, tid in enumerate(delim_ids):
        pos = ids.eq(tid).unsqueeze(-1)
        emb = torch.where(pos, delims[k].to(emb.dtype), emb)
    return emb


def pair_logps(model, ex, pad_id, device, delim_ids=None, delims=None):
    """Returns (log p(chosen), log p(rejected)) for one prompt type."""
    ids, lab, att = pad(ex, pad_id, device)
    if delims is None:
        emb = model.get_input_embeddings()(ids)
    else:
        emb = embed_with_delims(model, ids, delim_ids, delims)
    lp = seq_logps(model, emb, lab, att)
    return lp[0], lp[1]


def secpo_loss(pc, pr, rc, rr, beta):
    margin = (pc - pr) - (rc - rr)
    return -F.logsigmoid(beta * margin), (pc - pr).detach()


@torch.no_grad()
def evaluate(model, val, pad_id, device, delim_ids, delims, beta):
    losses, margins, chosen, rejected, beats_ref = [], [], [], [], []
    for ex in val:
        pc, pr = pair_logps(model, ex[:2], pad_id, device, delim_ids, delims)
        rc, rr = pair_logps(model, ex[2:], pad_id, device)
        loss, m = secpo_loss(pc, pr, rc, rr, beta)
        losses.append(loss.item())
        margins.append(m.item())
        chosen.append(pc.item())
        rejected.append(pr.item())
        beats_ref.append(((pc - pr) >= (rc - rr)).item())  # SecPO's goal: policy margin >= reference margin
    mean = lambda xs: sum(xs) / len(xs)
    return {"val_loss": mean(losses), "val_margin": mean(margins), "val_margin_acc": mean([m > 0 for m in margins]),
            "val_beats_ref": mean(beats_ref),
            "val_chosen_logp": mean(chosen), "val_rejected_logp": mean(rejected)}


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device()
    out_dir = args.out.format(n=args.n, lr=args.lr, ga=args.grad_accum, init=args.init)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    assert args.n >= 2 and args.n % 2 == 0, "n must be an even number >= 2"

    tok = AutoTokenizer.from_pretrained(args.model)
    pad_id = tok.convert_tokens_to_ids("<|finetune_right_pad_id|>")
    model = load_model(args, device)
    delim_ids, n_open = delimiter_ids(tok, args.n)
    delims = init_delims(model, tok, delim_ids, n_open, args.init).to(device)
    vocab_norm = model.get_input_embeddings().weight.float().norm(dim=-1).mean().item()
    print(f"device={device} delimiter ids={delim_ids} trainable params={delims.numel()}")
    print(f"init={args.init} delimiter norms={delims.norm(dim=-1).tolist()} mean vocab norm={vocab_norm:.4f}")

    ds = load_from_disk(args.data.format(n=args.n)).shuffle(seed=args.seed)
    val_ds, train_ds = ds.select(range(args.val_size)), ds.select(range(args.val_size, len(ds)))
    if args.train_limit > 0:
        train_ds = train_ds.select(range(min(args.train_limit, len(train_ds))))
    train, sk_t = build_examples(tok, train_ds, args.max_length)
    val, sk_v = build_examples(tok, val_ds, args.max_length)
    val = val[: args.val_limit]
    print(f"train={len(train)} (skipped {sk_t} over-length) val={len(val)} (skipped {sk_v})")
    assert len(train) >= args.grad_accum, f"only {len(train)} train rows, fewer than grad_accum={args.grad_accum}"
    assert len(val) > 0, "validation set is empty"

    opt = torch.optim.AdamW([delims], lr=args.lr, weight_decay=0.0)
    total = len(train) // args.grad_accum
    if args.max_steps > 0:
        total = min(total, args.max_steps)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    print(f"optimizer steps={total} (effective batch {args.grad_accum})")

    log = open(os.path.join(out_dir, "log.jsonl"), "w")

    def norms():
        return [round(x, 4) for x in delims.detach().norm(dim=-1).tolist()]

    # step 0: validation before any update, the starting point of every curve
    rec = {"step": 0, "delim_norms": norms(), "vocab_norm": round(vocab_norm, 4)}
    rec.update(evaluate(model, val, pad_id, device, delim_ids, delims, args.beta))
    print(f"[val] step 0: loss {rec['val_loss']:.4f} margin {rec['val_margin']:.3f} acc {rec['val_margin_acc']:.3f} beats_ref {rec['val_beats_ref']:.3f}")
    log.write(json.dumps(rec) + "\n")
    log.flush()

    step, run_loss, run_margin, run_c, run_r, t0 = 0, 0.0, 0.0, 0.0, 0.0, time.time()
    for i, ex in enumerate(train):
        pc, pr = pair_logps(model, ex[:2], pad_id, device, delim_ids, delims)
        with torch.no_grad():
            rc, rr = pair_logps(model, ex[2:], pad_id, device)
        loss, margin = secpo_loss(pc, pr, rc, rr, args.beta)
        (loss / args.grad_accum).backward()
        run_loss += loss.item() / args.grad_accum
        run_margin += margin.item() / args.grad_accum
        run_c += pc.item() / args.grad_accum
        run_r += pr.item() / args.grad_accum

        if (i + 1) % args.grad_accum:
            continue
        cur_lr = sched.get_last_lr()[0]  # lr used by this step
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1

        rec = {"step": step, "loss": run_loss, "policy_margin": run_margin, "chosen_logp": run_c,
               "rejected_logp": run_r, "lr": cur_lr, "delim_norms": norms(), "vocab_norm": round(vocab_norm, 4),
               "sec_per_step": (time.time() - t0) / step}
        if step % args.log_every == 0 or step == 1:
            print(json.dumps(rec))
        if step % args.val_every == 0 or step == total:
            rec.update(evaluate(model, val, pad_id, device, delim_ids, delims, args.beta))
            print(f"[val] step {step}: loss {rec['val_loss']:.4f} margin {rec['val_margin']:.3f} acc {rec['val_margin_acc']:.3f} beats_ref {rec['val_beats_ref']:.3f}")
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if step % args.save_every == 0 or step == total:
            torch.save({"delims": delims.detach().cpu(), "delim_ids": delim_ids, "n": args.n, "step": step},
                       os.path.join(out_dir, f"delims_step{step}.pt"))
        run_loss, run_margin, run_c, run_r = 0.0, 0.0, 0.0, 0.0
        if step >= total:
            break
    print(f"done. saved to {out_dir}")


if __name__ == "__main__":
    main()