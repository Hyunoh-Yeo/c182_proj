"""
Train the V2 variant (template tokens around the data) on a frozen LLM with the SecPO loss.

V1 (train/train_delims.py) trains n inserted delimiter vectors. V2 inserts nothing: data/make_data_v2.py
swaps the chat-template tokens around the untrusted data for placeholder tokens, and this script trains
one vector per placeholder. With --init template (the default), each vector starts from the embedding of
the template token it replaces, so at step 0 the defended model is exactly the undefended model on
SecPO's prompt (checked below before training starts). Only these positions change: the same template
tokens elsewhere in the prompt keep their frozen embeddings.

Loss, data split, optimizer, schedule and logging are the same as V1 and come from train/train_delims.py.

Usage (Windows, 8B, the V1 settings except the init):
  python data/make_data_v2.py
  python train/train_delims_v2.py --model meta-llama/Llama-3.1-8B-Instruct --load_in_4bit --lr 0.1 --grad_accum 16 --seed 42
Options:
  --tokens header,eot          train only some of the four template tokens (must match make_data_v2.py)
  --init std_normal            start from N(0, I) as V1 does, instead of the original embeddings
"""

import argparse
import json
import math
import os
import random
import sys
import time

import torch
from datasets import load_from_disk
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
from make_data_v2 import ALL_TOKENS, TEMPLATE, parse_tokens, placeholder, tag  # noqa: E402
from train_delims import (build_examples, embed_with_delims, evaluate, load_model, pair_logps,  # noqa: E402
                          pick_device, secpo_loss)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="unsloth/Llama-3.2-1B-Instruct")
    p.add_argument("--tokens", default="all", help=f"comma list from {ALL_TOKENS}, or all")
    p.add_argument("--data", default="data/processed/secpo_v2_{tag}")
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--grad_accum", type=int, default=16)
    p.add_argument("--max_steps", type=int, default=-1, help="optimizer steps; -1 = one epoch")
    p.add_argument("--val_size", type=int, default=500)
    p.add_argument("--val_every", type=int, default=50)
    p.add_argument("--val_limit", type=int, default=100, help="val rows used per check (speed)")
    p.add_argument("--log_every", type=int, default=5)
    p.add_argument("--save_every", type=int, default=50)
    p.add_argument("--out", default="outputs/v2_{tag}_lr{lr}_ga{ga}_{init}")
    p.add_argument("--load_in_4bit", action="store_true")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_limit", type=int, default=-1, help="use only the first N train rows (sweeps)")
    p.add_argument("--init", default="template", choices=["template", "std_normal"])
    return p.parse_args()


def single_id(tok, text):
    enc = tok.encode(text, add_special_tokens=False)
    assert len(enc) == 1, f"{text!r} is not a single token: {enc}"
    return enc[0]


def placeholder_ids(tok, keys):
    """(placeholder ids, ids of the template tokens they replace), in the order of keys."""
    return [single_id(tok, placeholder(k)) for k in keys], [single_id(tok, TEMPLATE[k]) for k in keys]


def init_vectors(model, replaced_ids, how):
    emb = model.get_input_embeddings().weight
    if how == "template":
        rows = emb[replaced_ids].float()
    else:
        rows = torch.randn((len(replaced_ids), emb.shape[1]), device=emb.device)
    return torch.nn.Parameter(rows.detach().clone())


@torch.no_grad()
def check_alignment(model, tok, row, ph_ids, replaced_ids, vecs, max_length):
    """The V2 prompt must line up token for token with the SecPO prompt, and with --init template its
    embeddings must equal the SecPO prompt's, so training starts exactly at the undefended model."""
    original = row["attacked"]
    for pid, rid in zip(ph_ids, replaced_ids):
        original = original.replace(tok.convert_ids_to_tokens(pid), tok.decode([rid]))
    ours = tok.encode(row["attacked"], add_special_tokens=False)
    base = tok.encode(original, add_special_tokens=False)
    assert len(ours) == len(base), f"V2 prompt has {len(ours)} tokens, SecPO prompt {len(base)}"
    diff = [i for i, (a, b) in enumerate(zip(ours, base)) if a != b]
    assert all(ours[i] in ph_ids for i in diff), "V2 prompt differs from SecPO's outside the placeholders"
    assert len(diff) == len(ph_ids), f"expected {len(ph_ids)} replaced positions, found {len(diff)}"
    dev = model.get_input_embeddings().weight.device
    a = embed_with_delims(model, torch.tensor([ours], device=dev), ph_ids, vecs)
    b = model.get_input_embeddings()(torch.tensor([base], device=dev))
    return (a.float() - b.float()).abs().max().item()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device()
    keys = parse_tokens(args.tokens)
    out_dir = args.out.format(tag=tag(keys), lr=args.lr, ga=args.grad_accum, init=args.init)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump({**vars(args), "version": 2, "tokens_resolved": list(keys)}, f, indent=2)

    tok = AutoTokenizer.from_pretrained(args.model)
    pad_id = tok.convert_tokens_to_ids("<|finetune_right_pad_id|>")
    model = load_model(args, device)
    ph_ids, replaced_ids = placeholder_ids(tok, keys)
    vecs = init_vectors(model, replaced_ids, args.init).to(device)
    vocab_norm = model.get_input_embeddings().weight.float().norm(dim=-1).mean().item()
    print(f"device={device} template tokens={list(keys)} placeholder ids={ph_ids} replace ids={replaced_ids}")
    print(f"trainable params={vecs.numel()} init={args.init} norms={vecs.norm(dim=-1).tolist()} "
          f"mean vocab norm={vocab_norm:.4f}")

    ds = load_from_disk(args.data.format(tag=tag(keys))).shuffle(seed=args.seed)
    gap = check_alignment(model, tok, ds[0], ph_ids, replaced_ids, vecs, args.max_length)
    print(f"alignment ok; max |embedding difference| from SecPO's prompt at step 0: {gap:.2e}")
    if args.init == "template":
        assert gap < 1e-2, "template init should reproduce SecPO's prompt embeddings"

    val_ds, train_ds = ds.select(range(args.val_size)), ds.select(range(args.val_size, len(ds)))
    if args.train_limit > 0:
        train_ds = train_ds.select(range(min(args.train_limit, len(train_ds))))
    train, sk_t = build_examples(tok, train_ds, args.max_length)
    val, sk_v = build_examples(tok, val_ds, args.max_length)
    val = val[: args.val_limit]
    print(f"train={len(train)} (skipped {sk_t} over-length) val={len(val)} (skipped {sk_v})")
    assert len(train) >= args.grad_accum and len(val) > 0

    opt = torch.optim.AdamW([vecs], lr=args.lr, weight_decay=0.0)
    total = len(train) // args.grad_accum
    if args.max_steps > 0:
        total = min(total, args.max_steps)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    print(f"optimizer steps={total} (effective batch {args.grad_accum})")

    log = open(os.path.join(out_dir, "log.jsonl"), "w")
    start = vecs.detach().clone()

    def norms():
        return [round(x, 4) for x in vecs.detach().norm(dim=-1).tolist()]

    def moved():
        return [round(x, 4) for x in (vecs.detach() - start).norm(dim=-1).tolist()]

    def save(step):
        torch.save({"delims": vecs.detach().cpu(), "delim_ids": ph_ids, "replaced_ids": replaced_ids,
                    "tokens": list(keys), "version": 2, "step": step},
                   os.path.join(out_dir, f"delims_v2_step{step}.pt"))

    rec = {"step": 0, "delim_norms": norms(), "moved": moved(), "vocab_norm": round(vocab_norm, 4)}
    rec.update(evaluate(model, val, pad_id, device, ph_ids, vecs, args.beta))
    print(f"[val] step 0: loss {rec['val_loss']:.4f} beats_ref {rec['val_beats_ref']:.3f}")
    log.write(json.dumps(rec) + "\n")
    log.flush()

    step, run_loss, run_margin, run_c, run_r, t0 = 0, 0.0, 0.0, 0.0, 0.0, time.time()
    for i, ex in enumerate(train):
        pc, pr = pair_logps(model, ex[:2], pad_id, device, ph_ids, vecs)
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
        cur_lr = sched.get_last_lr()[0]
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        rec = {"step": step, "loss": run_loss, "policy_margin": run_margin, "chosen_logp": run_c,
               "rejected_logp": run_r, "lr": cur_lr, "delim_norms": norms(), "moved": moved(),
               "vocab_norm": round(vocab_norm, 4), "sec_per_step": (time.time() - t0) / step}
        if step % args.log_every == 0 or step == 1:
            print(json.dumps(rec))
        if step % args.val_every == 0 or step == total:
            rec.update(evaluate(model, val, pad_id, device, ph_ids, vecs, args.beta))
            print(f"[val] step {step}: loss {rec['val_loss']:.4f} beats_ref {rec['val_beats_ref']:.3f}")
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if step % args.save_every == 0 or step == total:
            save(step)
        run_loss, run_margin, run_c, run_r = 0.0, 0.0, 0.0, 0.0
        if step >= total:
            break
    print(f"done. saved to {out_dir}")


if __name__ == "__main__":
    main()
