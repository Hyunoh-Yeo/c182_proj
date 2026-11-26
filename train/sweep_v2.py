"""
V2 learning-rate check (train/train_delims_v2.py), the runs summarized in results/sweep_v2/README.md.
V1's sweep is train/sweep.py.

V1's lr 0.1 was tuned for vectors that start from N(0, I) with norm 64. V2 starts from the original
template-token embeddings (norms 0.03 to 0.6), so each run here checks a smaller lr on a short budget.
The cosine schedule decays to 0 within each short run.

  smoke   3 steps at lr 0.1 (V1's value), grad_accum 4, 32 rows, 4 validation pairs
  lr      30 steps each at lr 1e-3 and 1e-4, grad_accum 16, 480 rows, 100 validation pairs
  copy    copy config.json and log.jsonl of these runs from outputs/ (git-ignored) to results/sweep_v2/

Runs write to outputs/; train/train_delims_v2.py needs data/processed/secpo_v2_all
(python data/make_data_v2.py). Works on Windows and Mac.
Usage: python train/sweep_v2.py smoke | lr | copy
"""
import os
import shutil
import subprocess
import sys

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
COMMON = ["--model", MODEL, "--load_in_4bit", "--seed", "42"]

RUNS = {
    # name in results/sweep_v2: (output folder under outputs/, extra arguments)
    "lr0.1_ga4_3steps": ("smoke_v2_all", ["--lr", "0.1", "--grad_accum", "4", "--max_steps", "3",
                                          "--train_limit", "32", "--val_size", "8", "--val_limit", "4",
                                          "--val_every", "3", "--save_every", "3"]),
    "lr0.001_ga16_30steps": ("lrcheck_v2_all_lr0.001", ["--lr", "1e-3", "--grad_accum", "16", "--max_steps", "30",
                                                        "--train_limit", "480", "--val_every", "10"]),
    "lr0.0001_ga16_30steps": ("lrcheck_v2_all_lr0.0001", ["--lr", "1e-4", "--grad_accum", "16", "--max_steps", "30",
                                                          "--train_limit", "480", "--val_every", "10"]),
}
GROUPS = {"smoke": ["lr0.1_ga4_3steps"], "lr": ["lr0.001_ga16_30steps", "lr0.0001_ga16_30steps"]}


def run(name):
    out, extra = RUNS[name]
    print(f"=== {name} -> outputs/{out} ===", flush=True)
    cmd = [sys.executable, "train/train_delims_v2.py", *COMMON, *extra, "--out", f"outputs/{out}"]
    if subprocess.run(cmd).returncode != 0:
        print(f"!!! failed: {name}", flush=True)


def copy():
    for name, (out, _) in RUNS.items():
        src, dst = os.path.join("outputs", out), os.path.join("results", "sweep_v2", name)
        if not os.path.exists(os.path.join(src, "log.jsonl")):
            print(f"skip {name}: outputs/{out}/log.jsonl not found")
            continue
        os.makedirs(dst, exist_ok=True)
        for f in ("config.json", "log.jsonl"):
            shutil.copy(os.path.join(src, f), os.path.join(dst, f))
        print(f"copied outputs/{out} -> {dst}")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("smoke", "lr", "copy"):
        raise SystemExit(__doc__)
    if sys.argv[1] == "copy":
        copy()
    else:
        for name in GROUPS[sys.argv[1]]:
            run(name)
