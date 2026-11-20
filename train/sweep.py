"""
Hyperparameter sweep on a 2,000-row subset. Works on Windows and Mac.
Stage 1: learning rate x effective batch, header init.
Stage 2: random init with the best stage-1 setting (the header run is already in stage 1).
Usage: python train/sweep.py stage1      or   python train/sweep.py stage2 <lr> <grad_accum>
"""
import subprocess
import sys

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
COMMON = ["--model", MODEL, "--load_in_4bit", "--train_limit", "2000", "--val_limit", "100",
          "--save_every", "1000", "--log_every", "1"]


def run(lr, ga, init):
    print(f"=== lr={lr} grad_accum={ga} init={init} ===", flush=True)
    cmd = [sys.executable, "train/train_delims.py", *COMMON, "--lr", lr, "--grad_accum", str(ga),
           "--init", init, "--val_every", str(max(1, 160 // ga))]
    if subprocess.run(cmd).returncode != 0:
        print(f"!!! failed: lr={lr} grad_accum={ga} init={init}", flush=True)


if sys.argv[1] == "stage1":
    for lr in ["1e-4", "1e-3", "1e-2"]:
        for ga in [16, 64]:
            run(lr, ga, "header")
elif sys.argv[1] == "stage2":
    run(sys.argv[2], int(sys.argv[3]), "random")