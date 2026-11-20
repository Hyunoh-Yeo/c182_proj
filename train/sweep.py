"""
Hyperparameter sweep on a 2,000-row subset, one axis at a time. Works on Windows and Mac.
lr:   learning rate at grad_accum 16, with N(0, 1) init (as in DefensiveToken and LTBD).
ga:   grad_accum at a chosen lr.
init: the other inits at a chosen lr and grad_accum.
Usage: python train/sweep.py lr | ga <lr> | init <lr> <grad_accum>
"""
import subprocess
import sys

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
COMMON = ["--model", MODEL, "--load_in_4bit", "--train_limit", "2000", "--val_limit", "100",
          "--save_every", "1000", "--log_every", "1"]
INIT = "std_normal"


def run(lr, ga, init):
    print(f"=== lr={lr} grad_accum={ga} init={init} ===", flush=True)
    cmd = [sys.executable, "train/train_delims.py", *COMMON, "--lr", lr, "--grad_accum", str(ga),
           "--init", init, "--val_every", str(max(1, 160 // ga))]
    if subprocess.run(cmd).returncode != 0:
        print(f"!!! failed: lr={lr} grad_accum={ga} init={init}", flush=True)


if sys.argv[1] == "lr":
    for lr in ["1e-2", "1e-1", "1"]:
        run(lr, 16, INIT)
elif sys.argv[1] == "ga":
    for ga in [32, 64]:
        run(sys.argv[2], ga, INIT)
elif sys.argv[1] == "init":
    for init in ["header", "mean", "zero"]:
        run(sys.argv[2], int(sys.argv[3]), init)