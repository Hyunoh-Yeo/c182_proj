#!/usr/bin/env bash
# Hyperparameter sweep on a 2,000-row subset.
# Stage 1: learning rate x effective batch (6 runs). Stage 2: init, with the best stage-1 setting.
# Usage: bash train/sweep.sh stage1        or   bash train/sweep.sh stage2 <lr> <grad_accum>
set -e
MODEL=${MODEL:-meta-llama/Llama-3.1-8B-Instruct}
EXTRA=${EXTRA:---load_in_4bit}
COMMON="--model $MODEL $EXTRA --train_limit 2000 --val_limit 100 --val_every 10 --save_every 1000 --log_every 1"

if [ "$1" = "stage1" ]; then
  for lr in 1e-3 1e-2 1e-1; do
    for ga in 32 256; do
      echo "=== lr=$lr grad_accum=$ga ==="
      python train/train_delims.py $COMMON --lr $lr --grad_accum $ga --init header
    done
  done
elif [ "$1" = "stage2" ]; then
  for init in header random; do
    echo "=== lr=$2 grad_accum=$3 init=$init ==="
    python train/train_delims.py $COMMON --lr $2 --grad_accum $3 --init $init
  done
else
  echo "usage: bash train/sweep.sh stage1 | stage2 <lr> <grad_accum>"
fi