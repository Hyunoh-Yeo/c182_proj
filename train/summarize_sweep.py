"""
Plot runs under outputs/: val margin accuracy, val SecPO loss, delimiter norm. x = samples seen.
Usage: python train/summarize_sweep.py [pattern] [name]
  pattern selects run folders (default "*"), e.g. "*ga16_std_normal" for the lr sweep
  name sets the output file outputs/<name>.png (default sweep_plots)
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
pattern = sys.argv[1] if len(sys.argv) > 1 else "*"
name_out = sys.argv[2] if len(sys.argv) > 2 else "sweep_plots"
for log in sorted(glob.glob(f"outputs/{pattern}/log.jsonl")):
    name = os.path.basename(os.path.dirname(log))
    ga = int(name.split("_ga")[1].split("_")[0])
    vals = [r for r in map(json.loads, open(log)) if "val_margin_acc" in r]
    if not vals:
        continue
    x = [v["step"] * ga for v in vals]
    axes[0].plot(x, [v["val_margin_acc"] for v in vals], marker="o", label=name)
    axes[1].plot(x, [v["val_loss"] for v in vals], marker="o")
    axes[2].plot(x, [sum(v["delim_norms"]) / len(v["delim_norms"]) / v["vocab_norm"] for v in vals], marker="o")
    print(f"{name:45s} step {vals[-1]['step']:4d}  acc {vals[-1]['val_margin_acc']:.3f}  loss {vals[-1]['val_loss']:.4f}")
for ax, title in zip(axes, ["val margin accuracy", "val SecPO loss", "delimiter norm / vocab norm (log)"]):
    ax.set_title(title)
    ax.set_xlabel("training samples seen")
axes[2].set_yscale("log")  # header starts near 0.05x, N(0, 1) near 100x
axes[0].legend(fontsize=7)
fig.tight_layout()
fig.savefig(f"outputs/{name_out}.png", dpi=150)
print(f"saved outputs/{name_out}.png")