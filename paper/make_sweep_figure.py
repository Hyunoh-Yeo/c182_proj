"""Figure and numbers for the hyperparameter sweep (Appendix B), from the logs in results/sweep/.

    python paper/make_sweep_figure.py

Inputs: results/sweep/<run>/log.jsonl and config.json, written by train/sweep.py (2,000 training rows,
100 validation pairs, N(0, 1) init). Output: paper/figures/sweep.pdf, and the numbers quoted in the
appendix table printed to stdout. Same style as paper/make_figures.py; kept separate so the two
scripts can be edited independently.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SWEEP = ROOT / "results" / "sweep"
FIG = ROOT / "paper" / "figures"

BLUE, ORANGE, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#a3a29d", "#0b0b0b", "#52514e"
plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#e4e3df", "grid.linewidth": 0.6, "axes.axisbelow": True,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
})

# (run folder, label, color). The selected setting (lr 0.1, batch 16) is blue in both panels.
LR_RUNS = [("lr0.01_ga16", "lr 0.01", GRAY), ("lr0.1_ga16", "lr 0.1 (chosen)", BLUE), ("lr1.0_ga16", "lr 1", ORANGE)]
GA_RUNS = [("lr0.1_ga16", "batch 16 (chosen)", BLUE), ("lr0.1_ga32", "batch 32", GRAY), ("lr0.1_ga64", "batch 64", ORANGE)]


def load(run):
    cfg = json.load(open(SWEEP / run / "config.json"))
    vals = [r for r in map(json.loads, open(SWEEP / run / "log.jsonl")) if "val_loss" in r]
    ga = cfg["grad_accum"]
    return cfg, [(v["step"] * ga, v["val_loss"], v["val_margin_acc"], sum(v["delim_norms"]) / 2, v["vocab_norm"])
                 for v in vals]


def at_samples(points, n):
    """Validation point at exactly n training samples seen (every run logs one there)."""
    return next(p for p in points if p[0] == n)


def panel(ax, runs, title):
    for run, label, color in runs:
        _, pts = load(run)
        x, y = [p[0] for p in pts], [p[1] for p in pts]
        ax.plot(x, y, color=color, lw=2 if color == BLUE else 1.5, marker="o", ms=3, label=label)
    ax.axhline(math.log(2), color=MUTED, lw=0.8, ls=":")
    ax.text(20, math.log(2) * 1.08, "log 2", ha="left", va="bottom", fontsize=7, color=MUTED)
    ax.set_yscale("log")
    ax.set_ylim(0.04, 10)
    ax.set_xlim(-40, 2040)
    ax.set_xlabel("Training samples seen")
    ax.set_title(title, loc="left")
    ax.legend(loc="lower left", frameon=False)


def main():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.5, 2.3), sharey=True)
    panel(a1, LR_RUNS, "Learning rate (batch 16)")
    panel(a2, GA_RUNS, "Effective batch (lr 0.1)")
    a1.set_ylabel("Validation SecPO loss (log)")
    _, ga64 = load("lr0.1_ga64")
    a2.text(ga64[-1][0] + 40, ga64[-1][1], "run stopped\nat 640 samples", fontsize=7, color=MUTED, va="center")
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "sweep.pdf")
    fig.savefig(FIG / "sweep.png", dpi=200)
    plt.close(fig)

    print("run            samples  val_loss  margin_acc  norm  norm/vocab   val_loss@640")
    for run in ["lr0.01_ga16", "lr0.1_ga16", "lr1.0_ga16", "lr0.1_ga32", "lr0.1_ga64"]:
        _, pts = load(run)
        n, loss, acc, norm, vocab = pts[-1]
        print(f"{run:13s} {n:8d} {loss:9.3f} {acc:11.2f} {norm:5.0f} {norm / vocab:11.0f} {at_samples(pts, 640)[1]:13.3f}")
    print(f"start (step 0): val_loss {load('lr0.1_ga16')[1][0][1]:.3f}")
    print(f"saved {FIG / 'sweep.pdf'}")


if __name__ == "__main__":
    main()
