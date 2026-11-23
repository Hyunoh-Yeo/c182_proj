"""Regenerate every figure and table in the paper from files already in the repo.

    python paper/make_figures.py

Inputs (no model or GPU needed):
  results/attacks/*.jsonl                      AlpacaFarm outputs, data filter on (main results)
  additional/nofilter/*.jsonl                  the same runs with the data filter off; also the
                                               step-300/600 checkpoints, which were only run this way
  results/sep/*.csv                            SEP outputs
  checkpoints/insert_n2_lr0.1_ga16/log.jsonl   training log of the final run
Outputs:
  paper/figures/*.pdf, paper/tables/*.tex
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "attacks"))
from summarize_results import any_attack, fisher, load, wilson  # noqa: E402
from injections import ADAPTIVE, STATIC, attacked  # noqa: E402
from eval.summarize_sep import load as load_sep, word_hit  # noqa: E402

RUN = ROOT / "checkpoints" / "insert_n2_lr0.1_ga16"
MAIN = ROOT / "results" / "attacks"
NOFILTER = ROOT / "additional" / "nofilter"
SEP = ROOT / "results" / "sep"
FIG = ROOT / "paper" / "figures"
TAB = ROOT / "paper" / "tables"

# (key, label in the paper). Order = order in every figure and table.
DEFENSES = [
    ("undefended", "Undefended"),
    ("no-delims", "Input-role template, no delimiters"),
    ("defensivetoken", "DefensiveToken (5 vectors)"),
    ("secpo", "SecPO (LoRA fine-tune)"),
    ("ours_s1163", "Ours (2 vectors, final)"),
]
# Training-curve checkpoints: all three exist only with the filter off, so the curve uses those runs.
CHECKPOINTS = [("ours_s300", 300), ("ours_s600", 600), ("ours_s1163", 1163)]
SEP_FILES = {"undefended": "undefended", "no-delims": "no-delims", "defensivetoken": "defensivetoken",
             "secpo": "secpo", "ours_s1163": "ours"}
FILTER_ROWS = [("undefended", "Undefended"), ("no-delims", "No delimiters"), ("ours_s1163", "Ours (final)")]

# Colors: one highlight for our method, neutral gray for everything else (one job: identity of "ours").
BLUE, ORANGE, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#a3a29d", "#0b0b0b", "#52514e"

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#e4e3df", "grid.linewidth": 0.6, "axes.axisbelow": True,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
})


def runs():
    """Main (filtered) AlpacaFarm runs, and the unfiltered ones under the key ("nofilter", name)."""
    R = {k: load(MAIN / f"{k}.jsonl") for k, _ in DEFENSES}
    for k in {c for c, _ in CHECKPOINTS} | {k for k, _ in FILTER_ROWS}:
        R["nofilter", k] = load(NOFILTER / f"{k}.jsonl")
    return R


def sep_runs():
    """defense -> list of (output, witness); every defense must cover the same SEP rows."""
    S = {k: load_sep(SEP / f"{f}.csv") for k, f in SEP_FILES.items()}
    idx = {k: sorted(r["idx"] for r in rows) for k, rows in S.items()}
    assert len({tuple(v) for v in idx.values()}) == 1, "SEP runs cover different rows"
    return S


def sep_rate(rows):
    k, n = sum(word_hit(r["output"], r["witness"]) for r in rows), len(rows)
    lo, hi = wilson(k, n)
    return k, n, k / n, lo, hi


def asr_any(by_attack):
    k, n = any_attack(by_attack, attacked)
    lo, hi = wilson(k, n)
    return k, n, k / n, lo, hi


# ---------------------------------------------------------------- Figure: method diagram
def fig_method():
    """Token layout of the prompt. Colors mark who wrote each span and what is trained."""
    spans = [
        ("BOS", "tmpl"), ("user", "tmpl"), ("instruction", "trusted"), ("input", "tmpl"),
        ("$d_1$", "delim"), ("untrusted data", "data"), ("$d_2$", "delim"),
        ("assistant", "tmpl"), ("response", "out"),
    ]
    style = {
        "tmpl": ("#f0efec", INK, "chat-template header (frozen)"),
        "trusted": ("#cde2fb", INK, "trusted instruction"),
        "data": ("#fbd9cb", INK, "untrusted data (may hold an injection)"),
        "delim": (BLUE, "white", "learned delimiter (the only trained parameters)"),
        "out": ("white", INK, "generated response"),
    }
    widths = {"instruction": 1.5, "untrusted data": 2.0, "response": 1.25, "$d_1$": 0.5, "$d_2$": 0.5,
              "assistant": 1.15, "BOS": 0.7}
    fig, ax = plt.subplots(figsize=(6.8, 1.6))
    x, gap = 0.0, 0.08
    for text, kind in spans:
        w = widths.get(text, 0.8)
        face, color, _ = style[kind]
        ax.add_patch(FancyBboxPatch((x, 0), w, 0.5, boxstyle="round,pad=0,rounding_size=0.06",
                                    fc=face, ec=MUTED if kind == "out" else face, lw=0.8,
                                    ls="--" if kind == "out" else "-"))
        ax.text(x + w / 2, 0.25, text, ha="center", va="center", color=color, fontsize=8.5,
                style="italic" if kind == "tmpl" else "normal")
        if text == "untrusted data":
            data_x = (x, w)
        x += w + gap
    x -= gap
    ax.plot([0, x], [0.72, 0.72], color=MUTED, lw=0.8)
    ax.text(x / 2, 0.8, "frozen Llama-3.1-8B-Instruct reads the whole sequence; no model weight changes",
            ha="center", va="bottom", fontsize=8, color=MUTED)
    ax.text(data_x[0] + data_x[1] / 2, -0.08, "attacker controls only this span", ha="center", va="top",
            fontsize=8, color=ORANGE)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=style[k][0], ec=MUTED if k == "tmpl" else style[k][0], lw=0.5)
               for k in ["trusted", "data", "delim", "tmpl"]]
    ax.legend(handles, [style[k][2] for k in ["trusted", "data", "delim", "tmpl"]], ncol=2, frameon=False,
              loc="upper center", bbox_to_anchor=(0.5, 0.06), fontsize=8, handlelength=1.2,
              columnspacing=1.5)
    ax.set_xlim(-0.05, x + 0.05)
    ax.set_ylim(-0.55, 1.1)
    ax.axis("off")
    fig.savefig(FIG / "method.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- Figure: ASR per defense
def fig_asr(R, S):
    """Left: AlpacaFarm, any of 10 attacks. Right: SEP, witness as a whole word. Both with Wilson CIs."""
    panels = [
        ([(label, *asr_any(R[k]), k) for k, label in DEFENSES], 150, [0, 25, 50, 75, 100],
         "AlpacaFarm: any of 10 attacks (%)"),
        ([(label, *sep_rate(S[k]), k) for k, label in DEFENSES], 70, [0, 15, 30, 45],
         "SEP: witness in output (%)"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 1.9), sharey=True, gridspec_kw={"wspace": 0.1})
    for ax, (rows, xmax, ticks, xlabel) in zip(axes, panels):
        for y, (label, k, n, p, lo, hi, key) in enumerate(reversed(rows)):
            c = BLUE if key.startswith("ours") else GRAY
            ax.barh(y, 100 * p, height=0.6, color=c)
            ax.errorbar(100 * p, y, xerr=[[100 * (p - lo)], [100 * (hi - p)]], fmt="none",
                        ecolor=INK, elinewidth=0.9, capsize=2.5)
            ax.text(100 * hi + xmax * 0.015, y, f"{100 * p:.1f}%  ({k}/{n})", va="center", fontsize=7.5,
                    color=INK)
        ax.set_yticks(range(len(rows)), [r[0] for r in reversed(rows)])
        ax.set_xlim(0, xmax)
        ax.set_xticks(ticks)
        ax.set_xlabel(xlabel)
        ax.grid(axis="y", visible=False)
    fig.savefig(FIG / "asr.pdf")
    plt.close(fig)
    return panels[0][0], panels[1][0]


# ---------------------------------------------------------------- Figure: training curves
def ema(xs, a=0.05):
    out, m = [], None
    for x in xs:
        m = x if m is None else a * x + (1 - a) * m
        out.append(m)
    return out


def fig_training(R):
    log = [json.loads(l) for l in open(RUN / "log.jsonl")]
    tr = [(r["step"], r["loss"]) for r in log if "loss" in r]
    va = [(r["step"], r["val_loss"], r["val_beats_ref"]) for r in log if "val_loss" in r]
    floor = 1e-6  # loss reaches ~1e-12 on some batches; clip so the log axis stays readable
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(5.5, 3.6), sharex=True,
                                 gridspec_kw={"height_ratios": [1.5, 1], "hspace": 0.12})
    s, l = zip(*tr)
    a1.plot(s, [max(x, floor) for x in l], color=GRAY, lw=0.5, alpha=0.6, label="train loss (per step)")
    a1.plot(s, [max(x, floor) for x in ema(l)], color=MUTED, lw=1.5, label="train loss (EMA, 0.05)")
    vs, vl, vb = zip(*va)
    a1.plot(vs, [max(x, floor) for x in vl], color=BLUE, lw=2, marker="o", ms=4,
            label="validation loss (100 held-out pairs)")
    a1.axhline(math.log(2), color=MUTED, lw=0.8, ls=":")
    a1.text(1163, math.log(2) * 1.25, "log 2 (zero margin)", ha="right", va="bottom", fontsize=7, color=MUTED)
    a1.set_yscale("log")
    a1.set_ylim(floor * 0.7, 20)
    a1.set_ylabel("SecPO loss (log scale)")
    a1.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, frameon=False, fontsize=7.5,
               handlelength=1.5, columnspacing=1.2)

    ck = [(step, *asr_any(R["nofilter", k])) for k, step in CHECKPOINTS]
    a2.plot(vs, [100 * (1 - b) for b in vb], color=BLUE, lw=2, marker="o", ms=4,
            label="validation pairs with margin below reference (%)")
    xs = [c[0] for c in ck]
    ys = [100 * c[3] for c in ck]
    err = [[100 * (c[3] - c[4]) for c in ck], [100 * (c[5] - c[3]) for c in ck]]
    a2.errorbar(xs, ys, yerr=err, color=ORANGE, lw=2, marker="s", ms=5, capsize=3,
                label="AlpacaFarm attack success, filter off (%), 95% Wilson CI")
    for x, y in zip(xs, ys):
        a2.text(x + 15, y + 2, f"{y:.1f}", fontsize=7.5, color=INK)
    a2.set_ylim(-3, 100)
    a2.set_ylabel("%")
    a2.set_xlabel("Optimizer step (effective batch 16)")
    a2.legend(loc="upper right", frameon=False)
    fig.savefig(FIG / "training.pdf")
    plt.close(fig)
    return va, ck


# ---------------------------------------------------------------- Figure: delimiter norms
def fig_norms():
    log = [json.loads(l) for l in open(RUN / "log.jsonl")]
    s = [r["step"] for r in log]
    d1 = [r["delim_norms"][0] for r in log]
    d2 = [r["delim_norms"][1] for r in log]
    vocab = log[0]["vocab_norm"]
    fig, ax = plt.subplots(figsize=(5.5, 2.0))
    ax.plot(s, d1, color=BLUE, lw=2)
    ax.plot(s, d2, color=ORANGE, lw=2)
    ax.text(s[-1], d1[-1] - 6, f"opening $d_1$: {d1[-1]:.0f}", ha="right", va="top", fontsize=8, color=INK)
    ax.text(s[-1], d2[-1] + 4, f"closing $d_2$: {d2[-1]:.0f}", ha="right", va="bottom", fontsize=8, color=INK)
    ax.axhline(vocab, color=MUTED, lw=1, ls="--")
    ax.text(s[-1], vocab * 1.4, f"mean vocabulary embedding norm: {vocab:.2f}", ha="right", va="bottom",
            fontsize=8, color=MUTED)
    ax.set_yscale("log")
    ax.set_ylim(0.3, 300)
    ax.set_ylabel("L2 norm (log scale)")
    ax.set_xlabel("Optimizer step")
    fig.savefig(FIG / "norms.pdf")
    plt.close(fig)
    return d1[-1], d2[-1], vocab


# ---------------------------------------------------------------- Tables
def tab_per_attack(R):
    keys = [k for k, _ in DEFENSES]
    heads = ["Undef.", "No delim.", "DefTok", "SecPO", "Ours"]
    names = {a: a.replace("_", r"\_") for a in STATIC + ADAPTIVE}
    lines = [r"\begin{tabular}{l" + "r" * len(keys) + "}", r"\toprule",
             "Attack & " + " & ".join(heads) + r" \\", r"\midrule"]
    for group, attacks in [("static", STATIC), ("adaptive", ADAPTIVE)]:
        lines.append(rf"\multicolumn{{{len(keys) + 1}}}{{l}}{{\emph{{{group}}}}} \\")
        for a in attacks:
            cells = []
            for k in keys:
                outs = R[k][a]
                cells.append(f"{100 * sum(attacked(o) for o in outs.values()) / len(outs):.1f}")
            lines.append(rf"\texttt{{{names[a]}}} & " + " & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    cells = [f"{100 * asr_any(R[k])[2]:.1f}" for k in keys]
    lines.append(r"Any of the 10 & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "per_attack.tex").write_text("\n".join(lines) + "\n")


def rate_cell(k, n):
    lo, hi = wilson(k, n)
    return f"{100 * k / n:.1f} [{100 * lo:.1f}, {100 * hi:.1f}]"


def tab_sep(S):
    ref = sep_rate(S["ours_s1163"])
    lines = [r"\begin{tabular}{lrrr}", r"\toprule",
             r"Defense & Broken / 500 & ASR \% [95\% CI] & $p$ vs.\ ours \\", r"\midrule"]
    for k, label in DEFENSES:
        hit, n, *_ = sep_rate(S[k])
        p = "--" if k == "ours_s1163" else (lambda v: "$<$0.0001" if v < 1e-4 else f"{v:.2f}")(
            fisher(hit, n, ref[0], ref[1]))
        lines.append(f"{label} & {hit} & {rate_cell(hit, n)} & {p} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "sep.tex").write_text("\n".join(lines) + "\n")


def tab_filter(R):
    """Same runs with the data filter off and on: any-attack ASR and the two attacks it affects."""
    def per(by, a):
        outs = by[a]
        return f"{100 * sum(attacked(o) for o in outs.values()) / len(outs):.1f}"
    lines = [r"\begin{tabular}{lcccccc}", r"\toprule",
             r" & \multicolumn{2}{c}{Any of 10 attacks} & \multicolumn{2}{c}{\texttt{completion\_chat}}"
             r" & \multicolumn{2}{c}{\texttt{completion\_chat\_ignore}} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
             r"Defense & off & on & off & on & off & on \\", r"\midrule"]
    for k, label in FILTER_ROWS:
        off, on = R["nofilter", k], R[k]
        cells = [rate_cell(*any_attack(off, attacked)), rate_cell(*any_attack(on, attacked)),
                 per(off, "completion_chat"), per(on, "completion_chat"),
                 per(off, "completion_chat_ignore"), per(on, "completion_chat_ignore")]
        lines.append(label + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "filter.tex").write_text("\n".join(lines) + "\n")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    R, S = runs(), sep_runs()
    fig_method()
    rows, sep_rows = fig_asr(R, S)
    va, ck = fig_training(R)
    norms = fig_norms()
    tab_per_attack(R)
    tab_sep(S)
    tab_filter(R)

    for name, table in [("AlpacaFarm any-attack ASR", rows), ("SEP whole-word ASR", sep_rows)]:
        print(f"{name} (Wilson 95%):")
        for label, k, n, p, lo, hi, _ in table:
            print(f"  {label:38s} {k:3d}/{n} = {100 * p:5.1f}%  [{100 * lo:.1f}, {100 * hi:.1f}]")
    print("checkpoints, filter off:", [(s, f"{100 * p:.1f}%") for s, _, _, p, _, _ in ck])
    print("val loss at 300/600/1163:", [(s, round(l, 4)) for s, l, _ in va if s in (300, 600, 1163)])
    print("final delimiter norms %.1f %.1f, vocab mean %.2f" % norms)
    print(f"wrote {sorted(p.name for p in FIG.iterdir())} and {sorted(p.name for p in TAB.iterdir())}")


if __name__ == "__main__":
    main()
