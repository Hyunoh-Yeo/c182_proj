# Paper

Regenerate every figure and table from files already in the repo (no GPU, no model download):

```
python paper/make_figures.py
```

Inputs: `results/attacks/*.jsonl` and `checkpoints/insert_n2_lr0.1_ga16/log.jsonl`.
Outputs: `paper/figures/{method,asr,training,norms}.pdf`, `paper/tables/per_attack.tex`.

## Compiling

The course requires the NIPS 2015 style. On Overleaf, make a Blank Project and upload `main.tex`,
`refs.bib`, `figures/`, `tables/` and `nips15submit_e.sty`
(from http://media.nips.cc/Conferences/2015/Styles/nips15submit_e.sty). `\nipsfinalcopy` in `main.tex`
prints author names; comment it out for an anonymous version.

Orange `[In progress: ...]` and red `[TODO: ...]` markers show what is unfinished. Each TODO names who
it is for.
