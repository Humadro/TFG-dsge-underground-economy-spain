# Project context for Claude Code

This repo is a **pruned, self-sufficient extract** of a larger working repo
(`RBC-parameter-identification`, the original TFG working directory) — see
`README.md` for what it contains and the full reproduction pipeline. It is
not itself the working repo; don't assume history or files beyond what's
checked in here exist anywhere.

## Status

- **Text is still in Catalan** (`lang: ca` in `_quarto.yml`). Translating
  `writing/*.qmd` + `index.qmd` to English is the next planned phase, done
  chapter by chapter, reviewed before merging each one. Don't translate
  unprompted or all at once.
- The book renders with `quarto render` and needs **no Julia at runtime** —
  zero executable code chunks in any `.qmd`, all figures are pre-rendered
  PNGs already checked into `outputs/`. Julia is only needed to *regenerate*
  those figures from raw data (see README pipeline).
- `code/post_analisis/post_analisis.jl`'s `estimated_params` loads
  automatically from `outputs/orsi/real_no_covid_3000iter/cadenes_orsi_real.jld2`
  via `load_posterior_means()` — do not reintroduce a hand-copied literal
  Dict; if the estimation is rerun, that `.jld2` is the single source of
  truth.

## Conventions when touching the Julia pipeline

- All scripts assume **repo root as CWD** (`julia --project=. code/....jl`),
  except `include()` calls which are `@__DIR__`-relative.
- NUTS sampler settings that have worked well in this project (from prior
  tuning): `AutoForwardDiff(chunksize=...)` as the AD backend, `max_depth`
  around 5-8, 1000 adaptation steps — see `code/bayes/*.jl` for the exact
  values in use; don't casually change sampler settings without a reason,
  convergence here was hard-won (`rhat` ≈ 1.00 across all params in
  `outputs/orsi/real_no_covid_3000iter/describe_orsi_real.txt`).
- `p_ss` (inspection probability) is calibrated from `data/p_inspeccion.xlsx`
  by hand, not by any script — that file has no consuming code and that's
  expected, not a bug.

## What NOT to do

- Don't restore the Python loaders or the abandoned basic-RBC
  Bayes-vs-Deep-Learning code — both were deliberately dropped (see
  README's "What was deliberately left out").
- Don't add files back from the original working repo without checking
  they're actually referenced by `writing/*.qmd` first — this repo is
  intentionally minimal.
