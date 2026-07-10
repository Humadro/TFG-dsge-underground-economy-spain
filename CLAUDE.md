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

- Don't restore the Python loaders — deliberately dropped (see README's
  "What was deliberately left out").
- Don't add files back from the original working repo without checking
  they're actually referenced by `writing/*.qmd` first — this repo is
  intentionally minimal.
- Don't silently re-add the DL+Indirect-Inference code (see below) without
  discussing it first — it's kept out of the reproducible pipeline on
  purpose, but the approach itself is meant to be revisited later.

## DL + Indirect Inference (discarded methodology — kept for future work)

`writing/03_metodologies.qmd` §"Aproximació explorada: Inferència Indirecta
amb Aprenentatge Profund" (`#sec-dl-ii`) documents, narratively only, an
alternative estimation approach that was partially built and then dropped in
favor of the Bayesian/NUTS route used everywhere else in this project. The
text explicitly flags it as "a very interesting avenue to explore in future
work" — that's the intent for keeping this note.

**What it was:** estimate a *basic* RBC model (not Orsi-Turino — this predates
the underground-economy extension) by combining:

1. A **deep-learning solver** (Maliar, Maliar & Winant 2021 "AiO" — All-in-One
   operator) approximating the policy functions `(c_t, n_t) = Ψ_θ(k_{t-1}, A_t)`
   with a small MLP (`Lux.jl`), trained by minimizing Euler-equation residuals
   via ADAM. The AiO trick: draw *two* independent future shock realizations
   and multiply the two residuals instead of squaring an expectation, which
   avoids nested Monte Carlo integration inside the loss.
2. **Indirect Inference** (Gouriéroux, Monfort & Renault 1993) as the outer
   loop: estimate structural parameters `Θ = (α, β, δ, γ, ρ)` by matching
   VAR(1) coefficients on simulated data (from the trained network) against
   VAR(1) coefficients on real/simulated observables, minimizing a quadratic
   distance `J(Θ)` with Nelder-Mead (non-differentiable objective, since each
   evaluation retrains the network stochastically).

**Where the code is:** restored at `code/rbc/DL/dl.jl` (recovered from the
*original* working repo's git history, `git -C <path-to>/RBC-parameter-identification
show cb7f190:code/rbc/DL/dl.jl`). It is **not** part of the reproducible
pipeline in `README.md` — it's kept for reference/future work only, and its
extra deps (`Lux`, `Zygote`, `Optimisers`) are deliberately **not** in
`Project.toml`. Add them yourself (`Pkg.add(["Lux", "Zygote", "Optimisers"])`)
if you resume this line of work. It also expects `data/simulated_data.csv`,
which doesn't exist in this repo (it read from the old basic-RBC pipeline,
not the Orsi-Turino one) — path needs updating before it'll run.
Also check the old repo's `outputs/dl_rbc/` and `outputs/bayes_rbc/` (not
copied here) for the last figures/results produced.

**Why it was actually dropped (beyond "ran out of time"):** the last run's
diagnostics, in the old repo at `outputs/dl_rbc/resultats_dl_{sim,real}.txt`,
show real problems, not just unfinished polish — worth knowing before
resuming this:
- On simulated data (known ground truth): parameter recovery was mediocre
  (`γ` off by 24.5%, `σ_ε` off by 110%; `α, β, δ, ρ` within ~1-12%).
- Gouriéroux/Monfort/Renault (1993) diagnostic tests — both the global
  specification test and the proxy-consistency test — **strongly rejected
  H₀** (p ≈ 0.0000) on both simulated and real data, meaning the chosen
  VAR(1) auxiliary moments were not actually well-identified/consistent for
  this model as implemented. Several individual moment z-scores were huge
  outliers (e.g. `n̄`, `y/i`, `σ(ŷ)` on simulated data).
- One inconsistency between code and text worth resolving if this is
  revived: the text (`03_metodologies.qmd` line ~96) says the network uses
  `ReLU` hidden activations; the actual `dl.jl` code uses `tanh`.

If this is picked back up for the paper, it would need: fixing/replacing the
auxiliary-moment set (the VAR(1) choice looks under-identified per the GMR
tests above), re-validating the AiO training loop's convergence, and likely
extending it to the full Orsi-Turino model rather than the basic RBC toy
model it was built against.
