# DSGE Estimation of the Underground Economy in Spain

Self-contained rebuild of a thesis project estimating an Orsi, Raggi & Turino
(2014) DSGE model of the underground economy, applied to Spain (1995-2024)
via Bayesian (NUTS) inference in Julia (`MacroModelling.jl` + `Turing.jl`).

This repository is a pruned, reproducible extract of a larger working
repository: it keeps only the code, data and outputs needed to (a) rebuild
every figure and table cited in the book (`writing/`), and (b) render the
book itself with Quarto. The text is currently in Catalan (`lang: ca` in
`_quarto.yml`); an English translation is planned as a follow-up pass and is
not part of this reproduction pipeline.

## Requirements

- **Julia** ≥ 1.10
- **Quarto** ≥ 1.4, with a LaTeX/XeLaTeX distribution (TinyTeX is enough) for
  the PDF format, and a working `xelatex` with the `Arial` font available for
  `pdf` output (HTML output has no such requirement)
- No Python is required — the two `.py` loaders that originally produced
  `data/aeat_is_irpf_trimestral.csv` were dropped; that CSV (and
  `data/p_inspeccion.xlsx`, hand-extracted from AEAT inspection statistics)
  are checked in directly as raw inputs

## Setup

```julia
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

This resolves a fresh `Manifest.toml` from `Project.toml` (none is checked
in). `MacroModelling.jl` and `Turing.jl` pull in a large dependency tree —
first instantiation can take a while.

All scripts below assume the **repository root** as the working directory
(`julia --project=. code/....jl`), not the `code/` folder itself.

## Reproduction pipeline

`outputs/` and the derived `data/orsi/*.csv` files are already checked in,
so `quarto render` works out of the box without rerunning anything. To
regenerate them from raw data, run the following in order:

1. **Prepare real data**
   ```
   julia --project=. code/load_real_data_orsi.jl
   ```
   Reads `data/namq_10_gdp*.csv`, `data/eurostat_population_1564.csv` and
   `data/aeat_is_irpf_trimestral.csv` (all Eurostat/AEAT sources), and writes
   `data/orsi/real_data_orsi.csv` + `real_data_orsi_levels.csv` — 5 observable
   series (`obs_c, obs_inv, obs_Gc, obs_Gh, obs_wh`), 1995Q1-2024Q4.

2. **Simulate data** (recovery check, used in `writing/05_simulat.qmd`)
   ```
   julia --project=. code/simulate_data_orsi.jl
   ```
   Writes `data/orsi/simulated_data_orsi.csv` and
   `outputs/orsi/observables_simulats.png`.

3. **Bayesian estimation — simulated data**
   ```
   julia --project=. code/bayes/bayesian_estimation_orsi_sim.jl
   ```
   NUTS, 3 chains. Writes `outputs/orsi/cadenes_orsi_sim_T120.jld2`,
   `describe_orsi_sim_T120.txt`, and the 9 `{distr,trace,corner}_*_sim.png`
   figures embedded in `writing/annex.qmd`.

4. **Bayesian estimation — real data**
   ```
   julia --project=. code/bayes/bayesian_estimation_orsi_real.jl
   ```
   NUTS, 3 chains × 3000 samples (long — hours, not minutes). Writes to a
   timestamped `outputs/orsi/real_<yyyymmdd_HH>/` directory. The book uses
   two runs, distinguished by one commented-out line in the script
   (`filter!(row -> row.date < Date(2020,1,1), datos_real)`):
   - **pre-COVID** (line active) → rename the output dir to
     `outputs/orsi/real_no_covid_3000iter/` — this is the run
     `code/post_analisis/post_analisis.jl` reads by default
     (`REAL_CHAIN_PATH`) for every figure in `writing/07_analisis.qmd` and
     `writing/annex.qmd`.
   - **full sample incl. COVID** (line commented out, as checked in) →
     rename to `outputs/orsi/real_covid_3000iter/` — source of the
     robustness table in `writing/06_real.qmd`.

5. **Steady-state diagnostic** (optional, justifies the `B_0` calibration
   quoted in `writing/05_simulat.qmd`/`06_real.qmd`)
   ```
   julia --project=. code/check_steady_state.jl
   ```

6. **Post-estimation analysis** (IRFs, FEVD, historical decomposition, Laffer
   curves — everything `writing/07_analisis.qmd` and `writing/annex.qmd`
   embed)
   ```julia
   julia --project=.
   julia> include("code/post_analisis/post_analisis.jl")
   julia> main()  # writes to outputs/orsi/real_no_covid_post/ by default
   ```
   `estimated_params` is no longer hand-copied: it is loaded automatically
   from `outputs/orsi/real_no_covid_3000iter/cadenes_orsi_real.jld2` (step 4)
   via `load_posterior_means`, merged with the fixed/calibrated parameters
   (`CALIBRATED_PARAMS`, at the top of `post_analisis.jl`). Re-running step 4
   and this step reproduces the book's figures end to end with no manual
   transcription step.

7. **Counterfactual fiscal experiments** (source of the CF1-CF4 table in
   `writing/07_analisis.qmd`)
   ```
   julia --project=. code/post_analisis/counterfactuals.jl
   ```
   Writes `outputs/orsi/counterfactuals/counterfactuals.csv`.

## Rendering the book

```
quarto render
```

Output goes to `_output/` (HTML + PDF, per `_quarto.yml`).

## What was deliberately left out

Pruned relative to the original working repository: exploratory/debug
scripts (steady-state diagnostics aside), one-off output directories from
earlier estimation runs, the advisor-facing figure export script, the
defense-slides Quarto project, and an abandoned basic-RBC
Bayesian-vs-Deep-Learning comparison whose source code no longer exists
(chapters 02-03 discuss it narratively only, with no embedded figures or
runnable code). None of these affect the book's rendered output.
