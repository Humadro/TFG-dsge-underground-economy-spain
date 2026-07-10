# load_real_data_orsi.jl
# Carrega i transforma dades espanyoles reals en les 5 variables observables
# del model Orsi-Raggi-Turino. Guarda data/orsi/real_data_orsi.csv amb
# columnes idèntiques a data/orsi/simulated_data_orsi.csv.
#
# FONTS DE DADES (totes ja disponibles a data/):
# ─────────────────────────────────────────────
#  data/namq_10_gdp.csv          Eurostat NAMQ_10_GDP, Espanya, CLV 2010,
#                                SCA, trimestral. Conté: GDP, C (P3), I (P51G).
#  data/namq_10_gdp_nominal.csv  Mateixa query, CP_MEUR (preus corrents).
#                                Conté: GDP, C, I. S'usa per calcular deflactor.
#  data/namq_10_gdp_d1.csv       Eurostat NAMQ_10_GDP, D1 (remuneració
#                                d'assalariats), CP_MEUR, SCA, trimestral.
#                                Format Eurostat llarg: columna OBS_VALUE.
#  data/eurostat_population_1564.csv  Eurostat DEMO_PJANBROAD, Espanya,
#                                     15–64 anys, anual. Columna OBS_VALUE.
#                                     S'interpolaran linealment a trimestral.
#  data/aeat_is_irpf_trimestral.csv   AEAT, recaptació IS i IRPF en M€
#                                      (generat per load_aeat_fiscal.py).
#                                      Columnes: fecha, IS_Meur, IRPF_Meur.
#
# VARIABLES OBSERVABLES (eq. observació com a orsi.jl):
#  obs_x = 100 × Δlog(X_real_percapita)  per a X ∈ {c, inv, Gc, Gh, wh}
#  obs_Gs i obs_p no s'usen com a observables (p és variable latent).
#
# COBERTURA TEMPORAL:
#  1995-Q1 – 2024-Q4 (120 trimestres; 119 obs. en primeres diferències).
#  Es talla a 2024-Q4 per evitar el forat al 2025-Q2 de les dades AEAT.
#
# Executar des de l'arrel del projecte:
#   julia code/orsi/load_real_data_orsi.jl

using DataFrames, CSV, Dates, Statistics

mkpath("data/orsi")

# ── Helper: parseja "1995-Q1" → Date(1995, 1, 1) ─────────────────────────────
parse_quarter(s) = Date(parse(Int, s[1:4]), (parse(Int, s[end]) - 1) * 3 + 1, 1)

# ── [A] Consum i inversió reals (Eurostat CLV 2010, semicolons) ───────────────
df_real = CSV.read("data/namq_10_gdp.csv", DataFrame; delim = ';', missingstring = ["", "NA"])
# Elimina files buits al final (típics en exports d'Excel/Eurostat)
dropmissing!(df_real, :TIME_PERIOD)
df_real.date = parse_quarter.(df_real.TIME_PERIOD)
sort!(df_real, :date)

# ── [B] PIB nominal (Eurostat CP_MEUR, semicolons) → deflactor implícit ───────
df_nom = CSV.read("data/namq_10_gdp_nominal.csv", DataFrame; delim = ';', missingstring = ["", "NA"])
dropmissing!(df_nom, :TIME_PERIOD)
df_nom.date = parse_quarter.(df_nom.TIME_PERIOD)
sort!(df_nom, :date)

# Intersecció A ∩ B per alinear dates
common_AB = sort(intersect(df_real.date, df_nom.date))
idx_r  = [findfirst(==(d), df_real.date) for d in common_AB]
idx_n  = [findfirst(==(d), df_nom.date)  for d in common_AB]

GDP_real_AB = Float64.(df_real.GDP[idx_r])
GDP_nom_AB  = Float64.(df_nom.GDP[idx_n])
deflactor   = GDP_nom_AB ./ GDP_real_AB   # índex de preus (≈ 1 en base 2010)
C_real_AB   = Float64.(df_real.C[idx_r])
I_real_AB   = Float64.(df_real.I[idx_r])
dates_AB    = common_AB

# ── [C] D1 — Remuneració d'assalariats, nominal (Eurostat, format llarg) ──────
# Format: columna TIME_PERIOD (ex. "1995-Q1"), columna OBS_VALUE (M€ corrents)
df_d1 = CSV.read("data/namq_10_gdp_d1.csv", DataFrame; missingstring = ["", "NA"])
dropmissing!(df_d1, :TIME_PERIOD)
df_d1.date = parse_quarter.(string.(df_d1.TIME_PERIOD))
sort!(df_d1, :date)

# ── [E] IS i IRPF trimestrals de l'AEAT (format net, M€ nominals) ─────────────
df_aeat = CSV.read("data/aeat_is_irpf_trimestral.csv", DataFrame)
df_aeat.date = parse_quarter.(string.(df_aeat.fecha))
sort!(df_aeat, :date)

# ── [G] Població 15–64, dades anuals → interpolació lineal a trimestral ───────
# Format: columna TIME_PERIOD (any, ex. "1995"), columna OBS_VALUE (n. persones)
df_pop = CSV.read("data/eurostat_population_1564.csv", DataFrame; missingstring = ["", "NA"])
dropmissing!(df_pop, :TIME_PERIOD)
pop_years = parse.(Int, string.(df_pop.TIME_PERIOD))
pop_vals  = Float64.(df_pop.OBS_VALUE)
sort_idx  = sortperm(pop_years)
pop_years = pop_years[sort_idx]
pop_vals  = pop_vals[sort_idx]

# Interpola la població per a una data trimestral donada
function pop_at(d::Date, yrs::Vector{Int}, vals::Vector{Float64})
    yr   = Dates.year(d)
    qtr  = div(Dates.month(d) - 1, 3) + 1      # 1..4
    frac = (qtr - 1) / 4.0                      # Q1→0, Q2→0.25, Q3→0.5, Q4→0.75
    i    = searchsortedlast(yrs, yr)
    i == 0              && return vals[1]
    i >= length(yrs)    && return vals[end]
    return vals[i] * (1 - frac) + vals[i + 1] * frac
end

# ── Alinear totes les sèries ──────────────────────────────────────────────────
# Limitem a 2024-Q4 per evitar el forat al 2025-Q2 de les dades AEAT
date_max  = Date(2024, 10, 1)   # 2024-Q4
aeat_ok   = filter(d -> d <= date_max, df_aeat.date)
dates_alin = sort(intersect(dates_AB, df_d1.date, aeat_ok))

idx_AB   = [findfirst(==(d), dates_AB)     for d in dates_alin]
idx_d1   = [findfirst(==(d), df_d1.date)   for d in dates_alin]
idx_aeat = [findfirst(==(d), df_aeat.date)  for d in dates_alin]

pop_alin = [pop_at(d, pop_years, pop_vals) for d in dates_alin]
defl_alin = deflactor[idx_AB]

# Variables en termes reals per càpita
C_pc      = C_real_AB[idx_AB]    ./ pop_alin
inv_pc    = I_real_AB[idx_AB]    ./ pop_alin

D1_nom       = Float64.(df_d1.OBS_VALUE[idx_d1])
wh_real_pc   = (D1_nom ./ defl_alin) ./ pop_alin

IS_nom       = Float64.(df_aeat.IS_Meur[idx_aeat])
IRPF_nom     = Float64.(df_aeat.IRPF_Meur[idx_aeat])

# Suma mòbil de 4 trimestres per evitar valors negatius estacionals (devolucions
# fiscals concentrades al Q1-Q2), mantenint la sèrie sempre positiva.
roll4(x) = [sum(x[max(1,i-3):i]) for i in 1:length(x)]
IS_smooth    = roll4(IS_nom)
IRPF_smooth  = roll4(IRPF_nom)
Gc_real_pc   = (IS_smooth   ./ defl_alin) ./ pop_alin
Gh_real_pc   = (IRPF_smooth ./ defl_alin) ./ pop_alin

# ── Equacions d'observació: 100 × Δlog(X_pc_real) ────────────────────────────
Δlog(x) = 100.0 .* diff(log.(x))

obs_c   = Δlog(C_pc)
obs_inv = Δlog(inv_pc)
obs_Gc  = Δlog(Gc_real_pc)
obs_Gh  = Δlog(Gh_real_pc)
obs_wh  = Δlog(wh_real_pc)

# ── Guardar CSV (mateix format que simulated_data_orsi.csv) ───────────────────
# Tallem des de 1996-Q1: els 3 primers Δlog (1995-Q2, Q3, Q4) estan distorsionats
# perquè roll4 acumula menys de 4 trimestres al denominador (1, 2 i 3 quarters).
# El primer Δlog vàlid (ambdós extrems amb suma completa) és 1996-Q1.
df_out = DataFrame(
    date    = dates_alin[2:end],
    obs_c   = obs_c,
    obs_inv = obs_inv,
    obs_Gc  = obs_Gc,
    obs_Gh  = obs_Gh,
    obs_wh  = obs_wh
)
filter!(row -> row.date >= Date(1996, 1, 1), df_out)
T_obs = nrow(df_out)
CSV.write("data/orsi/real_data_orsi.csv", df_out)
println("Guardat: data/orsi/real_data_orsi.csv  ($T_obs observacions, " *
        string(df_out.date[1]) * " – " * string(df_out.date[end]) * ")")

# Nivells reals p.c. (per als gràfics de l'annex)
df_levels = DataFrame(
    date    = dates_alin,
    lev_c   = C_pc,
    lev_inv = inv_pc,
    lev_Gc  = Gc_real_pc,
    lev_Gh  = Gh_real_pc,
    lev_wh  = wh_real_pc,
)
filter!(row -> row.date >= Date(1996, 1, 1), df_levels)
CSV.write("data/orsi/real_data_orsi_levels.csv", df_levels)
println("Guardat: data/orsi/real_data_orsi_levels.csv  ($(nrow(df_levels)) observacions)")
