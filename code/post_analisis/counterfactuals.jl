##############################################################################
#  counterfactuals.jl
#  Anàlisi contrafactual d'estat estacionari — Orsi, Raggi & Turino (2014)
#
#  Contrafactuals:
#    CF1 — Reducció de l'IS en 5 pp (tau_c: 0.25 → 0.20)
#    CF2 — Pujada de l'IS en 5 pp   (tau_c: 0.25 → 0.30)
#    CF3 — CF1 + reforç inspector   (tau_c: 0.20 + p: 0.026 → 0.039)
#
#  Execució: julia code/orsi/post_analisis/counterfactuals.jl
#  Resultat: taula impresa + outputs/orsi/counterfactuals/counterfactuals.csv
##############################################################################

using MacroModelling
using DataFrames, CSV, Printf

include(joinpath(@__DIR__, "..", "orsi.jl"))
include(joinpath(@__DIR__, "post_analisis.jl"))

OUTPUT_DIR = "outputs/orsi/counterfactuals"
mkpath(OUTPUT_DIR)

##############################################################################
#  Definició dels escenaris
##############################################################################

baseline = estimated_params   # ja definit a post_analisis.jl

cf1 = merge(baseline, Dict{Symbol,Float64}(
    :tau_c_ss => 0.20,        # IS −5 pp
))

cf2 = merge(baseline, Dict{Symbol,Float64}(
    :tau_c_ss => 0.30,        # IS +5 pp
))

cf3 = merge(baseline, Dict{Symbol,Float64}(
    :tau_c_ss => 0.20,        # IS −5 pp
    :p_ss     => 0.039,       # inspecció ×1.5 (~0.026 × 1.5)
))

cf4 = merge(baseline, Dict{Symbol,Float64}(
    :tau_c_ss => 0.30,        # IS +5 pp
    :tau_s_ss => 0.18,        # cotitzacions −5 pp
))

scenarios = [
    ("Baseline",                      baseline),
    ("CF1: IS −5pp",                  cf1),
    ("CF2: IS +5pp",                  cf2),
    ("CF3: IS −5pp + Inspecció ×1.5", cf3),
    ("CF4: IS +5pp + Cot. −5pp",      cf4),
]

##############################################################################
#  Variables de l'estat estacionari a comparar
##############################################################################

SS_VARS = [
    :y, :y_m, :y_u, :underground_share,
    :c, :inv,
    :G_c, :G_h, :G_s,
    :TE,
]

VAR_PRINT = Dict(
    :y                 => "PIB total (y)",
    :y_m               => "Output formal (y_m)",
    :y_u               => "Output informal (y_u)",
    :underground_share => "Quota submergida (y_u/y)",
    :c                 => "Consum (c)",
    :inv               => "Inversió (inv)",
    :G_c               => "Recaptació IS (G_c)",
    :G_h               => "Recaptació IRPF (G_h)",
    :G_s               => "Cotitzacions (G_s)",
    :TE                => "Evasió fiscal (TE)",
)

##############################################################################
#  Càlcul
##############################################################################

println("\n" * "="^70)
println("  Anàlisi contrafactual — Orsi, Raggi & Turino (2014)")
println("="^70)

# Steady states
ss_results = Dict{String, Dict{Symbol,Float64}}()
for (name, params) in scenarios
    print("  Calculant estat estacionari: $name ... ")
    ss_results[name] = compute_steady_state(OrsiTurinoModel, params)
    println("OK")
end

# Afegim recaptació total G = G_c + G_h + G_s per a cada escenari
for (name, ss) in ss_results
    ss[:G_total] = get(ss, :G_c, 0.0) + get(ss, :G_h, 0.0) + get(ss, :G_s, 0.0)
end
push!(SS_VARS, :G_total)
VAR_PRINT[:G_total] = "Recaptació total (G)"

# Welfare i ECV
println()
welfare_baseline = compute_welfare(OrsiTurinoModel, baseline)
welfare_cf1      = compute_welfare(OrsiTurinoModel, cf1; alt_params = baseline)
welfare_cf2      = compute_welfare(OrsiTurinoModel, cf2; alt_params = baseline)
welfare_cf3      = compute_welfare(OrsiTurinoModel, cf3; alt_params = baseline)
welfare_cf4      = compute_welfare(OrsiTurinoModel, cf4; alt_params = baseline)

ecv = Dict(
    "Baseline"                      => 0.0,
    "CF1: IS −5pp"                  => welfare_cf1[:ecv],
    "CF2: IS +5pp"                  => welfare_cf2[:ecv],
    "CF3: IS −5pp + Inspecció ×1.5" => welfare_cf3[:ecv],
    "CF4: IS +5pp + Cot. −5pp"      => welfare_cf4[:ecv],
)

##############################################################################
#  Taula comparativa
##############################################################################

scenario_names = [s[1] for s in scenarios]
baseline_ss    = ss_results["Baseline"]

println("\n" * "="^70)
println("  Resultats d'estat estacionari (canvi % respecte baseline)")
println("="^70)

cf1_key = "CF1: IS −5pp"
cf2_key = "CF2: IS +5pp"
cf3_key = "CF3: IS −5pp + Inspecció ×1.5"
cf4_key = "CF4: IS +5pp + Cot. −5pp"

@printf("%-32s  %10s  %10s  %10s  %10s\n", "Variable", "CF1 Δ%", "CF2 Δ%", "CF3 Δ%", "CF4 Δ%")
println("-"^78)

all_vars = vcat(SS_VARS)
for var in all_vars
    label    = get(VAR_PRINT, var, string(var))
    base_val = get(baseline_ss, var, NaN)
    pct(key) = isnan(base_val) || base_val == 0.0 ? NaN :
               (get(ss_results[key], var, NaN) - base_val) / abs(base_val) * 100
    @printf("%-32s  %+9.3f%%  %+9.3f%%  %+9.3f%%  %+9.3f%%\n",
        label, pct(cf1_key), pct(cf2_key), pct(cf3_key), pct(cf4_key))
end

println("-"^78)
@printf("%-32s  %+9.3f%%  %+9.3f%%  %+9.3f%%  %+9.3f%%\n",
    "ECV (benestar)",
    ecv[cf1_key]*100, ecv[cf2_key]*100, ecv[cf3_key]*100, ecv[cf4_key]*100)
println("="^78)

##############################################################################
#  Exportació CSV
##############################################################################

rows = []
for var in all_vars
    label    = get(VAR_PRINT, var, string(var))
    base_val = get(baseline_ss, var, NaN)
    pct(key) = isnan(base_val) || base_val == 0.0 ? NaN :
               round((get(ss_results[key], var, NaN) - base_val) / abs(base_val) * 100, digits=3)
    push!(rows, (
        variable = string(var),
        label    = label,
        cf1_pct  = pct(cf1_key),
        cf2_pct  = pct(cf2_key),
        cf3_pct  = pct(cf3_key),
        cf4_pct  = pct(cf4_key),
    ))
end

push!(rows, (
    variable = "ecv",
    label    = "ECV benestar (%)",
    cf1_pct  = round(ecv[cf1_key] * 100, digits=3),
    cf2_pct  = round(ecv[cf2_key] * 100, digits=3),
    cf3_pct  = round(ecv[cf3_key] * 100, digits=3),
    cf4_pct  = round(ecv[cf4_key] * 100, digits=3),
))

df_out = DataFrame(rows)
csv_path = joinpath(OUTPUT_DIR, "counterfactuals.csv")
CSV.write(csv_path, df_out)
println("\n  Taula guardada a: $csv_path")
println("="^70 * "\n")
