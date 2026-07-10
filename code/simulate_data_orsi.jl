# simulate_data_orsi.jl
# Simula T periodos del modelo Orsi-Raggi-Turino (2014) y guarda las 5
# variables observables en data/orsi/simulated_data_orsi.csv con el mismo
# formato que usará load_real_data_orsi.jl con datos españoles reales.
#
# Ejecutar desde la raíz del proyecto:
#   julia code/orsi/simulate_data_orsi.jl

using MacroModelling, DataFrames, CSV, Random, Plots

Random.seed!(2024)
mkpath("data/orsi")
mkpath("outputs/orsi")

include(joinpath(@__DIR__, "orsi.jl"))   # define OrsiTurinoModel + @parameters

# ── Simular ───────────────────────────────────────────────────────────────────
# Las ecuaciones de observación están definidas dentro del modelo (orsi.jl),
# por lo que simulate() ya las calcula directamente.
# Con T_sim = 400 se obtienen 400 observaciones trimestrales (~100 años).

# DGP: paràmetres del @parameters d'orsi.jl (beta=0.997, sigma=0.99, B_1=300, gamma_pct=0.20…)
# L'estimació simulada amb T=120 llegirà [1:120, :] d'aquest fitxer.
# L'estimació simulada amb T=400 llegirà totes les files.
T_sim = 400
sim   = simulate(OrsiTurinoModel, periods = T_sim)

obs_c   = vec(sim(:obs_c,   :, :simulate))
obs_inv = vec(sim(:obs_inv, :, :simulate))
obs_Gc  = vec(sim(:obs_Gc,  :, :simulate))
obs_Gh  = vec(sim(:obs_Gh,  :, :simulate))
obs_wh  = vec(sim(:obs_wh,  :, :simulate))

# ── Guardar CSV ───────────────────────────────────────────────────────────────
# Columnas: obs_c, obs_inv, obs_Gc, obs_Gh, obs_wh  [% Δlog trimestral]

df_sim = DataFrame(
    obs_c   = obs_c,
    obs_inv = obs_inv,
    obs_Gc  = obs_Gc,
    obs_Gh  = obs_Gh,
    obs_wh  = obs_wh
)
CSV.write("data/orsi/simulated_data_orsi.csv", df_sim)
println("Guardado: data/orsi/simulated_data_orsi.csv  ($(nrow(df_sim)) observaciones)")

# ── Gráfico de diagnóstico ────────────────────────────────────────────────────
labels = ["Δlog c" "Δlog inv" "Δlog Gc" "Δlog Gh" "Δlog wh"]
plts = [
    plot(df_sim[!, col], title = labels[i], legend = false, lw = 1.2, color = :steelblue)
    for (i, col) in enumerate(names(df_sim))
]
p_diag = plot(plts..., layout = (3, 2), size = (900, 700),
              suptitle = "Observables simulades (Orsi-Raggi-Turino)")
savefig(p_diag, "outputs/orsi/observables_simulats.png")
display(p_diag)
