using DataFrames, CSV, MacroModelling, Distributions, StatsPlots, Turing,
      AxisKeys, JLD2, MCMCChains, Random

# Comprova que hi ha prou threads (ha de ser ≥ n_cadenas)
println("Threads disponibles: ", Threads.nthreads())

Random.seed!(42)
mkpath("outputs/orsi")

include(joinpath(@__DIR__, "..", "orsi.jl"))   # defineix OrsiTurinoModel

nombres_params = Symbol.(get_parameters(OrsiTurinoModel))
param_idx      = Dict(p => i for (i, p) in enumerate(nombres_params))

# ===========================================================================
# 1. Paràmetres vertaders (copia els valors que has posat a simulate_data_orsi.jl)
#    S'usaran per dibuixar línies verticals als gràfics priori-posteriori.
# ===========================================================================

true_params = Dict{Symbol, Float64}(
    :alpha       => 0.63,
    :delta_k     => 0.03,
    :alpha_u     => 0.66,
    :sigma       => 0.99,
    :phi         => 0.93,
    :xi          => 1.60,
    :B_1         => 300.0,
    :gamma_pct   => 0.20,
    :p_ss        => 0.02,
    :rho_a       => 0.99,
    :rho_b       => 0.93,
    :rho_c       => 0.96,
    :rho_s       => 0.94,
    :rho_h       => 0.99,
    :rho_xi_h    => 0.60,
    :rho_x       => 0.94,
    :rho_p       => 0.95,
    :var_eps_a    => 0.01^2,
    :var_eps_b    => 0.01^2,
    :var_eps_c    => 0.01^2,
    :var_eps_s    => 0.01^2,
    :var_eps_h    => 0.02^2,
    :var_eps_xi_h => 0.01^2,
    :var_eps_x    => 0.01^2,
    :var_eps_p    => 0.06^2,
)

# ===========================================================================
# 2. Prioris
#    Estructurals: Uniform amb rangs plausibles
#    Persistències ρ: Beta(1,1) = Uniform(0,1), mateixa per a totes
#    Variàncies shocks σ²: InverseGamma(2, 0.01), mateixa per a totes
# ===========================================================================

prioris_dict = Dict{Symbol, Distributions.Distribution}(
    # ── Paràmetres estructurals ──────────────────────────────────────────────
    :alpha       => Distributions.Uniform(0.40, 0.85),    # vertader: 0.63
    :delta_k     => Distributions.Uniform(0.005, 0.10),   # vertader: 0.03
    :alpha_u     => Distributions.Uniform(0.40, 0.85),    # vertader: 0.66
    :sigma       => Distributions.Uniform(0.10, 5.00),    # vertader: 0.99
    :phi         => Distributions.Uniform(0.10, 3.00),    # vertader: 0.93
    :xi          => Distributions.Uniform(0.30, 4.00),    # vertader: 1.60
    :B_1         => Distributions.Uniform(10.0, 2000.0),  # vertader: 300.0
    :gamma_pct   => Distributions.Uniform(0.02, 0.80),    # vertader: 0.20

    # ── Persistències AR(1) — mateixa priori plana per a tots ────────────────
    # Beta(1,1) = Uniform(0,1): completament no informativa sobre [0,1]
    :rho_a       => Distributions.Beta(1.0, 1.0),         # vertader: 0.99
    :rho_b       => Distributions.Beta(1.0, 1.0),         # vertader: 0.93
    :rho_c       => Distributions.Beta(1.0, 1.0),         # vertader: 0.96
    :rho_s       => Distributions.Beta(1.0, 1.0),         # vertader: 0.94
    :rho_h       => Distributions.Beta(1.0, 1.0),         # vertader: 0.99
    :rho_xi_h    => Distributions.Beta(1.0, 1.0),         # vertader: 0.60
    :rho_x       => Distributions.Beta(1.0, 1.0),         # vertader: 0.94
    :rho_p       => Distributions.Beta(1.0, 1.0),         # vertader: 0.95

    # ── Probabilitat d'inspecció (ss) ────────────────────────────────────────
    :p_ss        => Distributions.Uniform(0.001, 0.1),        # vertader: 0.02

    # ── Desviacions estàndard dels shocks — mateixa priori IG per a totes ────
    # σ_i ~ InverseGamma(2, 0.01): mitjana=0.01, cua pesada per valors grans
    # (equivalent a σ_i^2 ~ IG amb mode a σ≈0.003, massa per a σ∈[0.001,0.15])
    :var_eps_a    => Distributions.InverseGamma(2, 0.01),
    :var_eps_b    => Distributions.InverseGamma(2, 0.01),
    :var_eps_c    => Distributions.InverseGamma(2, 0.01),
    :var_eps_s    => Distributions.InverseGamma(2, 0.01),
    :var_eps_h    => Distributions.InverseGamma(2, 0.01),
    :var_eps_xi_h => Distributions.InverseGamma(2, 0.01),
    :var_eps_x    => Distributions.InverseGamma(2, 0.01),
    :var_eps_p    => Distributions.InverseGamma(2, 0.01)
)

# ===========================================================================
# 3. Motor Turing
# ===========================================================================

Turing.@model function estimacio_orsi(data_k, model_dsge, param_names, prioris)

    alpha     ~ prioris[:alpha]
    delta_k   ~ prioris[:delta_k]
    alpha_u   ~ prioris[:alpha_u]
    sigma     ~ prioris[:sigma]
    phi       ~ prioris[:phi]
    xi        ~ prioris[:xi]
    B_1       ~ prioris[:B_1]
    gamma_pct ~ prioris[:gamma_pct]
    p_ss      ~ prioris[:p_ss]

    rho_a    ~ prioris[:rho_a]
    rho_b    ~ prioris[:rho_b]
    rho_c    ~ prioris[:rho_c]
    rho_s    ~ prioris[:rho_s]
    rho_h    ~ prioris[:rho_h]
    rho_xi_h ~ prioris[:rho_xi_h]
    rho_x    ~ prioris[:rho_x]
    rho_p    ~ prioris[:rho_p]

    var_eps_a    ~ prioris[:var_eps_a]
    var_eps_b    ~ prioris[:var_eps_b]
    var_eps_c    ~ prioris[:var_eps_c]
    var_eps_s    ~ prioris[:var_eps_s]
    var_eps_h    ~ prioris[:var_eps_h]
    var_eps_xi_h ~ prioris[:var_eps_xi_h]
    var_eps_x    ~ prioris[:var_eps_x]
    var_eps_p    ~ prioris[:var_eps_p]

    T        = typeof(alpha)
    p_vector = Vector{T}(undef, length(param_names))

    p_vector[param_idx[:alpha]]        = alpha
    p_vector[param_idx[:delta_k]]      = delta_k
    p_vector[param_idx[:alpha_u]]      = alpha_u
    p_vector[param_idx[:sigma]]        = sigma
    p_vector[param_idx[:phi]]          = phi
    p_vector[param_idx[:xi]]           = xi
    p_vector[param_idx[:B_1]]          = B_1
    p_vector[param_idx[:gamma_pct]]    = gamma_pct
    p_vector[param_idx[:p_ss]]         = p_ss
    p_vector[param_idx[:rho_a]]        = rho_a
    p_vector[param_idx[:rho_b]]        = rho_b
    p_vector[param_idx[:rho_c]]        = rho_c
    p_vector[param_idx[:rho_s]]        = rho_s
    p_vector[param_idx[:rho_h]]        = rho_h
    p_vector[param_idx[:rho_xi_h]]     = rho_xi_h
    p_vector[param_idx[:rho_x]]        = rho_x
    p_vector[param_idx[:rho_p]]        = rho_p
    p_vector[param_idx[:var_eps_a]]    = var_eps_a
    p_vector[param_idx[:var_eps_b]]    = var_eps_b
    p_vector[param_idx[:var_eps_c]]    = var_eps_c
    p_vector[param_idx[:var_eps_s]]    = var_eps_s
    p_vector[param_idx[:var_eps_h]]    = var_eps_h
    p_vector[param_idx[:var_eps_xi_h]] = var_eps_xi_h
    p_vector[param_idx[:var_eps_x]]    = var_eps_x
    p_vector[param_idx[:var_eps_p]]    = var_eps_p
    p_vector[param_idx[:beta]]         = T(0.997)   # bug fix: ha de coincidir amb el DGP (beta del @parameters)
    p_vector[param_idx[:B_0]]          = T(81.74)
    p_vector[param_idx[:gamma_ss]]     = T(1.00)
    p_vector[param_idx[:tau_c_ss]]     = T(0.40)
    p_vector[param_idx[:tau_h_ss]]     = T(0.35)
    p_vector[param_idx[:tau_s_ss]]     = T(0.20)
    p_vector[param_idx[:s]]            = T(1.70)

    Turing.@addlogprob! get_loglikelihood(model_dsge, data_k, p_vector)
end

# ===========================================================================
# 4. Funció de gràfics priori-posteriori amb línia del valor vertader
# ===========================================================================

function plot_priori_posteriori_chains(cadena, prioris, true_vals, params_plot, titol, ncols=4)
    nrows = ceil(Int, length(params_plot) / ncols)
    grafics = []
    for param in params_plot
        p_plot = plot(title = string(param), legend = false, ylabel = "Densitat")
        data_param = Array(cadena[param])
        for c in 1:size(data_param, 2)
            density!(p_plot, data_param[:, c], lw = 2, alpha = 0.7)
        end
        # Prior (línia negra discontínua)
        plot!(p_plot, prioris[param], color = :black, lw = 2, linestyle = :dash)
        # Valor vertader (línia vermella vertical)
        if haskey(true_vals, param)
            vline!(p_plot, [true_vals[param]], color = :red, lw = 2, linestyle = :solid)
        end
        push!(grafics, p_plot)
    end
    return plot(grafics..., layout = (nrows, ncols),
                size = (ncols * 280, nrows * 220),
                plot_title = titol, margin = 4Plots.mm)
end

# ===========================================================================
# 5. Carregar dades simulades
# ===========================================================================

T_sim     = 120   # ← canviar a 400 per a l'exercici T=400
datos_sim = CSV.read("data/orsi/simulated_data_orsi.csv", DataFrame)[1:T_sim, :]
T = nrow(datos_sim)
obs_names = [:obs_c, :obs_inv, :obs_Gc, :obs_Gh, :obs_wh]
matriz_sim = Matrix(datos_sim[!, obs_names])'
datos_sim_keyed = KeyedArray(matriz_sim, (obs_names, 1:T))

# ===========================================================================
# 6. Estimació bayesiana — 3 cadenes × 2000 mostres
# ===========================================================================

modelo_sim = estimacio_orsi(datos_sim_keyed, OrsiTurinoModel, nombres_params, prioris_dict)

n_muestras   = 2000
n_cadenas    = 3
n_adaptacion = 1000
sampler      = NUTS(n_adaptacion, 0.65; max_depth = 5, adtype = AutoForwardDiff(chunksize = 8))

chain_sim = sample(modelo_sim, sampler, MCMCThreads(), n_muestras, n_cadenas)
display(chain_sim)

# ===========================================================================
# 7. Guardar cadena i resum
# ===========================================================================

@save "outputs/orsi/cadenes_orsi_sim_T$(T_sim).jld2" chain_sim
open("outputs/orsi/describe_orsi_sim_T$(T_sim).txt", "w") do io
    show(io, MIME("text/plain"), chain_sim)
end
println("Cadena guardada: outputs/orsi/cadenes_orsi_sim_T$(T_sim).jld2")
println("Resum guardat:   outputs/orsi/describe_orsi_sim_T$(T_sim).txt")

@load "outputs/orsi/cadenes_orsi_sim_T$(T_sim).jld2" chain_sim

# ===========================================================================
# 8. Gràfics
# ===========================================================================

params_estructurals  = [:alpha, :delta_k, :alpha_u, :sigma, :phi, :xi, :B_1, :gamma_pct, :p_ss]
params_persistencies = [:rho_a, :rho_b, :rho_c, :rho_s, :rho_h, :rho_xi_h, :rho_x, :rho_p]
params_shocks        = [:var_eps_a, :var_eps_b, :var_eps_c, :var_eps_s,
                        :var_eps_h, :var_eps_xi_h, :var_eps_x, :var_eps_p]

# ── 8.1 Priori vs Posteriori (amb línia roja = valor vertader) ───────────────

p_distr_est = plot_priori_posteriori_chains(chain_sim, prioris_dict, true_params,
    params_estructurals, "Priori vs Posteriori — Estructurals", 4)
savefig(p_distr_est, "outputs/orsi/distr_estructurals_sim.png")
display(p_distr_est)

p_distr_per = plot_priori_posteriori_chains(chain_sim, prioris_dict, true_params,
    params_persistencies, "Priori vs Posteriori — Persistències AR(1)", 4)
savefig(p_distr_per, "outputs/orsi/distr_persistencies_sim.png")
display(p_distr_per)

p_distr_shk = plot_priori_posteriori_chains(chain_sim, prioris_dict, true_params,
    params_shocks, "Priori vs Posteriori — Desv. Std. Shocks", 4)
savefig(p_distr_shk, "outputs/orsi/distr_shocks_sim.png")
display(p_distr_shk)

# ── 8.2 Traceplots ────────────────────────────────────────────────────────────

function traceplot_group(cadena, params, titol, ncols = 4)
    nrows = ceil(Int, length(params) / ncols)
    plts = [
        plot(Array(cadena[p]), label = false, title = string(p),
             titlefontsize = 9, linewidth = 0.5, alpha = 0.7)
        for p in params
    ]
    return plot(plts..., layout = (nrows, ncols),
                size = (ncols * 280, nrows * 180),
                plot_title = titol, margin = 3Plots.mm)
end

p_trace_est = traceplot_group(chain_sim, params_estructurals, "Traceplots — Estructurals")
savefig(p_trace_est, "outputs/orsi/trace_estructurals_sim.png")
display(p_trace_est)

p_trace_per = traceplot_group(chain_sim, params_persistencies, "Traceplots — Persistències")
savefig(p_trace_per, "outputs/orsi/trace_persistencies_sim.png")
display(p_trace_per)

p_trace_shk = traceplot_group(chain_sim, params_shocks, "Traceplots — Shocks")
savefig(p_trace_shk, "outputs/orsi/trace_shocks_sim.png")
display(p_trace_shk)

# ── 8.3 Corner ────────────────────────────────────────────────────────────────

p_corner_est = corner(chain_sim[params_estructurals], size = (1200, 1200), plot_density = false)
savefig(p_corner_est, "outputs/orsi/corner_estructurals_sim.png")
display(p_corner_est)

p_corner_per = corner(chain_sim[params_persistencies], size = (1200, 1200), plot_density = false)
savefig(p_corner_per, "outputs/orsi/corner_persistencies_sim.png")
display(p_corner_per)

p_corner_shk = corner(chain_sim[params_shocks], size = (1200, 1200), plot_density = false)
savefig(p_corner_shk, "outputs/orsi/corner_shocks_sim.png")
display(p_corner_shk)

println("\nTots els gràfics guardats a outputs/orsi/")
