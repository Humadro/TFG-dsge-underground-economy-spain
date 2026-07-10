using DataFrames, CSV, MacroModelling, Distributions, StatsPlots, Turing,
      AxisKeys, JLD2, MCMCChains, Random, Dates

println("Threads disponibles: ", Threads.nthreads())

Random.seed!(42)
mkpath("outputs/orsi")
run_tag = Dates.format(now(), "yyyymmdd_HH")
outdir  = "outputs/orsi/real_$run_tag"
mkpath(outdir)

include(joinpath(@__DIR__, "..", "orsi.jl"))

nombres_params = Symbol.(get_parameters(OrsiTurinoModel))
param_idx      = Dict(p => i for (i, p) in enumerate(nombres_params))

# ===========================================================================
# 1. Prioris per a l'estimació amb dades reals espanyoles
#    Font: Taula 1 d'Orsi, Raggi i Turino (2014), "Size, trend, and policy
#    implications of the underground economy", Review of Economic Dynamics 17.
#
#    Famílies originals del paper:
#      · Beta  → α, δk, αu, p, persistències ρᵢ
#      · Gamma → σ (CRRA), φ (elasticitat treball irregular), ξ (elasticitat total)
#      · InverseGamma → desviacions estàndard dels xocs (100σᵢ al paper)
#
#    Adaptacions per a Espanya:
#      · sigma i B_1 es CALIBREN (sigma=0.99, B_1=300) — no identificables
#        sense dades de tipus d'interès ni observacions de n_u.
#      · gamma_pct: Normal truncada centrada a 0.37% (taxa creixement Espanya
#        1995-2025), vs 0.23% (Itàlia 1982-2006) al paper.
#      · Persistències AR(1): truncades a upper=0.97 per estabilitat numèrica
#        (evita ρ→1 i singularitats Kalman/Blanchard-Kahn).
#
#    Prioris per a variàncies: el model usa var_eps=σ² però el paper prioritza
#    σ (desv. estàndard). Convertim: si σ~IG(mitja=0.006), aleshores var_eps~
#    IG(3, 7.2×10⁻⁵) que centra la mitja_var ≈ (0.006)² = 3.6×10⁻⁵.
# ===========================================================================

prioris_dict = Dict{Symbol, Any}(
    # ── Paràmetres estructurals — Taula 1 d'Orsi, Raggi i Turino (2014) ──────
    # sigma i B_1 es CALIBREN (no s'estimen):
    #   sigma — no identificable sense dades de tipus d'interès
    #   B_1   — co-identificat amb phi a través del sector sumergit latent n_u
    #
    # Beta(a,b): mean=a/(a+b), SE=sqrt(ab)/((a+b)*sqrt(a+b+1))
    # Per obtenir Beta(mean=μ, SE=s): a+b = μ(1-μ)/s²-1; a=μ(a+b); b=(1-μ)(a+b)
    :alpha    => Distributions.Beta(1478.1, 795.9),     # mitja=0.650, SE=0.010  [paper]
    :delta_k  => Distributions.Beta(6.07,  236.68),     # mitja=0.025, SE=0.010  [paper]
    :alpha_u  => Distributions.Beta(1480.2, 766.0),     # mitja=0.659, SE=0.010  [paper]

    # Gamma(α,θ): mean=αθ, SE=√α·θ → α=(mean/SE)², θ=mean/α
    :phi      => Distributions.Gamma(400.0, 0.0025),    # mitja=1.000, SE=0.050  [paper]
    :xi       => Distributions.Gamma(25.0,  0.04),      # mitja=1.000, SE=0.200  [paper]

    # gamma_pct: taxa de creixement trimestral (%) — no al paper (Itàlia); adaptat Espanya
    :gamma_pct => Distributions.truncated(Distributions.Normal(0.37, 0.20); lower = 0.0),

    # ── Probabilitat d'inspecció (ss) ────────────────────────────────────────
    :p_ss     => Distributions.Beta(8.70, 281.30),      # mitja=0.030, SE=0.010  [paper]

    # ── Persistències AR(1) — Beta(mean, SE=0.10) del paper, truncades a 0.97 ─
    # Paper (Taula 1): tots els ρ ~ Beta, SE=0.100.
    # Mantenim upper=0.97 per estabilitat numèrica (evita zona unit-root).
    # Fórmula: a+b=μ(1-μ)/0.01-1; a=μ(a+b); b=(1-μ)(a+b)
    :rho_a    => Distributions.truncated(Distributions.Beta(12.0, 3.0);   upper = 0.97),  # mitja=0.800 [paper]
    :rho_b    => Distributions.truncated(Distributions.Beta(12.0, 12.0);  upper = 0.97),  # mitja=0.500 [paper]
    :rho_c    => Distributions.truncated(Distributions.Beta(7.2,  0.8);   upper = 0.97),  # mitja≈0.900 [paper]
    :rho_s    => Distributions.truncated(Distributions.Beta(7.2,  0.8);   upper = 0.97),  # mitja≈0.900 [paper]
    :rho_h    => Distributions.truncated(Distributions.Beta(7.2,  0.8);   upper = 0.97),  # mitja≈0.900 [paper]
    :rho_xi_h => Distributions.truncated(Distributions.Beta(13.8, 9.2);   upper = 0.97),  # mitja=0.600 [paper]
    :rho_x    => Distributions.truncated(Distributions.Beta(12.0, 3.0);   upper = 0.95),  # mitja=0.800 [paper]; cap 0.95: sim mostra rho_x→1 sense restricció
    :rho_p    => Distributions.truncated(Distributions.Beta(12.0, 3.0);   upper = 0.97),  # mitja=0.800 [paper]

    # ── Variàncies dels xocs (InverseGamma) ──────────────────────────────────
    # Paper: 100σᵢ ~ IG(mitja=0.600, SE=0.160) → σᵢ ~ IG(mitja=0.006, SE=0.0016)
    # El model usa var_eps = σ², E[σ²] ≈ (0.006)² = 3.6×10⁻⁵.
    # InverseGamma(3, 7.2×10⁻⁵): mitja_var=3.6×10⁻⁵ (≡ σ≈0.6%), cua dreta per σ>1%.
    :var_eps_a    => Distributions.InverseGamma(3.0, 7.2e-5),   # σ̄=0.60%  [paper]
    :var_eps_b    => Distributions.InverseGamma(3.0, 7.2e-5),   # σ̄=0.60%  [paper]
    :var_eps_c    => Distributions.InverseGamma(3.0, 7.2e-5),   # σ̄=0.60%  [paper]
    :var_eps_s    => Distributions.InverseGamma(3.0, 7.2e-5),   # σ̄=0.60%  [paper]
    :var_eps_h    => Distributions.InverseGamma(3.0, 7.2e-5),   # σ̄=0.60%  [paper]
    :var_eps_xi_h => Distributions.InverseGamma(3.0, 7.2e-5),   # σ̄=0.60%  [paper]
    :var_eps_x    => Distributions.InverseGamma(3.0, 7.2e-5),   # σ̄=0.60%  [paper]
    :var_eps_p    => Distributions.InverseGamma(3.0, 7.2e-5),   # σ̄=0.60%  [paper]
)

# ===========================================================================
# 2. Motor Turing
# ===========================================================================

Turing.@model function estimacio_orsi(data_k, model_dsge, param_names, prioris)

    alpha     ~ prioris[:alpha]
    delta_k   ~ prioris[:delta_k]
    alpha_u   ~ prioris[:alpha_u]
    phi       ~ prioris[:phi]
    xi        ~ prioris[:xi]
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
    p_vector[param_idx[:sigma]]        = T(0.99)     # calibrat: no identificable sense dades financeres
    p_vector[param_idx[:phi]]          = phi
    p_vector[param_idx[:xi]]           = xi
    p_vector[param_idx[:B_1]]          = T(300.0)    # calibrat: co-identificat amb phi via n_u latent
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
    p_vector[param_idx[:beta]]         = T(0.997)
    p_vector[param_idx[:B_0]]          = T(81.74)   # calibrat: bisecció per n_m_ss+n_u_ss≈0.19 (Espanya)
    p_vector[param_idx[:gamma_ss]]     = T(1.00)
    p_vector[param_idx[:tau_c_ss]]     = T(0.25)
    p_vector[param_idx[:tau_h_ss]]     = T(0.23)
    p_vector[param_idx[:tau_s_ss]]     = T(0.23)
    p_vector[param_idx[:s]]            = T(1.70)

    Turing.@addlogprob! get_loglikelihood(model_dsge, data_k, p_vector)
end

# ===========================================================================
# 3. Funció de gràfics priori-posteriori (sense línia de "valor vertader")
# ===========================================================================

function plot_priori_posteriori_chains(cadena, prioris, params_plot, titol, ncols = 4)
    nrows = ceil(Int, length(params_plot) / ncols)
    grafics = []
    for param in params_plot
        p_plot = plot(title = string(param), legend = false, ylabel = "Densitat")
        data_param = Array(cadena[param])
        for c in 1:size(data_param, 2)
            density!(p_plot, data_param[:, c], lw = 2, alpha = 0.7)
        end
        plot!(p_plot, prioris[param], color = :black, lw = 2, linestyle = :dash)
        push!(grafics, p_plot)
    end
    return plot(grafics..., layout = (nrows, ncols),
                size = (ncols * 280, nrows * 220),
                plot_title = titol, margin = 4Plots.mm)
end

# ===========================================================================
# 4. Carregar dades reals espanyoles
#    Generat per code/orsi/load_real_data_orsi.jl
# ===========================================================================

datos_real = CSV.read("data/orsi/real_data_orsi.csv", DataFrame)
#filter!(row -> row.date < Date(2020, 1, 1), datos_real)   # tallem abans del COVID
T = nrow(datos_real)
obs_names = [:obs_c, :obs_inv, :obs_Gc, :obs_Gh, :obs_wh]
matriz_real = Matrix(datos_real[!, obs_names])'
datos_real_keyed = KeyedArray(matriz_real, (obs_names, 1:T))

println("Dades carregades: $T trimestres, $(datos_real.date[1]) – $(datos_real.date[end])")

# ===========================================================================
# 5. Estimació bayesiana — 3 cadenes × 2000 mostres
#    Sampler: NUTS(1000 adaptació, target_accept=0.65, max_depth=7)
#    Igual que l'estimació simulada per fer els resultats comparables.
# ===========================================================================

modelo_real = estimacio_orsi(datos_real_keyed, OrsiTurinoModel, nombres_params, prioris_dict)

n_muestras   = 3000
n_cadenas    = 3
n_adaptacion = 1000
sampler      = NUTS(n_adaptacion, 0.65; max_depth = 5, adtype = AutoForwardDiff(chunksize = 8))
# target_accept=0.80 (vs 0.65): passos més curts, menys rebuigs en zones de curvatura alta
# max_depth=8 (vs 7): permet trajectòries lleugerament més llargues per persistències altes

# ── Punts d'inici centrats a les mitges de les prioris del paper (Taula 1) ───
# Cadena 1 = mitges de les prioris d'Orsi et al. (2014), excepte gamma_pct (Espanya).
# var_eps: mitja de IG(3, 7.2e-5) = 7.2e-5/2 = 3.6e-5 (≡ σ≈0.6%).
# Cadenes 2 i 3 = pertorbació mínima ±0.001, clamped a [0.05, 0.96].
init_paper = [
    0.650,    # alpha       (1) — mitja priori paper
    0.025,    # delta_k     (2) — mitja priori paper
    0.659,    # alpha_u     (3) — mitja priori paper
    1.000,    # phi         (4) — mitja priori paper (Gamma)
    1.000,    # xi          (5) — mitja priori paper (Gamma)
    0.27,     # gamma_pct   (6) — Espanya (no al paper)
    0.030,    # p_ss        (7) — mitja priori paper
    0.800,    # rho_a       (8) — mitja priori paper
    0.500,    # rho_b       (9) — mitja priori paper
    0.900,    # rho_c       (10) — mitja priori paper
    0.900,    # rho_s       (11) — mitja priori paper
    0.900,    # rho_h       (12) — mitja priori paper
    0.600,    # rho_xi_h    (13) — mitja priori paper
    0.800,    # rho_x       (14) — mitja priori paper
    0.800,    # rho_p       (15) — mitja priori paper
    3.6e-5,   # var_eps_a   (16) — mitja priori paper: IG(3,7.2e-5) → E=3.6e-5
    3.6e-5,   # var_eps_b   (17)
    3.6e-5,   # var_eps_c   (18)
    3.6e-5,   # var_eps_s   (19)
    3.6e-5,   # var_eps_h   (20)
    3.6e-5,   # var_eps_xi_h (21)
    3.6e-5,   # var_eps_x   (22)
    3.6e-5,   # var_eps_p   (23)
]

function perturb_paper(base, Δ_struct, Δ_rho, Δ_var)
    v = copy(base)
    for i in 1:7;  v[i] *= (1.0 + Δ_struct); end
    for i in 8:15; v[i] = clamp(v[i] + Δ_rho, 0.05, 0.96); end
    for i in 16:23; v[i] *= (1.0 + Δ_var); end
    return v
end

init_chain2 = perturb_paper(init_paper, +0.001, +0.001, +0.001)
init_chain3 = perturb_paper(init_paper, -0.001, -0.001, -0.001)

println("Iniciant mostreig NUTS — $(Dates.format(now(), "HH:MM:SS"))…")
chain_real = sample(modelo_real, sampler, MCMCThreads(), n_muestras, n_cadenas;
                    initial_params = [init_paper, init_chain2, init_chain3])
println("Mostreig completat  — $(Dates.format(now(), "HH:MM:SS"))")
display(chain_real)

# ===========================================================================
# 6. Guardar cadena i resum
# ===========================================================================

@save joinpath(outdir, "cadenes_orsi_real.jld2") chain_real
open(joinpath(outdir, "describe_orsi_real.txt"), "w") do io
    show(io, MIME("text/plain"), chain_real)
end
println("Cadena guardada: $outdir/cadenes_orsi_real.jld2")
println("Resum guardat:   $outdir/describe_orsi_real.txt")

@load joinpath(outdir, "cadenes_orsi_real.jld2") chain_real

# ===========================================================================
# 7. Gràfics
# ===========================================================================

params_estructurals  = [:alpha, :delta_k, :alpha_u, :phi, :xi, :gamma_pct, :p_ss]  # sigma i B_1 calibrats
params_persistencies = [:rho_a, :rho_b, :rho_c, :rho_s, :rho_h, :rho_xi_h, :rho_x, :rho_p]
params_shocks        = [:var_eps_a, :var_eps_b, :var_eps_c, :var_eps_s,
                        :var_eps_h, :var_eps_xi_h, :var_eps_x, :var_eps_p]

# ── 7.1 Priori vs Posteriori ─────────────────────────────────────────────────

p_distr_est = plot_priori_posteriori_chains(chain_real, prioris_dict,
    params_estructurals, "Priori vs Posteriori — Estructurals", 4)
savefig(p_distr_est, joinpath(outdir, "distr_estructurals_real.png"))
display(p_distr_est)

p_distr_per = plot_priori_posteriori_chains(chain_real, prioris_dict,
    params_persistencies, "Priori vs Posteriori — Persistències AR(1)", 4)
savefig(p_distr_per, joinpath(outdir, "distr_persistencies_real.png"))
display(p_distr_per)

p_distr_shk = plot_priori_posteriori_chains(chain_real, prioris_dict,
    params_shocks, "Priori vs Posteriori — Variàncies Shocks", 4)
savefig(p_distr_shk, joinpath(outdir, "distr_shocks_real.png"))
display(p_distr_shk)

# ── 7.2 Traceplots ────────────────────────────────────────────────────────────

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

p_trace_est = traceplot_group(chain_real, params_estructurals, "Traceplots — Estructurals")
savefig(p_trace_est, joinpath(outdir, "trace_estructurals_real.png"))
display(p_trace_est)

p_trace_per = traceplot_group(chain_real, params_persistencies, "Traceplots — Persistències")
savefig(p_trace_per, joinpath(outdir, "trace_persistencies_real.png"))
display(p_trace_per)

p_trace_shk = traceplot_group(chain_real, params_shocks, "Traceplots — Shocks")
savefig(p_trace_shk, joinpath(outdir, "trace_shocks_real.png"))
display(p_trace_shk)

# ── 7.3 Corner ────────────────────────────────────────────────────────────────

p_corner_est = corner(chain_real[params_estructurals], size = (1200, 1200), plot_density = false)
savefig(p_corner_est, joinpath(outdir, "corner_estructurals_real.png"))
display(p_corner_est)

p_corner_per = corner(chain_real[params_persistencies], size = (1200, 1200), plot_density = false)
savefig(p_corner_per, joinpath(outdir, "corner_persistencies_real.png"))
display(p_corner_per)

p_corner_shk = corner(chain_real[params_shocks], size = (1200, 1200), plot_density = false)
savefig(p_corner_shk, joinpath(outdir, "corner_shocks_real.png"))
display(p_corner_shk)

println("\nTots els gràfics guardats a $outdir/")
