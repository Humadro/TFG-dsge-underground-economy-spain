# Necessites afegir els paquets nous:
# using Pkg; Pkg.add(["Lux", "Optimisers"])

using Lux, Zygote, Optimisers
using LinearAlgebra, Optim, Random
using CSV, DataFrames, MacroModelling

Random.seed!(42)
const rng = Xoshiro(42)

# =====================================================================
# 1. MODEL RBC AMB MacroModelling → Estat Estacionari
#    Mantenim MacroModelling per definir el model i obtenir els valors
#    d'estat estacionari que necessitem per normalitzar les simulacions DL.
# =====================================================================
MacroModelling.@model RBC begin
    1/c[0] = beta * (r[1] + 1 - delta) / c[1]          # eq-euler
    gamma * c[0] / (1 - n[0]) = w[0]                    # eq-labor (intratemporal)
    w[0] = (1 - alpha) * y[0] / n[0]                    # preu del treball
    r[0] = alpha * y[0] / k[-1]                         # preu del capital
    y[0] = A[0] * k[-1]^alpha * n[0]^(1 - alpha)       # eq-production (Cobb-Douglas)
    y[0] = c[0] + i[0]                                  # eq-resources
    k[0] = (1 - delta) * k[-1] + i[0]                  # eq-capital
    log(A[0]) = rho * log(A[-1]) + epsilon[x]           # eq-shock (AR1 en logs)
end

MacroModelling.@parameters RBC begin
    beta    = 0.99
    alpha   = 0.33
    delta   = 0.025
    gamma   = 2.0
    rho     = 0.95
    std_epsilon = 0.01
end

# Estat estacionari: necessari per (a) mostreig ergòdic i (b) normalitzar simulacions
const ss   = steady_state(RBC)
const k_ss = Float32(ss[:k])
const y_ss = Float32(ss[:y])
const c_ss = Float32(ss[:c])
const i_ss = Float32(ss[:i])
const n_ss = Float32(ss[:n])

# Volatilitat del xoc: calibrada (no estimada), igual que a simulate_data.jl
const σ_ε_cal = 0.01f0

# =====================================================================
# 2. DADES REALS I ESTADÍSTICS VAR(1)
#    MacroModelling::simulate retorna log-desviacions de l'estat estacionari.
#    Usem les 4 observables del model: y, c, i, n.
# =====================================================================
df_real  = CSV.read("data/simulated_data.csv", DataFrame)
obs_real = Matrix{Float64}(df_real[!, [:y, :c, :i, :n]])  # (T × 4), log-desviacions de SS

function estimate_var1(data::Matrix{Float64})
    # OLS matricial: Y_t = B' X_{t-1} + E_t
    Y = data[2:end, :]
    X = data[1:end-1, :]
    return vec(X \ Y)   # aplana B en un vector de moments auxiliars
end

const coefs_var_real = estimate_var1(obs_real)

# =====================================================================
# 3. ARQUITECTURA DE LA XARXA NEURONAL (Lux.jl)
#    Entrades: (k_{t-1}, A_t) — Variables d'estat
#    Sortides: (c_t, n_t)     — Variables de control (funcions de política)
#
#    Lux usa paràmetres EXPLÍCITS (ps), separats de l'arquitectura.
#    Això permet diferenciació automàtica neta i composabilitat amb Zygote/Enzyme.
# =====================================================================
const policy_net = Lux.Chain(
    Lux.Dense(2 => 64, tanh),
    Lux.Dense(64 => 64, tanh),
    Lux.Dense(64 => 2)          # sortida crua: s'apliquen activacions a continuació
)

@inline function get_decisions(ps, st, states::AbstractMatrix{Float32})
    raw, st_new = policy_net(states, ps, st)
    c = softplus.(raw[1:1, :])   # c_t > 0  (consum positiu)
    n = sigmoid.(raw[2:2, :])    # n_t ∈ (0,1) (fracció de temps treballat)
    return vcat(c, n), st_new
end

# =====================================================================
# 4. BUCLE INTERN: ENTRENAMENT DL AMB OPERADOR AiO (Maliar et al. 2021)
#
#    Minimitza simultàniament:
#      (a) Residu d'Euler intertemporal (operador AiO per eliminar quadratures)
#      (b) Residu intratemporal treball-oci (sense expectatives, afegit directament)
#
#    Paràmetres: θ = [α, β, δ, γ, ρ]
# =====================================================================
function solve_model_dl(θ::Vector{Float64})
    α, β, δ, γ, ρ = Float32.(θ)

    # Desviació estàndard de la distribució ergòdica de log(A_t)
    σ_A_ergod = σ_ε_cal / sqrt(1f0 - ρ^2)

    ps, st = Lux.setup(rng, policy_net)
    opt_st = Optimisers.setup(Optimisers.Adam(5f-3), ps)

    for _ in 1:500
        # --- Mostreig del conjunt ergòdic ---
        k = k_ss .+ randn(rng, Float32, 256) .* (0.15f0 * k_ss)
        k = max.(k, 1f-3)
        A = exp.(randn(rng, Float32, 256) .* σ_A_ergod)

        # --- Xocs futurs independents per a l'operador AiO ---
        A1 = exp.(ρ .* log.(A) .+ randn(rng, Float32, 256) .* σ_ε_cal)
        A2 = exp.(ρ .* log.(A) .+ randn(rng, Float32, 256) .* σ_ε_cal)

        states_t = vcat(k', A')   # (2 × 256)

        # --- Gradient respecte als pesos de la xarxa (ps) ---
        _, gs = Zygote.withgradient(ps) do p
            # Decisions en t
            dec, _ = get_decisions(p, st, states_t)
            c_t = dec[1, :];  n_t = dec[2, :]

            y_t  = A .* (k .^ α) .* (n_t .^ (1f0 - α))
            k_t  = max.((1f0 - δ) .* k .+ y_t .- c_t, 1f-3)

            # Decisions en t+1 — xoc 1
            d1, _ = get_decisions(p, st, vcat(k_t', A1'))
            c1 = d1[1, :];  n1 = d1[2, :]
            r1 = α .* A1 .* (k_t .^ (α - 1f0)) .* (n1 .^ (1f0 - α))

            # Decisions en t+1 — xoc 2
            d2, _ = get_decisions(p, st, vcat(k_t', A2'))
            c2 = d2[1, :];  n2 = d2[2, :]
            r2 = α .* A2 .* (k_t .^ (α - 1f0)) .* (n2 .^ (1f0 - α))

            # (a) Residu d'Euler (AiO): E[1 - β·(c_t/c_{t+1})·(r_{t+1} + 1 - δ)] = 0
            e_euler_1 = 1f0 .- β .* (c_t ./ c1) .* (r1 .+ 1f0 .- δ)
            e_euler_2 = 1f0 .- β .* (c_t ./ c2) .* (r2 .+ 1f0 .- δ)

            # (b) Residu intratemporal treball-oci: γ·c/(1-n) = w = (1-α)·y/n
            w_t    = (1f0 - α) .* y_t ./ n_t
            e_labor = γ .* c_t ./ (1f0 .- n_t) .- w_t

            # Loss combinada: AiO per Euler + quadràtica per treball-oci
            mean(e_euler_1 .* e_euler_2) + 0.1f0 * mean(e_labor .^ 2)
        end

        opt_st, ps = Optimisers.update!(opt_st, ps, gs[1])
    end

    return ps, st
end

# =====================================================================
# 5. SIMULACIÓ DE L'ECONOMIA AMB LES FUNCIONS DE POLÍTICA APRESES
#    La sortida es normalitza com a log-desviació de l'estat estacionari
#    per ser comparable amb les dades de MacroModelling (obs_real).
# =====================================================================
function simulate_economy(ps, st, θ::Vector{Float64}, T::Int, burn_in::Int=200)
    α, _, δ, _, ρ = θ

    k = Float64(k_ss);  A = 1.0

    # Burn-in: deixem que l'economia convergeixi des del SS
    for _ in 1:burn_in
        dec, st = get_decisions(ps, st, reshape(Float32[k, A], 2, 1))
        c_t = Float64(dec[1, 1]);  n_t = Float64(dec[2, 1])
        y_t = A * k^α * n_t^(1-α)
        i_t = max(y_t - c_t, 1e-8)
        k   = max((1-δ)*k + i_t, 1e-4)
        A   = exp(ρ * log(A) + randn() * 0.01)
    end

    obs = Matrix{Float64}(undef, T, 4)
    for t in 1:T
        dec, st = get_decisions(ps, st, reshape(Float32[k, A], 2, 1))
        c_t = Float64(dec[1, 1]);  n_t = Float64(dec[2, 1])
        y_t = A * k^α * n_t^(1-α)
        i_t = max(y_t - c_t, 1e-8)

        # Log-desviació de l'estat estacionari (mateixa escala que obs_real)
        obs[t, 1] = log(y_t) - log(y_ss)
        obs[t, 2] = log(c_t) - log(c_ss)
        obs[t, 3] = log(i_t) - log(i_ss)
        obs[t, 4] = log(n_t) - log(n_ss)

        k = max((1-δ)*k + i_t, 1e-4)
        A = exp(ρ * log(A) + randn() * 0.01)
    end
    return obs
end

# =====================================================================
# 6. BUCLE EXTERN: INFERÈNCIA INDIRECTA
#    Minimitza la distància quadràtica entre els coeficients VAR(1)
#    estimats sobre dades simulades i sobre dades reals.
#    J(θ) = (B_sim(θ) - B_real)' (B_sim(θ) - B_real)
# =====================================================================
function indirect_inference_loss(θ::Vector{Float64})
    α, β, δ, γ, ρ = θ

    if !(0 < α < 1) || !(0 < β < 1) || !(0 < δ < 1) || !(γ > 0) || !(-1 < ρ < 1)
        return 1e6
    end

    # Bucle intern: entrena la xarxa neuronal per a aquest vector θ
    ps, st = solve_model_dl(θ)

    # Simula les dades amb les funcions de política apreses
    T_sim   = size(obs_real, 1)
    obs_sim = simulate_economy(ps, st, θ, T_sim)

    # Calcula la funció de pèrdua: distància entre moments VAR simulats i reals
    diff = estimate_var1(obs_sim) .- coefs_var_real
    J    = dot(diff, diff)

    println("θ = ", round.(θ, digits=3), " | J = ", round(J, digits=5))
    return J
end

# =====================================================================
# 7. EXECUCIÓ: OPTIMITZACIÓ PER NELDER-MEAD
#    Usem Nelder-Mead perquè la superfície J(θ) és no diferenciable
#    (cadascuna avaluació implica un entrenamient DL estocàstic).
# =====================================================================
θ_init = [0.30, 0.98, 0.020, 2.0, 0.90]   # [α, β, δ, γ, ρ]
θ_true = [0.33, 0.99, 0.025, 2.0, 0.95]   # valors reals de simulate_data.jl

println("Iniciant Inferència Indirecta (DL-Maliar + VAR auxiliar)...")
result = optimize(
    indirect_inference_loss, θ_init,
    NelderMead(), Optim.Options(iterations=100, show_trace=true)
)

println("\n--- ESTIMACIÓ FINALITZADA ---")
println("Paràmetres estimats: ", round.(Optim.minimizer(result), digits=4))
println("Valors reals:        ", θ_true)
