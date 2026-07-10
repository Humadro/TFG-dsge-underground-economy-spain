using MacroModelling, Printf

include(joinpath(@__DIR__, "orsi.jl"))
include(joinpath(@__DIR__, "post_analisis", "post_analisis.jl"))  # carrega estimated_params

# Usa els paràmetres definitius de post_analisis.jl
params_post = estimated_params

param_names = Symbol.(get_parameters(OrsiTurinoModel))
all_vars    = get_variables(OrsiTurinoModel)
idx_nm = findfirst(==(string(:n_m)), all_vars)
idx_nu = findfirst(==(string(:n_u)), all_vars)

function hours_given_B0(b0)
    d = copy(params_post)
    d[:B_0] = b0
    p = Float64[d[p] for p in param_names]
    ss = try
        get_steady_state(OrsiTurinoModel, parameters = p)
    catch
        return NaN
    end
    return ss[idx_nm] + ss[idx_nu]
end

# ── Verificació inicial ───────────────────────────────────────────────────────
h1 = hours_given_B0(1.0)
@printf("B_0 = 1.0  →  n_m+n_u = %.4f  (target: 0.19)\n", h1)

# ── Bisecció per trobar B_0* tal que n_m+n_u = 0.19 ─────────────────────────
h_target = 0.19
@printf("\nBisecció per n_m+n_u = %.2f ...\n", h_target)

function bisect_B0(target, lo, hi, tol=1e-5, maxiter=60)
    for _ in 1:maxiter
        mid = (lo + hi) / 2
        hm  = hours_given_B0(mid)
        isnan(hm) && (hi = mid; continue)
        abs(hm - target) < tol && return mid
        hm > target ? (lo = mid) : (hi = mid)
    end
    return (lo + hi) / 2
end

B0_star = bisect_B0(h_target, 1.0, 500.0)
h_check = hours_given_B0(B0_star)

@printf("\n── Resultat ────────────────────────────────────────────────\n")
@printf("  B_0*             = %.4f\n", B0_star)
@printf("  n_m+n_u obtingut = %.6f  (target: %.2f)\n", h_check, h_target)

# Estat estacionari final
d2 = copy(params_post); d2[:B_0] = B0_star
p2 = Float64[d2[p] for p in param_names]
ss2 = get_steady_state(OrsiTurinoModel, parameters = p2)

idx_nu2 = findfirst(==(string(:n_u)), all_vars)
idx_nm2 = findfirst(==(string(:n_m)), all_vars)
n_u2 = ss2[idx_nu2]; n_m2 = ss2[idx_nm2]
@printf("  n_u / (n_m+n_u)  = %.4f  (fracció submergida)\n", n_u2 / (n_m2 + n_u2))

for v in [:c, :y_m, :y_u, :w_m, :w_u]
    idx = findfirst(==(string(v)), all_vars)
    !isnothing(idx) && @printf("  %-12s = %.4f\n", v, ss2[idx])
end

# ── Quota submergida ──────────────────────────────────────────────────────────
idx_ym = findfirst(==(string(:y_m)), all_vars)
idx_yu = findfirst(==(string(:y_u)), all_vars)
if !isnothing(idx_ym) && !isnothing(idx_yu)
    y_m = ss2[idx_ym]; y_u = ss2[idx_yu]
    ug_share = y_u / (y_m + y_u)
    @printf("\n  underground_share (y_u/y) = %.4f  (%.2f%% del PIB)\n", ug_share, ug_share * 100)
end
