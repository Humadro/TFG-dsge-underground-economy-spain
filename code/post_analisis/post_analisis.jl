##############################################################################
#  post_analisis.jl
#  Anàlisi econòmica post-estimació — Orsi, Raggi & Turino (2014/2015)
##############################################################################
#
#  Ús:
#   1. Omple `estimated_params` amb els posterior means de l'estimació
#   2. Executa `main()` per generar tots els gràfics, o activa/desactiva
#      blocs individuals amb els keyword arguments de main()
#   3. Les figures es guarden a outputs/orsi/
#
#  Dependències: MacroModelling, Plots, StatsPlots, DataFrames, CSV, AxisKeys
#
##############################################################################

using MacroModelling
using Dates
using Plots, StatsPlots
using DataFrames, CSV
using AxisKeys
using Printf
using JLD2, MCMCChains

include(joinpath(@__DIR__, "..", "orsi.jl"))   # defineix OrsiTurinoModel

# ── Estil global per a tots els gràfics ──────────────────────────────────────
default(
    fontfamily    = "Helvetica",
    framestyle    = :box,
    grid          = true,
    gridalpha     = 0.25,
    linewidth     = 1.6,
    tickfontsize  = 7,
    labelfontsize = 8,
    titlefontsize = 9,
)

##############################################################################
#  SECCIÓ 1: PARÀMETRES ESTIMATS
##############################################################################
#
#  Els paràmetres estimats via NUTS es carreguen automàticament com a
#  posterior means de la cadena guardada per bayesian_estimation_orsi_real.jl
#  (evita copiar valors a mà des de describe_orsi_real.txt). Els paràmetres
#  calibrats (no estimats: preferències/fiscalitat d'estat estacionari) es
#  mantenen com a constants — veure writing/06_real.qmd per la justificació.
#
##############################################################################

const REAL_CHAIN_PATH = joinpath(@__DIR__, "..", "..", "outputs", "orsi",
                                  "real_no_covid_3000iter", "cadenes_orsi_real.jld2")

const CALIBRATED_PARAMS = Dict{Symbol, Float64}(
    :beta     => 0.997,    # bons a 10A Espanya 1995-2024
    :sigma    => 0.99,     # no identificable sense dades de tipus d'interès
    :B_0      => 81.74,    # calibrat bisecció: n_m_ss+n_u_ss≈0.19 (Espanya)
    :B_1      => 300.0,    # co-identificat amb phi a través de n_u latent
    :gamma_ss => 1.00,
    :tau_c_ss => 0.25,     # Espanya (vs 0.40 Itàlia)
    :tau_h_ss => 0.23,     # Espanya (vs 0.35 Itàlia)
    :tau_s_ss => 0.23,     # Espanya (vs 0.20 Itàlia)
    :s        => 1.70,     # LGT arts. 191-193
)

"""
    load_posterior_means(chain_path) -> Dict{Symbol, Float64}

Llegeix una cadena NUTS guardada (`@save ... chain_real`) i retorna la
mitjana posterior de cada paràmetre estimat, amb les mateixes xifres que
mostraria `describe(chain_real)`.
"""
function load_posterior_means(chain_path::String)
    chain = JLD2.load(chain_path, "chain_real")
    stats = MCMCChains.summarystats(chain)
    return Dict(Symbol(p) => Float64(m) for (p, m) in zip(stats.nt.parameters, stats.nt.mean))
end

estimated_params = merge(CALIBRATED_PARAMS, load_posterior_means(REAL_CHAIN_PATH))

# ── Nomenclatura llegible per als gràfics ─────────────────────────────────────

const SHOCK_LABELS = Dict{Symbol, String}(
    :eps_a    => "Productivitat formal (A)",
    :eps_b    => "Productivitat informal (B)",
    :eps_x    => "Tecnologia inversió (xi_x)",
    :eps_xi_h => "Preferència laboral (xi_h)",
    :eps_c    => "Impost corporatiu (tau_c)",
    :eps_h    => "Impost personal (tau_h)",
    :eps_s    => "Cotitzacions socials (tau_s)",
    :eps_p    => "Inspecció fiscal (p)",
)

const VAR_LABELS = Dict{Symbol, String}(
    :y                 => "Output total",
    :y_m               => "Output formal",
    :y_u               => "Output informal",
    :underground_share => "Quota submergida (y_u/y)",
    :c                 => "Consum",
    :inv               => "Inversió",
    :G_c               => "Recaptació corporativa",
    :G_h               => "Recaptació personal (IRPF)",
    :G_s               => "Cotitzacions",
    :TE                => "Evasió fiscal (TE)",
    :n_m               => "Treball formal",
    :n_u               => "Treball informal",
)

# Llistes de variables i shocks d'ús freqüent
const IRF_VARIABLES = [:y, :y_m, :y_u, :underground_share,
                        :c, :inv, :G_c, :G_h, :G_s, :TE, :n_m, :n_u]

const ALL_SHOCKS    = [:eps_a, :eps_b, :eps_x, :eps_xi_h,
                        :eps_c, :eps_h, :eps_s, :eps_p]
const SHOCKS_STR    = Set(string.(ALL_SHOCKS))

const FEVD_VARS     = [:underground_share, :y, :c, :inv, :G_c, :TE]
const FEVD_HORIZONS = [1, 4, 8, 20]
const HD_VARS       = [:underground_share, :y, :TE]

"""
Resol la correspondència entre els noms de shocks definits (ALL_SHOCKS) i els
noms reals que MacroModelling usa a l'eix d'un KeyedArray.

Retorna un Dict{String,String} que mapeja "nom_definit" → "nom_en_eix".
Si tots es troben per coincidència exacta, el diccionari és buit.
Si hi ha N shocks sense coincidència i N claus sobrants a l'eix, s'aparellen
en ordre (solució única quan N=1, que és el cas típic amb eps_x).
"""
function _build_shock_axis_map(ax_shocks::Vector)
    # MacroModelling afegeix el sufix ₍ₓ₎ als shocks (p.ex. eps_a → eps_a₍ₓ₎).
    # Fem prefix-match: "eps_a" troba "eps_a₍ₓ₎".
    match = Dict{String,String}()
    for s in ALL_SHOCKS
        s_str = string(s)
        i = findfirst(k -> startswith(string(k), s_str), ax_shocks)
        if i !== nothing
            match[s_str] = string(ax_shocks[i])
        else
            @warn "Shock '$s_str' no trobat a l'eix de get_shock_decomposition."
        end
    end
    return match
end

##############################################################################
#  SECCIÓ 1B: AUXILIARS DE PARÀMETRES I STEADY STATE
##############################################################################

"""
Construeix el vector de paràmetres en l'ordre que espera MacroModelling.
Usa `model.parameter_values` com a base per als paràmetres no inclosos a `params`.
"""
function params_to_vector(model, params::Dict{Symbol, Float64})
    param_names = Symbol.(get_parameters(model))
    idx         = Dict(n => i for (i, n) in enumerate(param_names))
    p_vec       = try
        copy(model.parameter_values)
    catch
        zeros(Float64, length(param_names))
    end
    for (name, val) in params
        if haskey(idx, name)
            p_vec[idx[name]] = val
        else
            @warn "Paràmetre '$name' no reconegut al model; s'ignora."
        end
    end
    return p_vec
end

"""
Actualitza els paràmetres del model in-place (útil per simplificar crides).
"""
function update_parameters!(model, params::Dict{Symbol, Float64})
    p_vec = params_to_vector(model, params)
    try
        model.parameter_values .= p_vec
    catch
        @warn "No s'ha pogut modificar model.parameter_values in-place."
    end
    return nothing
end

"""
Retorna l'estat estacionari com a Dict{Symbol, Float64}.
Gestiona els dos formats de retorn que utilitza MacroModelling
(KeyedArray i DataFrame).
"""
function compute_steady_state(model, params::Dict{Symbol, Float64})
    p_vec = params_to_vector(model, params)
    ss    = get_steady_state(model; parameters = p_vec)
    return _parse_steady_state(ss)
end

function _parse_steady_state(ss)
    d = Dict{Symbol, Float64}()
    # Intent 1: KeyedArray — usa parent() per evitar l'ambigüitat [] vs ()
    try
        ax1 = axiskeys(ss, 1)
        raw  = parent(ss)
        nd   = ndims(raw)
        for (i, name) in enumerate(ax1)
            d[Symbol(string(name))] = Float64(nd == 1 ? raw[i] : raw[i, 1])
        end
        isempty(d) || return d
    catch end
    # Intent 2: DataFrame / taula amb files
    try
        for row in eachrow(ss)
            d[Symbol(string(row[1]))] = Float64(row[2])
        end
        isempty(d) || return d
    catch end
    @warn "Format de steady state no reconegut: $(typeof(ss))"
    return d
end

"""Resol el model i retorna la representació d'espai d'estats."""
solve_model(model, params::Dict{Symbol, Float64}) =
    get_solution(model; parameters = params_to_vector(model, params))

##############################################################################
#  SECCIÓ 2: FUNCIONS D'IMPULS-RESPOSTA (IRFs)
##############################################################################
#
#  get_irfs (MacroModelling) retorna un KeyedArray [variable x shock x periode]
#  amb desviacions logarítmiques respecte l'estat estacionari.
#  La magnitud del shock = 1 desviació estàndard (conveni estàndard DSGE).
#  Genera un PNG per shock amb un subplot per cada variable d'interès.
#
##############################################################################

"""
    plot_irfs(model, params; horizon, shocks, variables, save_figs, output_dir)

Calcula les IRFs i genera un arxiu PNG per shock.
Retorna Dict{Symbol, Plot} amb una figura per shock.
"""
function plot_irfs(model, params::Dict{Symbol, Float64};
                   horizon   ::Int             = 40,
                   shocks    ::Vector{Symbol}  = ALL_SHOCKS,
                   variables ::Vector{Symbol}  = IRF_VARIABLES,
                   save_figs ::Bool            = true,
                   output_dir::String          = "outputs/orsi/post_pruebas")

    mkpath(output_dir)
    p_vec = params_to_vector(model, params)

    # Shape: [variable × período × shock]  (eix 1 = vars, eix 2 = períodes, eix 3 = shocks)
    irfs = get_irfs(model; parameters = p_vec, periods = horizon)

    ax_vars   = collect(axiskeys(irfs, 1))   # noms de variables
    ax_shocks = collect(axiskeys(irfs, 3))   # noms de shocks

    find_var(v_str)   = findfirst(k -> string(k) == v_str, ax_vars)
    find_shock(s_str) = findfirst(k -> string(k) == s_str, ax_shocks)

    raw = parent(irfs)   # Array plain [var × période × shock]

    figures = Dict{Symbol, Plots.Plot}()
    ncols   = 4

    for shock in shocks
        s_idx = find_shock(string(shock))
        if s_idx === nothing
            @warn "Shock $(shock) no trobat a l'eix 3; s'omet. Shocks disponibles: $ax_shocks"
            continue
        end

        shock_label = get(SHOCK_LABELS, shock, string(shock))
        nrows       = ceil(Int, length(variables) / ncols)
        panels      = Plots.Plot[]

        for var in variables
            var_label = get(VAR_LABELS, var, string(var))
            v_idx     = find_var(string(var))
            if v_idx === nothing
                push!(panels, plot(title = var_label, framestyle = :none, label = false))
                continue
            end

            vals = vec(raw[v_idx, :, s_idx])   # [variable × período × shock]

            p = plot(1:horizon, vals;
                     title     = var_label,
                     label     = false,
                     ylabel    = "% dev. s.e.",
                     color     = :steelblue,
                     linewidth = 1.6)
            hline!(p, [0.0]; color = :black, linewidth = 0.6,
                   linestyle = :dot, label = false)
            push!(panels, p)
        end

        fig = plot(panels...;
                   layout     = (nrows, ncols),
                   size       = (ncols * 230, nrows * 185),
                   plot_title = "IRF — $shock_label",
                   margin     = 3Plots.mm,
                   top_margin = 8Plots.mm)

        figures[shock] = fig
        if save_figs
            path = joinpath(output_dir, "irf_$(shock).png")
            savefig(fig, path)
            println("  Guardat: $path")
        end
    end

    return figures
end

##############################################################################
#  SECCIÓ 3: DESCOMPOSICIÓ DE VARIÀNCIA (FEVD)
##############################################################################
#
#  get_variance_decomposition retorna la fracció de variància (0–1) explicada
#  per cada shock per a cada variable i horitzó temporal.
#  Multipliquem per 100 per expressar en percentatge.
#  Genera un stacked bar plot per variable i un CSV amb la taula completa.
#
##############################################################################

const FEVD_PALETTE = [
    :steelblue, :darkorange, :seagreen, :purple,
    :gold,      :tomato,     :teal,     :royalblue,
]

"""
    variance_decomposition(model, params; variables, shocks, ...)

Calcula la FEVD de llarg termini (∞ horitzó) i genera:
  · fevd_table.csv amb els percentatges per variable i shock
  · Un stacked bar plot per a cada variable de `variables`
Retorna (DataFrame, Dict{Symbol, Plot}).

Nota: get_variance_decomposition retorna [variable × shock] (2D), sense eix d'horitzó.
"""
function variance_decomposition(model, params::Dict{Symbol, Float64};
                                 variables ::Vector{Symbol} = FEVD_VARS,
                                 shocks    ::Vector{Symbol} = ALL_SHOCKS,
                                 save_figs ::Bool           = true,
                                 output_dir::String         = "outputs/orsi/post_pruebas")

    mkpath(output_dir)
    p_vec = params_to_vector(model, params)

    # Shape: [variable × shock]  (2D, llarg termini)
    fevd_raw  = get_variance_decomposition(model; parameters = p_vec)
    ax_vars   = collect(axiskeys(fevd_raw, 1))
    ax_shocks = collect(axiskeys(fevd_raw, 2))
    raw       = parent(fevd_raw)   # Matrix Float64 [variable × shock]

    find_var(v_str)   = findfirst(k -> string(k) == v_str, ax_vars)
    find_shock(s_str) = findfirst(k -> string(k) == s_str, ax_shocks)

    shock_labels_vec = [get(SHOCK_LABELS, s, string(s)) for s in shocks]

    # ── Taula DataFrame ────────────────────────────────────────────────────
    rows = Dict{Symbol, Any}[]
    for var in variables
        v_idx = find_var(string(var))
        v_idx === nothing && continue
        row = Dict{Symbol, Any}(:variable => string(var))
        for shock in shocks
            s_idx = find_shock(string(shock))
            row[shock] = s_idx === nothing ? NaN :
                         round(raw[v_idx, s_idx] * 100, digits = 2)
        end
        push!(rows, row)
    end
    df_fevd = DataFrame(rows)
    println("\n── FEVD llarg termini (%) ──")
    show(df_fevd; allcols = true)
    println()
    CSV.write(joinpath(output_dir, "fevd_table.csv"), df_fevd)

    # ── Stacked bar plots (un per variable) ───────────────────────────────
    figs = Dict{Symbol, Plots.Plot}()

    for var in variables
        var_label = get(VAR_LABELS, var, string(var))
        v_idx     = find_var(string(var))
        v_idx === nothing && continue

        # vals[i] = % variancia explicada pel shock i
        vals = map(shocks) do shock
            s_idx = find_shock(string(shock))
            s_idx === nothing ? 0.0 : raw[v_idx, s_idx] * 100
        end

        # groupedbar espera mat [n_grups × n_barres]; aquí 1 barra
        mat = reshape(vals, :, 1)

        fig = groupedbar(
            mat';
            bar_position   = :stack,
            label          = reshape(shock_labels_vec, 1, :),
            color          = reshape(FEVD_PALETTE[1:length(shocks)], 1, :),
            xticks         = (1:1, [var_label]),
            ylabel         = "Variancia explicada (%)",
            title          = "FEVD (∞) — $var_label",
            legend         = :outertopright,
            legendfontsize = 6,
            size           = (500, 400),
            ylims          = (0, 105),
            bar_width      = 0.5,
        )

        figs[var] = fig
        if save_figs
            path = joinpath(output_dir, "fevd_$(var).png")
            savefig(fig, path)
            println("  Guardat: $path")
        end
    end

    return df_fevd, figs
end

##############################################################################
#  SECCIÓ 4: DESCOMPOSICIÓ HISTÒRICA
##############################################################################
#
#  get_shock_decomposition aplica el Kalman smoother i retorna les contribucions
#  individuals de cada shock a la desviació de cada variable respecte el ss.
#
#  Sortida MacroModelling: KeyedArray [variable x (shocks + extras) x T]
#  on "extras" pot incloure :initial_conditions i :mean.
#
#  Visualització: barres apilades (positives cap amunt, negatives cap avall)
#  amb una línia negra pel total observat. Un PNG per variable.
#
##############################################################################

# Agrupació dels shocks per naturalesa econòmica
const HD_SHOCK_GROUPS = [
    ("Productivitat",      [:eps_a, :eps_b],          :steelblue),
    ("Fiscal",             [:eps_c, :eps_h, :eps_s],  :darkorange),
    ("Inspeccio",          [:eps_p],                  :seagreen),
    ("Tecnologia inv.",    [:eps_x],                  :gold),
    ("Laboral",            [:eps_xi_h],               :purple),
]

"""
    historical_decomposition(model, data, params; ...)

Descompon l'evolució de `variables` en contribucions per grup de shocks
utilitzant el Kalman smoother de MacroModelling.

`data`: DataFrame amb les mateixes columnes observables que real_data_orsi.csv
        (date, obs_c, obs_inv, obs_Gc, obs_Gs, obs_Gh, obs_wh, obs_p).

Retorna Dict{Symbol, Plot}.
"""
function historical_decomposition(model,
                                   data::DataFrame,
                                   params::Dict{Symbol, Float64};
                                   variables ::Vector{Symbol} = HD_VARS,
                                   obs_names ::Vector{Symbol} = [:obs_c, :obs_inv,
                                                                  :obs_Gc, :obs_Gh,
                                                                  :obs_wh],
                                   save_figs ::Bool           = true,
                                   output_dir::String         = "outputs/orsi/post_pruebas")

    mkpath(output_dir)
    p_vec = params_to_vector(model, params)
    T     = nrow(data)

    # Format KeyedArray [observables x T] (idèntic al de l'estimació bayesiana)
    mat        = Matrix(data[!, obs_names])'
    data_keyed = KeyedArray(mat, (obs_names, 1:T))

    hd_raw = get_shock_decomposition(model,
                                      data_keyed;
                                      parameters = p_vec)
    # hd_raw: KeyedArray [variable x (shocks + initial_conditions + mean) x T]

    ax_hd_vars  = axiskeys(hd_raw, 1)
    all_keys_raw  = axiskeys(hd_raw, 2)
    # MacroModelling pot usar "initial_conditions", "Initial_values", "mean", "constant"
    non_shock_ids = Set(["mean", "initial_conditions", "Initial_values", "constant"])
    shock_keys    = [k for k in all_keys_raw if !(string(k) in non_shock_ids)]

    find_key_hd(ax, s) = findfirst(k -> string(k) == s, ax)

    # Mapeja nom definit → nom real a l'eix (resol eps_x si cal)
    hd_shock_map     = _build_shock_axis_map(collect(shock_keys))
    inv_hd_shock_map = Dict(v => Symbol(k) for (k, v) in hd_shock_map)

    # Dict{Symbol,Int}: símbol canònic → índex a all_keys_raw
    hd_shock_to_idx = Dict{Symbol, Int}()
    for (k_idx, k) in enumerate(all_keys_raw)
        string(k) in non_shock_ids && continue
        canon = get(inv_hd_shock_map, string(k), Symbol(string(k)))
        hd_shock_to_idx[canon] = k_idx
    end

    figs = Dict{Symbol, Plots.Plot}()

    for var in variables
        var_str   = string(var)
        var_label = get(VAR_LABELS, var, var_str)
        periods   = 1:T

        v_pos = find_key_hd(ax_hd_vars, var_str)
        if v_pos === nothing
            @warn "Variable '$var_str' no trobada a la HD; s'omet."
            continue
        end

        # Sèrie total = suma de totes les contribucions excepte la mitjana
        hd_mat = parent(hd_raw)
        shock_positions = [find_key_hd(all_keys_raw, string(k)) for k in shock_keys]
        total_vals = zeros(Float64, T)
        for k_pos in filter(!isnothing, shock_positions)
            total_vals .+= vec(hd_mat[v_pos, k_pos, :])
        end
        # Suma condicions inicials si existeixen
        ic_key = something(
            find_key_hd(all_keys_raw, "Initial_values"),
            find_key_hd(all_keys_raw, "initial_conditions"),
            nothing,
        )
        if ic_key !== nothing
            total_vals .+= vec(hd_mat[v_pos, ic_key, :])
        end

        # Eix x: índexs seqüencials (1:T) — bar!+fillrange al backend GR
        # ignora x-values flotants grans; les dates apareixen via xticks.
        date_strs  = string.(data.date)
        years_all  = [parse(Int, s[1:4]) for s in date_strs]
        months_all = [parse(Int, s[6:7]) for s in date_strs]
        xvals      = collect(1:T)

        # Ticks: primer trimestre de cada any múltiple de 4
        # (i sempre el primer punt de la mostra)
        tick_pos    = Int[]
        tick_labels = String[]
        for i in 1:T
            yr = years_all[i]
            if months_all[i] == 1 && (i == 1 || yr % 4 == 0)
                push!(tick_pos, i)
                push!(tick_labels, string(yr))
            end
        end

        fig = plot(;
                   title          = "Descomposicio historica — $var_label",
                   ylabel         = "Desviacio del s.e. (%)",
                   xlabel         = "",
                   xticks         = (tick_pos, tick_labels),
                   xrotation      = 45,
                   legend         = :outertopright,
                   legendfontsize = 6,
                   size           = (820, 400))

        # Stacking correcte per a contribucions mixtes (+ i -)
        pos_bottom = zeros(Float64, T)
        neg_bottom = zeros(Float64, T)

        for (group_name, group_shocks, color) in HD_SHOCK_GROUPS
            contrib = zeros(Float64, T)
            for sh in group_shocks
                k_pos = get(hd_shock_to_idx, sh, nothing)
                k_pos === nothing && continue
                contrib .+= vec(hd_mat[v_pos, k_pos, :])
            end

            pos_part = max.(contrib, 0.0)
            neg_part = min.(contrib, 0.0)

            bar!(fig, xvals, pos_bottom .+ pos_part;
                 fillrange = pos_bottom, label = group_name,
                 color = color, alpha = 0.75, linewidth = 0, bar_width = 0.85)
            bar!(fig, xvals, neg_bottom .+ neg_part;
                 fillrange = neg_bottom, label = false,
                 color = color, alpha = 0.75, linewidth = 0, bar_width = 0.85)

            pos_bottom .+= pos_part
            neg_bottom .+= neg_part
        end

        # Condicions inicials (si MacroModelling les retorna)
        if ic_key !== nothing
            ic_v = vec(hd_mat[v_pos, ic_key, :])
            bar!(fig, xvals, ic_v; label = "Cond. inicials",
                 color = :gray60, alpha = 0.5, linewidth = 0, bar_width = 0.85)
        end

        # Sèrie total observada (línia negra)
        plot!(fig, xvals, total_vals;
              color = :black, linewidth = 1.8, label = "Total")
        hline!(fig, [0.0]; color = :black, linewidth = 0.5,
               linestyle = :dot, label = false)

        display(fig)
        figs[var] = fig
        if save_figs
            path = joinpath(output_dir, "historical_decomposition_$(var_str).png")
            savefig(fig, path)
            println("  Guardat: $path")
        end
    end

    return figs
end

##############################################################################
#  SECCIÓ 4B: SÈRIES OBSERVABLES — NIVELLS I TRANSFORMACIÓ
##############################################################################

const OBS_DISPLAY = [
    (:obs_c,   :lev_c,   "Consum"),
    (:obs_inv, :lev_inv, "Inversió"),
    (:obs_Gc,  :lev_Gc,  "Recaptació IS"),
    (:obs_Gh,  :lev_Gh,  "Recaptació IRPF"),
    (:obs_wh,  :lev_wh,  "Massa salarial"),
]

"""
    plot_observable_pairs(; data_path, levels_path, output_dir, save_figs)

Panell 5×2 per a l'annex: columna esquerra = nivell real p.c. indexat (base 100),
columna dreta = 100·Δlog (la sèrie que entra al model).
Requereix que existeixi `data/orsi/real_data_orsi_levels.csv`
(generat per `load_real_data_orsi.jl`).
"""
function plot_observable_pairs(;
    data_path  ::String = "data/orsi/real_data_orsi.csv",
    levels_path::String = "data/orsi/real_data_orsi_levels.csv",
    output_dir ::String = "outputs/orsi/real_no_covid_post",
    save_figs  ::Bool   = true)

    mkpath(output_dir)
    df_t = CSV.read(data_path,   DataFrame)
    df_l = CSV.read(levels_path, DataFrame)

    mk_ticks(df) = begin
        n    = nrow(df)
        yrs  = [parse(Int, string(d)[1:4]) for d in df.date]
        pos  = collect(1:16:n)
        (pos, [string(yrs[i]) for i in pos])
    end
    tp_l, tl_l = mk_ticks(df_l)
    tp_t, tl_t = mk_ticks(df_t)

    Tl = nrow(df_l)
    Tt = nrow(df_t)

    panels = Plots.Plot[]
    for (obs_col, lev_col, name) in OBS_DISPLAY
        lev_raw = Float64.(df_l[!, lev_col])
        lev_idx = 100.0 .* lev_raw ./ lev_raw[1]
        trans   = Float64.(df_t[!, obs_col])

        pl = plot(1:Tl, lev_idx;
                  title = "$name — nivell (base 100)",
                  label = false, color = :steelblue, linewidth = 1.4,
                  ylabel = "Índex")
        hline!(pl, [100.0]; color = :black, linewidth = 0.5,
               linestyle = :dot, label = false)
        plot!(pl; xticks = (tp_l, tl_l))

        pt = plot(1:Tt, trans;
                  title = "$name — 100·Δlog",
                  label = false, color = :darkorange, linewidth = 1.4,
                  ylabel = "%")
        hline!(pt, [0.0]; color = :black, linewidth = 0.5,
               linestyle = :dot, label = false)
        plot!(pt; xticks = (tp_t, tl_t))

        push!(panels, pl, pt)
    end

    fig = plot(panels...;
               layout     = (5, 2),
               size       = (920, 1100),
               margin     = 3Plots.mm,
               top_margin = 5Plots.mm)

    display(fig)
    if save_figs
        path = joinpath(output_dir, "observable_pairs.png")
        savefig(fig, path)
        println("  Guardat: $path")
    end
    return fig
end

##############################################################################
#  SECCIÓ 5: CORBES DE LAFFER
##############################################################################
#
#  Per a cada paràmetre fiscal (tau_c_ss, tau_h_ss, tau_s_ss, p_ss),
#  varia el valor en una graella de n_points punts i recalcula l'estat
#  estacionari per obtenir output, recaptació, quota submergida, evasió
#  i benestar W = U_ss / (1-beta).
#
#  Nota de rendiment: el bloc computa n_points x length(tax_params)
#  estats estacionaris. Amb n_points=50 i 4 impostos = 200 resolucions.
#  Pot trigar uns minuts. La línia vertical negra indica el valor baseline.
#
##############################################################################

"""
    run_laffer_analysis(model, params; tax_params, n_points, ...)

Genera les corbes de Laffer per als impostos de `tax_params`.
Retorna Dict{Symbol, Plot}.
"""
function run_laffer_analysis(model, params::Dict{Symbol, Float64};
                              tax_params ::Vector{Symbol} = [:tau_c_ss, :tau_h_ss,
                                                             :tau_s_ss, :p_ss],
                              n_points   ::Int            = 25,
                              save_figs  ::Bool           = true,
                              output_dir ::String         = "outputs/orsi/post_pruebas")

    mkpath(output_dir)
    p_vec_base  = params_to_vector(model, params)
    param_names = Symbol.(get_parameters(model))
    param_idx   = Dict(n => i for (i, n) in enumerate(param_names))
    beta_val    = get(params, :beta, 0.997)

    RANGES = Dict{Symbol, Tuple{Float64, Float64}}(
        :tau_c_ss => (0.05, 0.70),
        :tau_h_ss => (0.05, 0.65),
        :tau_s_ss => (0.02, 0.50),
        :p_ss     => (0.001, 0.15),
    )
    TAX_LABELS = Dict{Symbol, String}(
        :tau_c_ss => "Tipus corporatiu (tau_c)",
        :tau_h_ss => "Tipus personal (tau_h)",
        :tau_s_ss => "Cotitzacions socials (tau_s)",
        :p_ss     => "Prob. inspeccio (p)",
    )

    ss_track_vars = [:y, :G_c, :G_h, :G_s, :underground_share, :TE, :utility]
    figs_all      = Dict{Symbol, Plots.Plot}()

    for tax in tax_params
        lo, hi     = RANGES[tax]
        grid       = range(lo, hi; length = n_points)
        baseline_v = get(params, tax, NaN)
        tax_label  = get(TAX_LABELS, tax, string(tax))

        results     = Dict(v => fill(NaN, n_points) for v in ss_track_vars)
        welfare_v   = fill(NaN, n_points)
        total_rev_v = fill(NaN, n_points)

        for (i, val) in enumerate(grid)
            p_vec_new = copy(p_vec_base)
            haskey(param_idx, tax) && (p_vec_new[param_idx[tax]] = val)
            try
                ss_dict = _parse_steady_state(
                    get_steady_state(model; parameters = p_vec_new)
                )
                for v in ss_track_vars
                    results[v][i] = get(ss_dict, v, NaN)
                end
                u_ss = get(ss_dict, :utility, NaN)
                isnan(u_ss) || (welfare_v[i] = u_ss / (1 - beta_val))

                gc = get(ss_dict, :G_c, 0.0)
                gh = get(ss_dict, :G_h, 0.0)
                gs = get(ss_dict, :G_s, 0.0)
                total_rev_v[i] = gc + gh + gs
            catch e
                @warn "Laffer ($tax = $(round(val, digits=4))): $e"
            end
        end

        grid_v  = collect(grid)
        ncols   = 2
        panels  = Plots.Plot[]

        # Panells principals: y, underground_share, TE, G_c
        for (var, col) in [(:y, :steelblue), (:underground_share, :tomato),
                            (:TE, :darkorange), (:G_c, :seagreen)]
            var_label = get(VAR_LABELS, var, string(var))
            valid     = .!isnan.(results[var])
            p = plot(grid_v[valid], results[var][valid];
                     title = var_label, xlabel = tax_label,
                     label = false, color = col, linewidth = 1.8)
            isnan(baseline_v) || vline!(p, [baseline_v];
                color = :black, linestyle = :dash, linewidth = 1.0,
                label = "baseline")
            push!(panels, p)
        end

        # Panell: recaptació total (G = G_c + G_h + G_s)
        valid_r = .!isnan.(total_rev_v)
        p_r = plot(grid_v[valid_r], total_rev_v[valid_r];
                   title = "Recaptacio total (G)", xlabel = tax_label,
                   label = false, color = :coral, linewidth = 1.8)
        isnan(baseline_v) || vline!(p_r, [baseline_v];
            color = :black, linestyle = :dash, linewidth = 1.0, label = "baseline")
        push!(panels, p_r)

        # Panell: benestar
        valid_w = .!isnan.(welfare_v)
        p_w = plot(grid_v[valid_w], welfare_v[valid_w];
                   title = "Benestar (W = U/(1-beta))", xlabel = tax_label,
                   label = false, color = :darkgreen, linewidth = 1.8)
        isnan(baseline_v) || vline!(p_w, [baseline_v];
            color = :black, linestyle = :dash, linewidth = 1.0, label = "baseline")
        push!(panels, p_w)

        nrows = ceil(Int, length(panels) / ncols)
        fig = plot(panels...;
                   layout     = (nrows, ncols),
                   size       = (ncols * 340, nrows * 255),
                   plot_title = "Corba de Laffer — $tax_label",
                   margin     = 5Plots.mm,
                   top_margin = 10Plots.mm)

        figs_all[tax] = fig
        if save_figs
            path = joinpath(output_dir, "laffer_$(tax).png")
            savefig(fig, path)
            println("  Guardat: $path")
        end
    end

    return figs_all
end

##############################################################################
#  SECCIÓ 6: BENESTAR (WELFARE)
##############################################################################
#
#  Utilitat instantània (definida al model, orsi.jl):
#    U = log(c) - B_0·xi_h·(n_m+n_u)^(1+xi)/(1+xi) - B_1·n_u^(1+phi)/(1+phi)
#
#  Benestar descomptat en ss:
#    W = U_ss / (1 - beta)
#
#  Variació equivalent de consum (ECV) entre baseline i contrafactual:
#    lambda = exp((1 - beta) * (W_cf - W_baseline)) - 1    [log-utilitat en c]
#
#  Si `alt_params` és nothing, només calcula el baseline.
#  Si `alt_params` és un Dict, calcula la ECV respecte la configuració
#  alternativa (útil per comparar règims fiscals o d'inspecció).
#
##############################################################################

"""
    compute_welfare(model, params; alt_params)

Calcula el benestar en estat estacionari W = U_ss/(1-beta) i, opcionalment,
la variació equivalent de consum (ECV) respecte una configuració alternativa.
Retorna Dict{Symbol, Any} amb claus :baseline, :counterfactual (opcional) i
:ecv (opcional).
"""
function compute_welfare(model, params::Dict{Symbol, Float64};
                          alt_params::Union{Dict{Symbol, Float64}, Nothing} = nothing)

    beta_val = get(params, :beta, 0.997)

    function eval_welfare(pars::Dict{Symbol, Float64})
        p_vec   = params_to_vector(model, pars)
        ss_dict = _parse_steady_state(get_steady_state(model; parameters = p_vec))
        c_ss    = get(ss_dict, :c, NaN)
        nm_ss   = get(ss_dict, :n_m, NaN)
        nu_ss   = get(ss_dict, :n_u, NaN)
        u_ss    = get(ss_dict, :utility, NaN)
        W       = isnan(u_ss) ? NaN : u_ss / (1 - beta_val)
        return (c = c_ss, n_m = nm_ss, n_u = nu_ss, utility = u_ss, W = W)
    end

    b = eval_welfare(params)

    println("\n── Benestar en estat estacionari (baseline) ──────────────────")
    @printf("  U_ss  = %+.6f\n", b.utility)
    @printf("  W     = %+.4f\n", b.W)
    @printf("  c_ss  = %.6f\n",  b.c)
    @printf("  n_m   = %.6f\n",  b.n_m)
    @printf("  n_u   = %.6f\n",  b.n_u)

    result = Dict{Symbol, Any}(:baseline => b)

    if alt_params !== nothing
        cf     = eval_welfare(alt_params)
        lambda = exp((1 - beta_val) * (cf.W - b.W)) - 1

        println("\n── Variacio equivalent de consum (ECV) ───────────────────────")
        @printf("  W_alt   = %+.4f\n",          cf.W)
        @printf("  ECV (lambda) = %+.4f (%.2f%%)\n", lambda, lambda * 100)

        result[:counterfactual] = cf
        result[:ecv]            = lambda
    end

    return result
end

##############################################################################
#  SECCIÓ 7: FUNCIÓ PRINCIPAL
##############################################################################

"""
    main(; run_irfs, run_fevd, run_hd, run_laffer, run_welfare, ...)

Executa el pipeline complet d'anàlisi post-estimació.
Cada bloc és independent i es pot activar/desactivar amb el keyword corresponent.

Keyword arguments:
  run_irfs     :: Bool   — IRFs per a tots els shocks           (default: true)
  run_fevd     :: Bool   — Descomposició de variància FEVD       (default: true)
  run_hd       :: Bool   — Descomposició històrica              (default: false)
  run_laffer   :: Bool   — Corbes de Laffer                     (default: true)
  run_welfare  :: Bool   — Benestar en ss                       (default: true)
  horizon_irfs :: Int    — Horitzó IRF en trimestres            (default: 40)
  data_path    :: String — CSV amb dades reals per a HD         (real_data_orsi.csv)
  output_dir   :: String — Directori de sortida                 (outputs/orsi)
"""
function main(;
    run_irfs    ::Bool   = true,
    run_fevd    ::Bool   = true,
    run_hd      ::Bool   = true,
    run_laffer  ::Bool   = true,
    run_welfare ::Bool   = true,
    run_obs_pairs::Bool  = true,
    horizon_irfs::Int    = 40,
    data_path   ::String = "data/orsi/real_data_orsi.csv",
    output_dir  ::String = "outputs/orsi/real_no_covid_post",
    hd_end_date ::Date   = Date(2020, 1, 1),   # talla les dades per a la HD (consistent amb l'estimació)
)
    line = "=" ^ 62
    println(line)
    println("  Post-analisi — Orsi, Raggi & Turino (2014)")
    println("  $(today())")
    println(line)
    mkpath(output_dir)

    if run_irfs
        println("\n[1/5] IRFs (horitzó = $horizon_irfs trimestres)...")
        plot_irfs(OrsiTurinoModel, estimated_params;
                  horizon    = horizon_irfs,
                  save_figs  = true,
                  output_dir = output_dir)
    else
        println("\n[1/5] IRFs omeses.")
    end

    if run_fevd
        println("\n[2/5] Descomposicio de variancia (FEVD)...")
        variance_decomposition(OrsiTurinoModel, estimated_params;
                               save_figs  = true,
                               output_dir = output_dir)
    else
        println("\n[2/5] FEVD omesa.")
    end

    if run_hd
        println("\n[3/5] Descomposicio historica...")
        if isfile(data_path)
            data_real = CSV.read(data_path, DataFrame)
            filter!(row -> row.date < hd_end_date, data_real)
            println("    Dades HD: $(nrow(data_real)) trimestres (fins $(hd_end_date - Day(1)))")
            historical_decomposition(OrsiTurinoModel, data_real, estimated_params;
                                     save_figs  = true,
                                     output_dir = output_dir)
        else
            @warn "Fitxer de dades no trobat: $data_path  (bloc omes)"
        end
    else
        println("\n[3/5] Descomposicio historica omesa (run_hd=false).")
    end

    if run_laffer
        println("\n[4/5] Corbes de Laffer (pot trigar uns minuts)...")
        run_laffer_analysis(OrsiTurinoModel, estimated_params;
                            n_points   = 50,
                            save_figs  = true,
                            output_dir = output_dir)
    else
        println("\n[4/5] Laffer omes.")
    end

    if run_welfare
        println("\n[5/5] Benestar en estat estacionari...")
        compute_welfare(OrsiTurinoModel, estimated_params)
    else
        println("\n[5/5] Welfare omes.")
    end

    if run_obs_pairs
        println("\n[+] Sèries observables (nivells + transformació)...")
        plot_observable_pairs(;
            data_path  = data_path,
            output_dir = output_dir)
    end

    println("\n$line")
    println("  Analisi completada. Figures guardades a: $output_dir")
    println(line)
end

# ── Punt d'entrada ────────────────────────────────────────────────────────────
# Descomenta per executar el pipeline complet directament.
# Quan s'inclou com a mòdul (include(...)) des d'altres scripts, deixar comentat.
main()
