
##############################################################################
#  Orsi, Raggi & Turino (2014) — "Size, trend, and policy implications
#  of the underground economy", Review of Economic Dynamics 17, 417-436.
#
#  Todas las variables con sombrero (^) son detrended: Ŝ_t = S_t / Γ_t
#  donde Γ_t = γ * Γ_{t-1} es el progreso tecnológico labor-augmenting.
#
#  Con ese cambio de variables la utilidad marginal detrended es:
#       λ̂_t = ĉ_t^{-σ}   (relación con nivel: λ_t = λ̂_t / Γ_t)
#  y la Euler recoge el factor γ en el descuento efectivo.
#
#  Numeración de ecuaciones según el paper:
#    (1)-(2)  Producción regular y sumergida
#    (3)      Output total de la firma
#    (4)-(7)  CPO de la firma (demandas de factores)
#    (8)      Ley de movimiento del capital
#    (9)      Restricción presupuestaria del hogar
#   (10)      Asignación del capital entre sectores
#   (11)      Euler del hogar (detrended)
#   (12)-(13) Oferta de trabajo regular y sumergido
#   (14)      Arbitraje de capital entre sectores
#   (15)-(17) Recaudación fiscal
#   (18)      Vaciamiento del mercado de bienes (recursos)
##############################################################################

using MacroModelling

MacroModelling.@model OrsiTurinoModel begin

    # -----------------------------------------------------------------------
    # Definición del multiplicador de Lagrange detrended
    # (utilidad marginal del consumo detrended)
    # -----------------------------------------------------------------------
    # λ̂_t = ĉ_t^{-σ}
    lambda[0] = c[0]^(-sigma)

    # -----------------------------------------------------------------------
    # Ecuación de Euler — eq. (11) en variables detrended
    #
    # Paper (niveles):   λ_t / ξ_x_t = β E_t[ λ_{t+1}((1-δ_k)/ξ_x_{t+1}
    #                                           + (1-τ_h_{t+1}) r_m_{t+1}) ]
    # Detrended (÷ Γ_t): usando λ_t = λ̂_t/Γ_t y Γ_{t+1} = γΓ_t,
    #   los Γ_t se cancelan dejando un factor γ en el lado izquierdo:
    #
    #   γ λ̂_t / ξ_x_t = β E_t[ λ̂_{t+1}((1-δ_k)/ξ_x_{t+1}
    #                                      + (1-τ_h_{t+1}) r_m_{t+1}) ]
    # -----------------------------------------------------------------------
    gamma_ss * lambda[0] / xi_x[0] = beta * lambda[1] * (
        (1 - delta_k) / xi_x[1] + (1 - tau_h[1]) * r_m[1]
    )

    # -----------------------------------------------------------------------
    # Oferta de trabajo regular — eq. (12)
    #
    # B_0 (ĥ_m + ĥ_u)^ξ ξ_h_t = (1 - τ_h_t) ŵ_m_t λ̂_t
    # -----------------------------------------------------------------------
    B_0 * (n_m[0] + n_u[0])^xi * xi_h[0] = (1 - tau_h[0]) * w_m[0] * lambda[0]

    # -----------------------------------------------------------------------
    # Oferta de trabajo sumergido — eq. (13)
    #
    # B_0 (ĥ_m + ĥ_u)^ξ ξ_h_t + B_1 ĥ_u^φ = ŵ_u_t λ̂_t
    # -----------------------------------------------------------------------
    B_0 * (n_m[0] + n_u[0])^xi * xi_h[0] + B_1 * n_u[0]^phi = w_u[0] * lambda[0]

    # -----------------------------------------------------------------------
    # Arbitraje del capital entre sectores — eq. (14)
    #
    # r_u_t = (1 - τ_h_t) r_m_t
    # -----------------------------------------------------------------------
    r_u[0] = (1 - tau_h[0]) * r_m[0]


    utility[0] = (
        log(c[0])
        - B_0 * xi_h[0] * ((n_m[0] + n_u[0])^(1 + xi)) / (1 + xi)
        - B_1 * (n_u[0]^(1 + phi)) / (1 + phi)
    )
    # -----------------------------------------------------------------------
    # Funciones de producción detrended — eqs. (1) y (2)
    #
    # ŷ_m_t = A_t k̂_m_{t-1}^{1-α} (γ ĥ_m_t / γ)^α  ←  el factor Γ_t
    #          se cancela porque Γ_{t-1} = Γ_t/γ:
    #   en el paper: y_m = A_t (Γ_t h_m)^α k_m^{1-α}
    #   detrended:   ŷ_m = A_t γ^α ĥ_m^α k̂_m^{1-α}   (k̂_m predeterminado)
    #
    # El factor γ^α se absorbe típicamente en la calibración de A en ss.
    # Lo dejamos explícito para transparencia; se puede incluir en gamma_ss^alpha.
    # -----------------------------------------------------------------------
    y_m[0] = A[0] * (gamma_ss * n_m[0])^alpha * k_m[0]^(1 - alpha)

    y_u[0] = B[0] * (gamma_ss * n_u[0])^alpha_u * k_u[0]^(1 - alpha_u)

    y[0] = y_m[0] + y_u[0]

    # -----------------------------------------------------------------------
    # Condiciones de primer orden de la firma — demanda de factores
    #
    # Capital regular — eq. (4): (1-α) ŷ_m / k̂_m = r̂_m / (1 - τ_c)
    # -----------------------------------------------------------------------
    (1 - alpha) * y_m[0] / k_m[0] = r_m[0] / (1 - tau_c[0])

    # Trabajo regular — eq. (5):
    #   α ŷ_m / ĥ_m = ŵ_m (1 + τ_s - τ_c) / (1 - τ_c)
    alpha * y_m[0] / n_m[0] = w_m[0] * (1 + tau_s[0] - tau_c[0]) / (1 - tau_c[0])

    # Capital sumergido — eq. (6):
    #   (1-α_u) ŷ_u / k̂_u = r̂_u / (1 - p s τ_c)
    (1 - alpha_u) * y_u[0] / k_u[0] = r_u[0] / (1 - p[0] * s * tau_c[0])

    # Trabajo sumergido — eq. (7):
    #   α_u ŷ_u / ĥ_u = ŵ_u
    alpha_u * y_u[0] / n_u[0] = w_u[0]

    # -----------------------------------------------------------------------
    # Ley de movimiento del capital (detrended) — eq. (8)
    #
    # En niveles:  k_{t+1} = ξ_x_t x_t + (1-δ_k) k_t
    # Detrended (÷ Γ_{t+1} = γ Γ_t):
    #   γ k̂_{t+1} = ξ_x_t x̂_t + (1-δ_k) k̂_t
    #
    # En convención MacroModelling (k[0] = k̂_t predeterminado en t+1):
    #   γ k[0] = ξ_x[-1] * inv[-1] + (1-delta_k) * k[-1]
    #
    # Nota: k[0] aquí es el stock que se usa en producción en t+1.
    # -----------------------------------------------------------------------
    gamma_ss * k[0] = xi_x[-1] * inv[-1] + (1 - delta_k) * k[-1]

    # -----------------------------------------------------------------------
    # Asignación del capital entre sectores — eq. (10)
    #
    # k̂_m_t + k̂_u_t = k̂_{t-1}   (capital predeterminado del período anterior)
    # -----------------------------------------------------------------------
    k_m[0] + k_u[0] = k[-1]

    # -----------------------------------------------------------------------
    # Vaciamiento del mercado de bienes (detrended) — eq. en sección 3.5
    #
    # ĉ_t + x̂_t + ĝ_t = ŷ_t
    # -----------------------------------------------------------------------
    c[0] + inv[0] + g[0] = y[0]

    # -----------------------------------------------------------------------
    # Recaudación fiscal detrended — eqs. (15), (16), (17)
    # -----------------------------------------------------------------------
    # Renta personal — eq. (15)
    G_h[0] = tau_h[0] * (w_m[0] * n_m[0] + r_m[0] * k_m[0])

    # Impuesto corporativo (con probabilidad de inspección) — eq. (16)
    G_c[0] = tau_c[0] * (
        p[0] * s * (y_u[0] - w_u[0] * n_u[0])
        + y_m[0] - w_m[0] * n_m[0]
    )

    # Seguridad social — eq. (17)
    G_s[0] = tau_s[0] * w_m[0] * n_m[0]

    # Presupuesto del gobierno balanceado
    g[0] = G_h[0] + G_c[0] + G_s[0]

    # -----------------------------------------------------------------------
    # Evasión fiscal total (variable de análisis de política, sección 3.3)
    # TE_t = τ_s ŵ_u ĥ_u + τ_h(ŵ_u ĥ_u + r̂_u k̂_u) + (1-p) τ_c (ŷ_u - ŵ_u ĥ_u)
    # -----------------------------------------------------------------------
    TE[0] = tau_s[0] * w_u[0] * n_u[0] +
            tau_h[0] * (w_u[0] * n_u[0] + r_u[0] * k_u[0]) +
            (1 - p[0]) * tau_c[0] * (y_u[0] - w_u[0] * n_u[0])

    # Tamaño del sector sumergido (ratio de política)
    underground_share[0] = y_u[0] / y[0]

    # -----------------------------------------------------------------------
    # Procesos estocásticos AR(1) en logaritmos — sección 3.4
    # Todos los shocks: ε ~ N(0, σ²)
    # -----------------------------------------------------------------------
    log(A[0])    = rho_a    * log(A[-1])    + sqrt(var_eps_a)    * eps_a[x]
    log(B[0])    = rho_b    * log(B[-1])    + sqrt(var_eps_b)    * eps_b[x]
    log(xi_x[0]) = rho_x    * log(xi_x[-1]) + sqrt(var_eps_x)    * eps_x[x]

    log(xi_h[0]) = rho_xi_h * log(xi_h[-1]) + sqrt(var_eps_xi_h) * eps_xi_h[x]

    log(tau_c[0]) = (1 - rho_c) * log(tau_c_ss) + rho_c * log(tau_c[-1]) + sqrt(var_eps_c) * eps_c[x]
    log(tau_h[0]) = (1 - rho_h) * log(tau_h_ss) + rho_h * log(tau_h[-1]) + sqrt(var_eps_h) * eps_h[x]
    log(tau_s[0]) = (1 - rho_s) * log(tau_s_ss) + rho_s * log(tau_s[-1]) + sqrt(var_eps_s) * eps_s[x]

    log(p[0]) = (1 - rho_p) * log(p_ss) + rho_p * log(p[-1]) + sqrt(var_eps_p) * eps_p[x]
    # -----------------------------------------------------------------------
    # Ecuaciones de observación — eq. (18) del paper
    #
    # 5 observables usats en l'estimació:
    #   obs_c, obs_inv, obs_Gc, obs_Gh, obs_wh : gamma_pct + 100×Δlog
    #
    # obs_Gs i obs_p no s'usen: obs_Gs requeriria G_s[-1] (augmenta l'espai
    # d'estat); obs_p és latent i difícil d'observar a les dades reals.
    # obs_p es manté per poder simular i analitzar p.
    # -----------------------------------------------------------------------
    obs_c[0]   = gamma_pct + 100 * (log(c[0])   - log(c[-1]))
    obs_inv[0] = gamma_pct + 100 * (log(inv[0]) - log(inv[-1]))
    obs_Gc[0]  = gamma_pct + 100 * (log(G_c[0]) - log(G_c[-1]))
    obs_Gh[0]  = gamma_pct + 100 * (log(G_h[0]) - log(G_h[-1]))
    obs_wh[0]  = gamma_pct + 100 * (log((1 + tau_s[0]) * w_m[0] * n_m[0]) - log((1 + tau_s[-1]) * w_m[-1] * n_m[-1]))
    obs_p[0]   = 100 * log(p[0])

end


##############################################################################
#  Calibración / valores de los parámetros (posterior means, Tabla 1)
##############################################################################
MacroModelling.@parameters OrsiTurinoModel begin

    # --- Preferencias ---
    beta    = 0.997    # factor de descuento subjetivo (calibrat bons 10A Espanya 1995-2024)
    sigma   = 0.99     # inversa elasticidad sustitución intertemporal (post. mean)
    xi      = 1.60     # inversa elasticidad oferta trabajo total (post. mean)
    phi     = 0.93     # inversa elasticidad oferta trabajo sumergido (post. mean)
    B_1     = 300.0    # desutilidad trabajo sumergido (post. mean)

    # B_0 es calibra numèricament per satisfer n_m_ss + n_u_ss ≈ 0.19 (19% del temps)
    # amb els paràmetres fiscals actuals. En l'estimació es fixa en aquest valor
    # ja que recalibrar-lo a cada iteració MCMC requereix resoldre l'estat estacionari.
    B_0     = 81.74    # valor per a simulació; per a dades reals recalibrar si canvien tau_*_ss

    # --- Tecnología ---
    alpha   = 0.63     # elasticidad trabajo en producción regular (post. mean)
    alpha_u = 0.66     # elasticidad trabajo en producción sumergida (post. mean)
    delta_k = 0.03     # tasa de depreciación del capital (post. mean)
    gamma_ss  = 1.00   # creixement tecnològic en l'estat estacionari (fix en 1 per al DGP simulat)
    gamma_pct = 0.20   # tendència trimestral (%) en les equacions d'observació; s'estima des de les dades

    # --- Política fiscal (medias del proceso estocástico = tax rates promedio) ---
    tau_c_ss = 0.40    # tipo corporativo medio 1982-2006
    tau_h_ss = 0.35    # tipo renta personal medio
    tau_s_ss = 0.20    # cotizaciones sociales medias

    # --- Enforcement ---
    # s calibrat a partir de la LGT espanyola (arts. 191-193): sanció greu (100%) +
    # interessos de demora (~2 anys × 3.75%) → total ~2.075, arrodonit a 1.70
    # (valor conservador; el paper original usa 1.30 per Itàlia)
    s       = 1.70

    # p_ss: probabilitat d'inspecció en estat estacionari.
    # NO és observable: s'estima com a paràmetre amb prior Beta(2,98).
    # Calibració prior (dades espanyoles, 2024):
    #   Cens empreses (INE): SL = 1.127.515 + SA = 46.763 = 1.174.278
    #   Inspeccions AEAT (Memòria 2024, actuaciones programadas): 26.749
    #   p_empíric = 26.749 / 1.174.278 ≈ 0.0228 ≈ 2%
    # → Prior Beta(2, 98): mitjana = 0.02, std ≈ 0.014
    p_ss    = 0.02     # valor per a simulació; s'estima en l'estimació bayesiana

    # --- Persistencia de los shocks (posterior means, Tabla 1) ---
    rho_a    = 0.99
    rho_b    = 0.93
    rho_x    = 0.94     # rho_I en el paper
    rho_xi_h = 0.60     # rho_H en el paper
    rho_c    = 0.96
    rho_h    = 0.99
    rho_s    = 0.94
    rho_p    = 0.95

    # --- Desviaciones estándar de los shocks (posterior means / 100, Tabla 1) ---
    # MacroModelling requiere prefijo std_ para distinguirlos de las variables shock
    var_eps_a    = 0.01^2
    var_eps_b    = 0.01^2
    var_eps_c    = 0.01^2
    var_eps_s    = 0.01^2
    var_eps_h    = 0.02^2
    var_eps_xi_h = 0.01^2
    var_eps_x    = 0.01^2
    var_eps_p    = 0.06^2

end

##############################################################################
#  NOTAS DE IMPLEMENTACIÓN
#
#  1. VARIABLES DETRENDED
#     Todas las variables con minúscula son Ŝ_t = S_t / Γ_t.
#     El progreso tecnológico γ aparece explícitamente en:
#       - La Euler (factor γ en el lado izquierdo)
#       - La ley de movimiento del capital (factor γ en el lado izquierdo)
#       - Las funciones de producción (factor (γ n)^α en vez de n^α)
#
#  2. TIMING DEL CAPITAL
#     k[0]   = k̂_t  → stock disponible en t+1 (predeterminado en t+1)
#     k[-1]  = k̂_{t-1} → stock heredado, usado en producción en t
#     k_m[0] y k_u[0] son variables de CONTROL en t (se eligen en t
#     dado k[-1]), no variables predeterminadas. Esto es consistente
#     con la eq. (10) del paper donde la división del stock es una
#     decisión del período corriente.
#
#  3. FUNCIONES DE PRODUCCIÓN
#     El paper escribe: y_m = A_t (Γ_t h_m)^α k_m^{1-α}
#     Al detrend (÷ Γ_t):  ŷ_m = A_t γ^α ĥ_m^α k̂_m^{1-α}
#     El factor γ^α se incluye explícitamente como (gamma_ss * n_m)^alpha.
#     En calibración se puede absorber en el nivel de A en ss.
#
#  4. ECUACIONES DE OBSERVACIÓN (para estimación Bayesiana)
#     El paper (ec. 18) usa 7 observables. Nosaltres en usem 5:
#       obs_c, obs_inv, obs_Gc, obs_Gh, obs_wh
#     obs_Gs s'omet per no augmentar l'espai d'estat amb G_s[-1].
#     obs_p (100*log(p_t)) es manté com a variable del model per anàlisi,
#     però NO s'inclou a obs_names en get_loglikelihood.
#
#  5. B_0
#     El paper fija B_0 endógenamente en cada iteración del MCMC para que
#     h_m_ss + h_u_ss = 0.19 (fracción del tiempo dedicada a trabajar).
#     La fórmula es: B_0 = (1-τ_h_ss)*w_m_ss * λ̂_ss / (h_ss^ξ * ξ_h_ss)
#     con ξ_h_ss = 1 (shock en ss = 1).
##############################################################################