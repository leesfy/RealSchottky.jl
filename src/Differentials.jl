# -----------------------------------------------------------------------------
# Differentials.jl
#
# Poincare series and Schottky products through traversal.
#
# Adaptive traversal is allowed for both sums and products, as in the Fortran
# code. For products, eps/error_bound are based on the geometric tail size
# |Sz1-Sz2|, not on a rigorous multiplicative error estimate.
# -----------------------------------------------------------------------------

# Coefficient 1/(u-z), with convention 1/(u-∞)=0.
function invdiff(u, z)
    is_infinity(z) && return 0.0 + 0.0im
    return inv(u - z)
end

# Vectorized coefficient 1/(u-z) for several evaluation points u.
invdiff_vec(uvec, z) = [invdiff(u, z) for u in uvec]

# Vectorized linear factor (u-z), used in Schottky products.
function linear_factor_vec(uvec, z)
    is_infinity(z) &&
        throw(ArgumentError("products with infinite z are not implemented"))
    return [u - z for u in uvec]
end

"""
    eta(G, u, z, w; max_depth, algorithm=:bogatyrev, eps=0.0, bounds=nothing, tail_size=default_tail_size)

Coefficient `η_zw(u)/du` of the normalized Abelian differential of the third kind.
"""
function eta(G::RealSchottkyGroup, u, z, w;
             max_depth,
             algorithm=:bogatyrev,
             eps=1e-6,
             bounds=nothing,
             estimate=:k3,
             l_estimate=:derivative,
             tail_size=default_tail_size,
             return_stats=false)

    uvec = as_complex_vector(u)
    zvec = as_complex_vector([z, w])

    term = (uu, Sz) -> invdiff_vec(uu, Sz[1]) .- invdiff_vec(uu, Sz[2])
    idterm = term(uvec, zvec)

    params = TraversalParameters(
        z=zvec,
        u=uvec,
        term=term,
        operation=sum_op,
        id_transform_term=idterm,
        max_depth=max_depth,
        algorithm=algorithm,
        eps=eps,
        bounds=bounds,
        estimate=estimate,
        l_estimate=l_estimate,
        tail_size=tail_size,
        return_stats=return_stats,
    )

    return maybe_scalar(traverse(G, params), u)
end

"""
    eta_zero_infty(G, u; max_depth, algorithm=:bogatyrev, eps=0.0, bounds=nothing, tail_size=default_tail_size)

Shortcut for the coefficient `η_{0,∞}(u)/du`.
"""
eta_zero_infty(G::RealSchottkyGroup, u;
               max_depth,
               algorithm=:bogatyrev,
               eps=1e-6,
               bounds=nothing,
               estimate=:k3,
               l_estimate=:derivative,
               tail_size=default_tail_size,
               return_stats=false) =
    eta(G, u, 0.0, CP1_INF;
        max_depth=max_depth,
        algorithm=algorithm,
        eps=eps,
        bounds=bounds,
        estimate=estimate,
        l_estimate=l_estimate,
        tail_size=tail_size,
        return_stats=return_stats)

"""
    zeta_j(G, j, u; max_depth, algorithm=:bogatyrev, eps=0.0, bounds=nothing, tail_size=default_tail_size)

Coefficient `ζ_j(u)/du` of the j-th normalized Abelian differential of the first kind.
"""
function zeta_j(G::RealSchottkyGroup, j::Integer, u;
                max_depth,
                algorithm=:bogatyrev,
                eps=1e-6,
                bounds=nothing,
                estimate=:k3,
                l_estimate=:derivative,
                tail_size=default_tail_size,
                return_stats=false)

    1 <= j <= G.g || throw(ArgumentError("j must be in 1:g"))

    uvec = as_complex_vector(u)
    zvec = [G.alpha_plus[j], G.alpha_minus[j]]

    term = (uu, Sz) -> invdiff_vec(uu, Sz[1]) .- invdiff_vec(uu, Sz[2])
    idterm = term(uvec, zvec)

    params = TraversalParameters(
        z=zvec,
        u=uvec,
        term=term,
        operation=sum_op,
        id_transform_term=idterm,
        right_coset=[j],
        max_depth=max_depth,
        algorithm=algorithm,
        eps=eps,
        bounds=bounds,
        estimate=estimate,
        l_estimate=l_estimate,
        tail_size=tail_size,
        return_stats=return_stats,
    )

    return maybe_scalar(traverse(G, params), u)
end

"""
    zeta(G, coeffs, u; max_depth, algorithm=:bogatyrev, eps=0.0, bounds=nothing, tail_size=default_tail_size)

Coefficient of the linear combination `sum_j coeffs[j] * ζ_j(u) / du`.
"""
function zeta(G::RealSchottkyGroup, coeffs, u;
              max_depth,
              algorithm=:bogatyrev,
              eps=1e-6,
              bounds=nothing,
              estimate=:k3,
              l_estimate=:derivative,
              tail_size=default_tail_size,
              return_stats=false)

    length(coeffs) == G.g || throw(ArgumentError("coeffs must have length g"))

    uvec = as_complex_vector(u)
    result = zeros(ComplexF64, length(uvec))

    for j in 1:G.g
        result .+= ComplexF64(coeffs[j]) .* zeta_j(
            G,
            j,
            uvec;
            max_depth=max_depth,
            algorithm=algorithm,
            eps=eps,
            bounds=bounds,
            estimate=estimate,
            l_estimate=l_estimate,
            tail_size=tail_size,
            return_stats=return_stats
        )
    end

    return maybe_scalar(result, u)
end

"""
    exp_int_eta(G, u, z, w; max_depth, algorithm=:bogatyrev, eps=0.0, bounds=nothing, tail_size=default_tail_size)

Schottky product for `exp ∫_∞^u η_zw`.

For adaptive traversal, `eps` controls the same geometric tail size as for sums.
"""
function exp_int_eta(G::RealSchottkyGroup, u, z, w;
                     max_depth,
                     algorithm=:bogatyrev,
                     eps=1e-6,
                     bounds=nothing,
                     estimate=:k3,
                     l_estimate=:derivative,
                     tail_size=default_tail_size,
                     return_stats=false)

    uvec = as_complex_vector(u)
    zvec = as_complex_vector([z, w])

    term = (uu, Sz) -> linear_factor_vec(uu, Sz[1]) ./ linear_factor_vec(uu, Sz[2])
    idterm = term(uvec, zvec)

    params = TraversalParameters(
        z=zvec,
        u=uvec,
        term=term,
        operation=product_op,
        id_transform_term=idterm,
        max_depth=max_depth,
        algorithm=algorithm,
        eps=eps,
        bounds=bounds,
        estimate=estimate,
        l_estimate=l_estimate,
        tail_size=tail_size,
        return_stats=return_stats
    )

    return maybe_scalar(traverse(G, params), u)
end

"""
    exp_int_zeta_j(G, j, u; max_depth, algorithm=:bogatyrev, eps=0.0, bounds=nothing, tail_size=default_tail_size)

Schottky product for `exp ∫_∞^u ζ_j`.

For adaptive traversal, `eps` controls the same geometric tail size as for sums.
"""
function exp_int_zeta_j(G::RealSchottkyGroup, j::Integer, u;
                        max_depth,
                        algorithm=:bogatyrev,
                        eps=1e-6,
                        bounds=nothing,
                        estimate=:k3,
                        l_estimate=:derivative,
                        tail_size=default_tail_size,
                        return_stats=false)

    1 <= j <= G.g || throw(ArgumentError("j must be in 1:g"))

    uvec = as_complex_vector(u)
    zvec = [G.alpha_plus[j], G.alpha_minus[j]]

    term = (uu, Sz) -> linear_factor_vec(uu, Sz[1]) ./ linear_factor_vec(uu, Sz[2])
    idterm = term(uvec, zvec)

    params = TraversalParameters(
        z=zvec,
        u=uvec,
        term=term,
        operation=product_op,
        id_transform_term=idterm,
        right_coset=[j],
        max_depth=max_depth,
        algorithm=algorithm,
        eps=eps,
        bounds=bounds,
        estimate=estimate,
        l_estimate=l_estimate,
        tail_size=tail_size,
        return_stats=return_stats
    )

    return maybe_scalar(traverse(G, params), u)
end

"""
    int_zeta_j(G, j, u; max_depth)

Naive branch logarithm of `exp ∫_∞^u ζ_j`.
"""
function int_zeta_j(G::RealSchottkyGroup, j::Integer, u; 
                    max_depth, 
                    algorithm=:bogatyrev, 
                    eps=1e-6, 
                    bounds=nothing, 
                    estimate=:k3,
                    l_estimate=:derivative, 
                    tail_size=default_tail_size, 
                    return_stats=false)

    uvec = as_complex_vector(u)
    vals = exp_int_zeta_j(
        G, 
        j, 
        uvec; 
        max_depth=max_depth, 
        algorithm=algorithm, 
        eps=eps, 
        bounds=bounds, 
        estimate=estimate,
        l_estimate=l_estimate, 
        tail_size=tail_size, 
        return_stats=return_stats
    )

    result = log.(vals)
    return maybe_scalar(result, u)
end

"""
    int_zeta(G, coeffs, u; max_depth)

Naive branch integral `∫_∞^u sum_j coeffs[j] ζ_j`.
"""
function int_zeta(G::RealSchottkyGroup, coeffs, u; 
                  max_depth, 
                  algorithm=:bogatyrev, 
                  eps=1e-6, 
                  bounds=nothing, 
                  estimate=:k3,
                  l_estimate=:derivative, 
                  tail_size=default_tail_size, 
                  return_stats=false)

    length(coeffs) == G.g || throw(ArgumentError("coeffs must have length g"))

    uvec = as_complex_vector(u)
    result = zeros(ComplexF64, length(uvec))

    for j in 1:G.g
        vals = exp_int_zeta_j(G, j, uvec; max_depth=max_depth, algorithm=algorithm, eps=eps, bounds=bounds, estimate=estimate,
        l_estimate=l_estimate, tail_size=tail_size, return_stats=return_stats)
        result .+= ComplexF64(coeffs[j]) .* log.(vals)
    end

    return maybe_scalar(result, u)
end

# Cross-ratio factor in the Schottky product for periods.
function cross_ratio_period_factor(αlp, αlm, βp, βm)
    return ((βp - αlp) * (βm - αlm)) /
           ((βp - αlm) * (βm - αlp))
end

"""
    exp_period(G, l, j; max_depth, algorithm=:bogatyrev, eps=0.0, bounds=nothing, tail_size=default_tail_size)

Schottky product for `exp ∫_{b_j} ζ_l`.

For adaptive traversal, `eps` controls the same geometric tail size as for sums.
"""
function exp_period(G::RealSchottkyGroup, l::Integer, j::Integer;
                    max_depth,
                    algorithm=:bogatyrev,
                    eps=1e-6,
                    bounds=nothing,
                    estimate=:k3,
                    l_estimate=:derivative,
                    tail_size=default_tail_size,
                    return_stats=false)

    1 <= l <= G.g || throw(ArgumentError("l must be in 1:g"))
    1 <= j <= G.g || throw(ArgumentError("j must be in 1:g"))

    αlp = G.alpha_plus[l]
    αlm = G.alpha_minus[l]

    zvec = [G.alpha_plus[j], G.alpha_minus[j]]
    dummy_u = [0.0 + 0.0im]

    term = (_uu, Sz) -> [cross_ratio_period_factor(αlp, αlm, Sz[1], Sz[2])]
    idterm = l == j ? [G.multipliers[j]] : term(dummy_u, zvec)

    params = TraversalParameters(
        z=zvec,
        u=dummy_u,
        term=term,
        operation=product_op,
        id_transform_term=idterm,
        left_coset=[l],
        right_coset=[j],
        max_depth=max_depth,
        algorithm=algorithm,
        eps=eps,
        bounds=bounds,
        estimate=estimate,
        l_estimate=l_estimate,
        tail_size=tail_size,
        return_stats=return_stats
    )

    return traverse(G, params)[1]
end

"""
    period_matrix(G; max_depth, algorithm=:bogatyrev, eps=0.0, bounds=nothing, tail_size=default_tail_size)

Matrix `{∫_{b_j} ζ_l}_{l,j=1}^g`, computed as `log(exp_period)`.
"""
function period_matrix(G::RealSchottkyGroup;
                       max_depth,
                       algorithm=:bogatyrev,
                       eps=1e-6,
                       bounds=nothing,
                       estimate=:k3,
                       l_estimate=:derivative,
                       tail_size=default_tail_size,
                       return_stats=false)

    Ω = Matrix{ComplexF64}(undef, G.g, G.g)

    for l in 1:G.g, j in 1:G.g
        Ω[l, j] = log(exp_period(
            G,
            l,
            j;
            max_depth=max_depth,
            algorithm=algorithm,
            eps=eps,
            bounds=bounds,
            estimate=estimate,
            l_estimate=l_estimate,
            tail_size=tail_size,
            return_stats=return_stats
        ))
    end

    return Ω
end