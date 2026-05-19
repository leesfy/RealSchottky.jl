# -----------------------------------------------------------------------------
# ChildrenOrder.jl
#
# Non-hyperelliptic child estimates for real Schottky groups with circles
# centered on the real axis.
# -----------------------------------------------------------------------------

"""
Estimate data used by adaptive traversal.

- `L[(j,t)]`: one-step estimate
  `|S_j x - S_j y| ≤ L[(j,t)] |x-y|`, for `x,y ∈ C_t`.
- `K[t]`: descendant tail estimate
  `∑_{S>T} |Sz-Sw| ≤ K[t] |Tz-Tw|`.
"""
struct Bounds
    letters::Vector{Int}
    L::Dict{Tuple{Int,Int},Float64}
    K::Dict{Int,Float64}
    K_pair::Dict{Tuple{Int,Int},Float64}
    estimate::Symbol
    l_estimate::Symbol
end

function Bounds(letters::Vector{Int},
                L::Dict{Tuple{Int,Int},Float64},
                K::Dict{Int,Float64};
                K_pair=Dict{Tuple{Int,Int},Float64}(),
                estimate::Symbol=:custom,
                l_estimate::Symbol=:derivative)
    return Bounds(letters, L, K, K_pair, estimate, l_estimate)
end


"""
    build_L_estimates(G; l_estimate=:derivative)

Build one-step estimates L[(j,t)].

Currently implemented:
- `:derivative`: `sup_{x∈C_t} |S_j'(x)|`.
"""
function build_L_estimates(G::RealSchottkyGroup; l_estimate::Symbol=:derivative)
    l_estimate == :derivative ||
        throw(ArgumentError("unknown L estimate: $l_estimate"))

    letters = alphabet(G)
    L = Dict{Tuple{Int,Int},Float64}()

    for t in letters
        Ct = circle(G, t)

        for j in letters
            j == -t && continue
            L[(j, t)] = derivative_bound_on_circle(generator(G, j), Ct)
        end
    end

    return L
end


"""
    build_simple_bounds(G; ...)

Simple self-consistent derivative-based estimate.

It solves `K[t] ≥ Σ_{j != -t} L[j,t]*(1+K[j])`.
This is a fallback estimate, not the main one.
"""
function build_simple_bounds(G::RealSchottkyGroup;
                             max_iter::Int=10_000,
                             tol::Float64=1e-12,
                             max_bound::Float64=1e12,
                             l_estimate::Symbol=:derivative)
    letters = alphabet(G)
    L = build_L_estimates(G; l_estimate=l_estimate)

    K = Dict(j => 0.0 for j in letters)

    for _ in 1:max_iter
        maxdiff = 0.0
        Knew = Dict{Int,Float64}()

        for t in letters
            s = 0.0

            for j in letters
                j == -t && continue
                s += L[(j, t)] * (1.0 + K[j])
            end

            !isfinite(s) && throw(ArgumentError(
                "simple estimate failed: infinite derivative bound for letter $t",
            ))

            s > max_bound && throw(ArgumentError(
                "simple estimate diverged for letter $t; use build_k3_bounds or algorithm=:full",
            ))

            Knew[t] = s
            maxdiff = max(maxdiff, abs(s - K[t]))
        end

        K = Knew

        if maxdiff < tol
            return Bounds(
                letters,
                L,
                K;
                estimate=:simple,
                l_estimate=l_estimate,
            )
        end
    end

    @warn "simple bound iteration did not reach tolerance" tol max_iter

    return Bounds(
        letters,
        L,
        K;
        estimate=:simple,
        l_estimate=l_estimate,
    )
end


"""Cyclic order of letters by centers of their circles on the real axis."""
function ordered_circle_letters(G::RealSchottkyGroup)
    letters = alphabet(G)
    return sort(letters; by = j -> circle(G, j).center)
end


"""
    gamma_constants(G)

Compute Bogatyrev diameter-contraction constants γ_j.

For every circle C_j, only its two neighbours in the cyclic real-line order
are used.
"""
function gamma_constants(G::RealSchottkyGroup)
    ordered = ordered_circle_letters(G)
    n = length(ordered)

    γ = Dict{Int,Float64}()

    for pos in 1:n
        j = ordered[pos]
        left = ordered[pos == 1 ? n : pos - 1]
        right = ordered[pos == n ? 1 : pos + 1]

        Cj = circle(G, j)
        Cl = circle(G, left)
        Cr = circle(G, right)

        dj = circle_diameter(Cj)

        γ[j] =
            (1.0 + dj / circle_distance(Cl, Cj)) *
            (1.0 + dj / circle_distance(Cr, Cj))
    end

    return γ
end

"""
    k3_h_factor(G, j, t)

The denominator factor from the K3 proof:

    h(j,t) =
        1 / (4 dist(C_j,C_t) * (1 + dist(C_j,C_t)/diam(C_j))).

Here j and t are real letters.
"""
function k3_h_factor(G::RealSchottkyGroup, j::Int, t::Int)
    j == t && return 0.0

    Cj = circle(G, j)
    Ct = circle(G, t)

    δ = circle_distance(Cj, Ct)
    dj = circle_diameter(Cj)

    return 1.0 / (4.0 * δ * (1.0 + δ / dj))
end

"""
    k3_diameter_sum_estimate(G)

Old K3 estimate for the full diameter sum:

    D_all = ((sqrt(gamma_max)+1)/2) * sum_{i in Ξ} diam(C_i).
"""
function k3_diameter_sum_estimate(G::RealSchottkyGroup)
    γ = gamma_constants(G)
    γmax = maximum(values(γ))
    diam_sum = total_circle_diameter(G)

    return ((sqrt(γmax) + 1.0) / 2.0) * diam_sum
end

"""
    k3_constant(G, t, γmax, diam_sum)

General K3 tail estimate for a word whose leftmost letter is `t`.

This is the non-hyperelliptic version: the diameter sum is over all 2g
boundary circles.
"""
# function k3_constant(G::RealSchottkyGroup,
#                      t::Int,
#                      γmax::Float64,
#                      diam_sum::Float64)
#     Ct = circle(G, t)

#     min_denom = Inf

#     for j in alphabet(G)
#         j == t && continue

#         Cj = circle(G, j)
#         δ = circle_distance(Cj, Ct)
#         dj = circle_diameter(Cj)

#         denom = 4.0 * δ * (1.0 + δ / dj)
#         min_denom = min(min_denom, denom)
#     end

#     isfinite(min_denom) || throw(ArgumentError(
#         "could not compute K3 denominator for letter $t",
#     ))

#     return ((sqrt(γmax) + 1.0) / 2.0) * diam_sum / min_denom
# end

function k3_constant(G::RealSchottkyGroup,
                     t::Int,
                     diameter_sum_estimate::Float64)
    max_h = 0.0

    for j in alphabet(G)
        j == t && continue
        max_h = max(max_h, k3_h_factor(G, j, t))
    end

    return diameter_sum_estimate * max_h
end


"""
    build_k3_bounds(G)

Build bounds using the generalized K3 estimate.

`K[t]` is explicit from the K3 theorem; `L[(j,t)]` is still used only for
child pre-estimates.
"""
function build_k3_bounds(G::RealSchottkyGroup; l_estimate::Symbol=:derivative)
    letters = alphabet(G)
    L = build_L_estimates(G; l_estimate=l_estimate)

    Dall = k3_diameter_sum_estimate(G)

    K = Dict{Int,Float64}()

    for t in letters
        # K[t] = k3_constant(G, t, γmax, diam_sum)
        K[t] = k3_constant(G, t, Dall)
    end

    return Bounds(letters, L, K; estimate=:k3, l_estimate=l_estimate)
end


"""
    build_k4_bounds(G)

Build two-index K4 estimates

    K4(j,t) = D_all * h(j,t),

where D_all is the old K3 diameter-sum estimate, but the denominator factor
h(j,t) is kept instead of taking max over j.

The one-index field K[t] is set to sum_{j != t} K4(j,t), so ordinary Bogatyrev
can also use this estimate.
"""
function build_k4_bounds(G::RealSchottkyGroup; l_estimate::Symbol=:derivative)
    letters = alphabet(G)
    L = build_L_estimates(G)

    Dall = k3_diameter_sum_estimate(G)

    K_pair = Dict{Tuple{Int,Int},Float64}()

    for t in letters
        for j in letters
            j == t && continue
            K_pair[(j, t)] = Dall * k3_h_factor(G, j, t)
        end
    end

    K = Dict{Int,Float64}()

    for t in letters
        s = 0.0

        for j in letters
            j == t && continue
            s += K_pair[(j, t)]
        end

        K[t] = s
    end

    return Bounds(
        letters,
        L,
        K;
        K_pair=K_pair,
        estimate=:k4,
        l_estimate=l_estimate,
    )
end


# function build_k4_bounds(G::RealSchottkyGroup; l_estimate::Symbol=:derivative)
#     base = build_k3_bounds(G; l_estimate=l_estimate)

#     letters = base.letters
#     Dall = k3_diameter_sum_estimate(G)

#     K_pair = Dict{Tuple{Int,Int},Float64}()

#     for t in letters
#         for child in letters
#             child == -t && continue

#             # child = -j in the notation of the proof:
#             # A ends with S_child = S_{-j}, hence j = -child.
#             K_pair[(child, t)] = Dall * k3_h_factor(G, -child, t)
#         end
#     end

#     return Bounds(
#         base.letters,
#         base.L,
#         base.K;              # keep K3 as the one-index tail estimate
#         K_pair=K_pair,
#         estimate=:k4,
#         l_estimate=l_estimate,
#     )
# end


function build_bounds(G::RealSchottkyGroup; estimate::Symbol=:k3, l_estimate::Symbol=:derivative)
    if estimate == :k3
        return build_k3_bounds(G, l_estimate=l_estimate)

    elseif estimate == :k4
        return build_k4_bounds(G, l_estimate=l_estimate)

    elseif estimate == :simple
        return build_simple_bounds(G, l_estimate=l_estimate)

    else
        throw(ArgumentError("unknown estimate: $estimate"))
    end
end

"""Tail estimate after a word starting with S_j."""
tail_K(bounds::Bounds, j::Int) = bounds.K[j]

"""
Tail estimate after T = S_j Q, where Q starts with S_t.

If K_pair[(j,t)] is absent, fall back to K[j].
"""
tail_K(bounds::Bounds, j::Int, t::Int) = get(bounds.K_pair, (j, t), bounds.K[j])


"""Tail estimate after a concrete word."""
function tail_K(bounds::Bounds, word::Vector{Int})
    isempty(word) && return 0.0

    j = last(word)

    if length(word) >= 2
        t = word[end - 1]
        return tail_K(bounds, j, t)
    else
        return tail_K(bounds, j)
    end
end

"""
Descendant-only tail bound after an already evaluated vertex.

Example for η:
`∑_{S>T} |Sz-Sw| ≤ K[t] |Tz-Tw|`.
"""
subtree_tail_bound(bounds::Bounds, word::Vector{Int}, size::Float64) =
    tail_K(bounds, word) * size


"""Backward-compatible one-letter tail bound."""
subtree_tail_bound(bounds::Bounds, j::Int, size::Float64) =
    tail_K(bounds, j) * size

# """
# Child plus all descendants after the child has been transformed.

# The child itself is not in the sum yet, hence the factor `(1+K)`.
# """
# child_subtree_bound(bounds::Bounds, child_letter::Int, child_size::Float64) =
#     (1.0 + bounds.K[child_letter]) * child_size


"""
Coefficient for the whole child subtree S_j T relative to parent T.

M(j,t) = (1 + K(j,t)) L(j,t).
"""
function child_coeff(bounds::Bounds, j::Int, t::Int)
    j == -t && return 0.0
    return (1.0 + tail_K(bounds, j, t)) * bounds.L[(j, t)]
end


# function child_coeff(bounds::Bounds, child::Int, parent::Int)
#     child == -parent && return 0.0
#     old = (1.0 + bounds.K[child]) * bounds.L[(child, parent)]

#     if bounds.estimate == :k4
#         new = bounds.K_pair[(child, parent)]
#         return min(old, new)
#     else
#         return old
#     end
# end

# function child_coeff(bounds::Bounds,
#                      child::Int,
#                      parent::Int;
#                      child_bound::Symbol=:k3L)
#     child == -parent && return 0.0

#     if child_bound == :k3L
#         return (1.0 + bounds.K[child]) * bounds.L[(child, parent)]

#     elseif child_bound == :k4
#         haskey(bounds.K_pair, (child, parent)) || throw(ArgumentError(
#             "child_bound=:k4 requires K_pair[(child,parent)]. " *
#             "Use estimate=:k4 when building bounds.",
#         ))

#         return bounds.K_pair[(child, parent)]

#     elseif child_bound == :min
#         old = (1.0 + bounds.K[child]) * bounds.L[(child, parent)]

#         new = get(bounds.K_pair, (child, parent), Inf)

#         return min(old, new)

#     else
#         throw(ArgumentError(
#             "unknown child_bound=$child_bound; expected :k3L, :k4, or :min",
#         ))
#     end
# end

"""
Pre-bound for a child before evaluating it.

Uses `|S_j x - S_j y| ≤ L[(j,t)] |x-y|` and then the tail estimate `K[j]`.
"""
function child_subtree_prebound(bounds::Bounds,
                                child_letter::Int,
                                parent_left_letter::Int,
                                parent_size::Float64)
    child_letter == -parent_left_letter && return 0.0
    return child_coeff(bounds, child_letter, parent_left_letter) * parent_size
end

"""
Child plus all descendants after the child has already been transformed.

At root there is no previous letter, so use one-index K[j].
"""
child_subtree_bound(bounds::Bounds, child_letter::Int, child_size::Float64) =
    (1.0 + tail_K(bounds, child_letter)) * child_size


"""Order children by M(j,t) pre-bound, largest first."""
function ordered_children_by_prebound(G::RealSchottkyGroup,
                                      bounds::Bounds,
                                      parent_left_letter::Int)
    letters = alphabet(G)

    if parent_left_letter == 0
        return letters
    end

    allowed = [j for j in letters if j != -parent_left_letter]

    return sort(
        allowed;
        by = j -> child_coeff(bounds, j, parent_left_letter),
        rev = true,
    )
end


# -----------------------------------------------------------------------------
# Full Lyamaev children order
# -----------------------------------------------------------------------------

"""
One child in the full Lyamaev children-order table.

- `child_index`: letter j for child S_j T.
- `child_eps`: threshold eps / M(j,t).
- `M`: estimate for this child subtree divided by parent size.
- `block_M`: sum of M for this child and all younger siblings.
- `block_count`: number of children in this block.
"""
struct OrderedChild
    child_index::Int
    child_eps::Float64
    M::Float64
    block_M::Float64
    block_count::Int
end

# struct OrderedChild
#     child_index::Int
#     child_eps::Float64
#     coeff::Float64
#     block_coeff::Float64
#     block_count::Int
# end

"""
Children-order table for the full Lyamaev traversal.

For each parent leftmost letter t, stores children S_j T ordered by decreasing
M(j,t) = (1 + K[j]) * L[(j,t)].
"""
struct ChildOrderTable
    children::Dict{Int,Vector{OrderedChild}}
end

"""
    build_child_order(G, bounds, eps)

Build full Lyamaev children order.

Children are ordered by decreasing M(j,t).  For each position k we also store
the block sum M_k + M_{k+1} + ... over this child and all younger siblings.
"""
function build_child_order(G::RealSchottkyGroup,
                           bounds::Bounds,
                           eps::Float64)
    eps > 0.0 || throw(ArgumentError("eps must be positive for child order"))

    letters = alphabet(G)
    table = Dict{Int,Vector{OrderedChild}}()

    for t in letters
        raw = Tuple{Int,Float64}[]

        for j in letters
            j == -t && continue

            Mjt = child_coeff(bounds, j, t)

            !isfinite(Mjt) && throw(ArgumentError(
                "non-finite child estimate M($j,$t) = $Mjt",
            ))

            Mjt < 0.0 && throw(ArgumentError(
                "negative child estimate M($j,$t) = $Mjt",
            ))

            push!(raw, (j, Mjt))
        end

        # Older children first: larger M means potentially larger subtree.
        sort!(raw; by = item -> item[2], rev = true)

        children = Vector{OrderedChild}(undef, length(raw))
        running_block_M = 0.0

        for k in length(raw):-1:1
            j, Mjt = raw[k]
            running_block_M += Mjt

            child_eps = Mjt == 0.0 ? Inf : eps / Mjt
            block_count = length(raw) - k + 1

            children[k] = OrderedChild(
                j,
                child_eps,
                Mjt,
                running_block_M,
                block_count,
            )
        end

        table[t] = children
    end

    return ChildOrderTable(table)
end

"""Root children order. At root there is no parent circle, so no Lyamaev prebound."""
function root_children(G::RealSchottkyGroup; reduced::Bool=false)
    letters = alphabet(G)

    if reduced
        return [j for j in letters if j > 0]
    else
        return letters
    end
end