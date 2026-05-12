# -----------------------------------------------------------------------------
# ChildrenOrder.jl
#
# Non-hyperelliptic child estimates for real Schottky groups with circles
# centered on the real axis.
# -----------------------------------------------------------------------------

"""
    derivative_bound_on_circle(M, C)

Upper bound for `|M'(z)|` on the closed disk bounded by `C`.

For `M(z)=(az+b)/(cz+d)`, use
`|M'(z)| = |ad-bc| / |cz+d|^2`.
"""
function derivative_bound_on_circle(M::Mobius, C::RealCircle; pole_tol=1e-14)
    denom_min = abs(M.c * C.center + M.d) - abs(M.c) * C.radius

    if denom_min <= pole_tol
        return Inf
    end

    return abs(mobius_det(M)) / denom_min^2
end


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
                             max_bound::Float64=1e12)
    letters = alphabet(G)
    L = Dict{Tuple{Int,Int},Float64}()

    for t in letters
        Ct = circle(G, t)

        for j in letters
            j == -t && continue
            L[(j, t)] = derivative_bound_on_circle(generator(G, j), Ct)
        end
    end

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
        maxdiff < tol && return Bounds(letters, L, K)
    end

    @warn "simple bound iteration did not reach tolerance" tol max_iter
    return Bounds(letters, L, K)
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


"""Sum of diameters of all 2g boundary circles."""
function total_circle_diameter(G::RealSchottkyGroup)
    return sum(circle_diameter(circle(G, j)) for j in alphabet(G))
end


"""
    k3_constant(G, t, γmax, diam_sum)

General K3 tail estimate for a word whose leftmost letter is `t`.

This is the non-hyperelliptic version: the diameter sum is over all 2g
boundary circles.
"""
function k3_constant(G::RealSchottkyGroup,
                     t::Int,
                     γmax::Float64,
                     diam_sum::Float64)
    Ct = circle(G, t)

    min_denom = Inf

    for j in alphabet(G)
        j == t && continue

        Cj = circle(G, j)
        δ = circle_distance(Cj, Ct)
        dj = circle_diameter(Cj)

        denom = 4.0 * δ * (1.0 + δ / dj)
        min_denom = min(min_denom, denom)
    end

    isfinite(min_denom) || throw(ArgumentError(
        "could not compute K3 denominator for letter $t",
    ))

    return ((sqrt(γmax) + 1.0) / 2.0) * diam_sum / min_denom
end


"""
    build_k3_bounds(G)

Build bounds using the generalized K3 estimate.

`K[t]` is explicit from the K3 theorem; `L[(j,t)]` is still used only for
child pre-estimates.
"""
function build_k3_bounds(G::RealSchottkyGroup)
    letters = alphabet(G)

    L = Dict{Tuple{Int,Int},Float64}()

    for t in letters
        Ct = circle(G, t)

        for j in letters
            j == -t && continue
            L[(j, t)] = derivative_bound_on_circle(generator(G, j), Ct)
        end
    end

    γ = gamma_constants(G)
    γmax = maximum(values(γ))
    diam_sum = total_circle_diameter(G)

    K = Dict{Int,Float64}()

    for t in letters
        K[t] = k3_constant(G, t, γmax, diam_sum)
    end

    return Bounds(letters, L, K)
end


"""Default bound builder used by traversal."""
build_bounds(G::RealSchottkyGroup) = build_k3_bounds(G)


"""
Descendant-only tail bound after an already evaluated vertex.

Example for η:
`∑_{S>T} |Sz-Sw| ≤ K[t] |Tz-Tw|`.
"""
subtree_tail_bound(bounds::Bounds, t::Int, size::Float64) =
    bounds.K[t] * size


"""
Child plus all descendants after the child has been transformed.

The child itself is not in the sum yet, hence the factor `(1+K)`.
"""
child_subtree_bound(bounds::Bounds, child_letter::Int, child_size::Float64) =
    (1.0 + bounds.K[child_letter]) * child_size


"""
Pre-bound for a child before evaluating it.

Uses `|S_j x - S_j y| ≤ L[(j,t)] |x-y|` and then the tail estimate `K[j]`.
"""
function child_subtree_prebound(bounds::Bounds,
                                child_letter::Int,
                                parent_left_letter::Int,
                                parent_size::Float64)
    child_letter == -parent_left_letter && return 0.0

    return (1.0 + bounds.K[child_letter]) *
           bounds.L[(child_letter, parent_left_letter)] *
           parent_size
end


"""Order children by their pre-bound, largest first."""
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
        by = j -> (1.0 + bounds.K[j]) * bounds.L[(j, parent_left_letter)],
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

"""
Children-order table for the full Lyamaev traversal.

For each parent leftmost letter t, stores children S_j T ordered by decreasing
M(j,t) = (1 + K[j]) * L[(j,t)].
"""
struct ChildOrderTable
    children::Dict{Int,Vector{OrderedChild}}
end

"""
Estimate for child subtree S_j T relative to parent T.

If T starts with S_t, then

    child subtree ≤ M(j,t) * |Tz - Tw|,

where M(j,t) = (1 + K[j]) * L[(j,t)].
"""
function child_M(bounds::Bounds, j::Int, t::Int)
    j == -t && return 0.0
    return (1.0 + bounds.K[j]) * bounds.L[(j, t)]
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

            Mjt = child_M(bounds, j, t)

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