# -----------------------------------------------------------------------------
# Traversal.jl
#
# Generic Cayley tree traversal.
#
# A word `[i1,...,in]` means `S_in ... S_i1`, matching the old code convention.
# -----------------------------------------------------------------------------

"""Elementwise addition for Poincaré sums."""
sum_op(x, y) = x .+ y

"""Elementwise multiplication for Schottky products."""
product_op(x, y) = x .* y

"""Convert scalar or vector input to Vector{ComplexF64}."""
as_complex_vector(x) = x isa AbstractVector ? ComplexF64.(x) : ComplexF64[x]

"""Return scalar output if the original input was scalar."""
maybe_scalar(vals, original) = original isa AbstractVector ? vals : vals[1]

"""
Default geometric tail size for current Poincare terms.
Current adaptive estimates are derived for pairs of transformed points:
    Sz = [S z1, S z2].

The size is:
    |S z1 - S z2|.

For other types of series, pass an explicit `tail_size`.
"""
function default_tail_size(Sz::Vector{ComplexF64})
    length(Sz) == 2 || throw(ArgumentError(
        "default_tail_size expects exactly two transformed points [Sz1, Sz2]; " *
        "for other series pass an explicit tail_size",
    ))

    if is_infinity(Sz[1]) || is_infinity(Sz[2])
        return Inf
    end

    return abs(Sz[1] - Sz[2])
end

"""
Stats returned by traversal when `return_stats=true`.

- `error_bound` is the raw geometric tail estimate.
- `dist2_prefactor` is `dist(u,∂F)^(-2)` for every evaluation point.
- `dist_error_bound` is `dist2_prefactor .* error_bound`.
"""
struct TraversalStats
    value::Vector{ComplexF64}
    error_bound::Float64
    dist2_prefactor::Vector{Float64}
    dist_error_bound::Vector{Float64}
    visited::Int
    skipped::Int
end

"""Do not unwrap TraversalStats; its fields already contain vectorized data."""
maybe_scalar(stats::TraversalStats, original) = stats

"""
    TraversalParameters(; z, u, term, operation=sum_op,
                         id_transform_term,
                         left_coset=Int[], right_coset=Int[],
                         reduced=false,
                         max_depth=10_000,
                         algorithm=:full,
                         eps=0.0,
                         bounds=nothing,
                         tail_size=default_tail_size,
                         return_stats=false)

Parameters for Cayley tree traversal.

Algorithms:

- `:full`: full depth-limited traversal, no estimates.
- `:bogatyrev`: visit a vertex, add its term, then cut descendants if
  `K(letter) * tail_size(Sz) < eps`.
- `:bogatyrev_prebound`: pre-estimate every child and cut the child with its whole
  descendant subtree before evaluating the term if the bound is below `eps`.

Adaptive traversal is allowed for both sums and products, as in the Fortran
code. For products, eps/error_bound are based on the geometric tail size
|Sz1-Sz2|, not on a rigorous multiplicative error estimate.
"""
struct TraversalParameters
    z::Vector{ComplexF64}                    # Parameters transformed by words, for example `[z,w]` in η_zw.
    u::Vector{ComplexF64}                    # Evaluation points, not transformed.
    term::Function                           # `term(u, Sz)`: one summand/factor as vector of length `u`.
    operation::Function                      # Usually `sum_op` or `product_op`.
    id_transform_term::Vector{ComplexF64}    # Identity contribution/factor.
    left_coset::Vector{Int}                  # Suffix exclusion for relative series.
    right_coset::Vector{Int}                 # Prefix exclusion for relative series.
    reduced::Bool                            # Old hyperelliptic symmetry reduction; normally false here.
    max_depth::Int                           # Maximum word length; 0 means identity only.
    algorithm::Symbol                        # `:full`, `:bogatyrev`, or `:bogatyrev_prebound`.
    eps::Float64                             # Cutoff for adaptive algorithms.
    bounds::Any                              # Estimate object, e.g. `Bounds`.
    tail_size::Function                      # Size of transformed parameters for tail estimates.
    return_stats::Bool                       # Return `TraversalStats` instead of only values.
end

function TraversalParameters(; z, u, term,
                              operation=sum_op,
                              id_transform_term,
                              left_coset=Int[],
                              right_coset=Int[],
                              reduced=false,
                              max_depth=10_000,
                              algorithm=:full,
                              eps=1e-5,
                              bounds=nothing,
                              tail_size=default_tail_size,
                              return_stats=false)
    zc = as_complex_vector(z)
    uc = as_complex_vector(u)
    idc = as_complex_vector(id_transform_term)

    length(idc) == length(uc) ||
        throw(ArgumentError("id_transform_term must have same length as u"))
    max_depth >= 0 ||
        throw(ArgumentError("max_depth must be non-negative"))
    algorithm in (:full, :bogatyrev, :bogatyrev_prebound) ||
        throw(ArgumentError("unknown traversal algorithm: $algorithm"))

    if algorithm != :full && eps <= 0.0
        throw(ArgumentError(
            "adaptive traversal requires eps > 0; use algorithm=:full for depth-limited traversal without pruning",
        ))
    end

    return TraversalParameters(
        zc, uc, term, operation, idc,
        Int.(left_coset), Int.(right_coset),
        Bool(reduced), Int(max_depth),
        Symbol(algorithm), Float64(eps), bounds, tail_size, Bool(return_stats),
    )
end

"""True if words are equal or inverse to each other."""
function equal_or_inverse_word(a::AbstractVector{<:Integer}, b::AbstractVector{<:Integer})
    length(a) == length(b) || return false
    aa = collect(a)
    bb = collect(b)
    return aa == bb || aa == -reverse(bb)
end

"""True if `word` starts with right-coset representative `rc`, up to inverse."""
function has_right_coset_prefix(word::Vector{Int}, rc::Vector{Int})
    isempty(rc) && return false
    length(word) < length(rc) && return false
    return equal_or_inverse_word(@view(word[1:length(rc)]), rc)
end

"""True if `word` ends with left-coset representative `lc`, up to inverse."""
function has_left_coset_suffix(word::Vector{Int}, lc::Vector{Int})
    isempty(lc) && return false
    length(word) < length(lc) && return false
    return equal_or_inverse_word(@view(word[end-length(lc)+1:end]), lc)
end

function pack_traversal_result(G::RealSchottkyGroup,
                               params::TraversalParameters,
                               result::Vector{ComplexF64},
                               err::Float64,
                               visited::Int,
                               skipped::Int)
    dist2 = boundary_dist2_prefactor(G, params.u)
    dist_err = dist2 .* err

    stats = TraversalStats(result, err, dist2, dist_err, visited, skipped)

    return params.return_stats ? stats : result
end

"""
    traverse(G, params)

Run the traversal algorithm selected by `params.algorithm`.
"""
function traverse(G::RealSchottkyGroup, params::TraversalParameters)
    if params.reduced && !G.hyperelliptic_reduction
        throw(ArgumentError(
            "reduced=true is reserved for old hyperelliptic symmetry; " *
            "this group is not marked compatible",
        ))
    end

    if params.algorithm == :full
        return traverse_full(G, params)
    elseif params.algorithm == :bogatyrev
        return traverse_bogatyrev(G, params)
    elseif params.algorithm == :bogatyrev_prebound
        return traverse_bogatyrev_prebound(G, params)
    end

    error("unreachable")
end

"""
    traverse_full(G, params)

Full depth-limited traversal of all reduced words up to `max_depth`.
No estimates, no adaptive stopping.
"""
function traverse_full(G::RealSchottkyGroup, params::TraversalParameters)
    result = copy(params.id_transform_term)
    word = Int[]
    visited = 0

    alphabet_g = alphabet(G)

    function visit!(depth::Int, Sz::Vector{ComplexF64})
        depth >= params.max_depth && return

        for j in alphabet_g
            if !isempty(word) && j == -last(word)
                continue
            end
            if params.reduced && isempty(word) && j < 0
                continue
            end

            push!(word, j)

            if has_right_coset_prefix(word, params.right_coset)
                pop!(word)
                continue
            end

            newSz = generator(G, j)(Sz)

            if !has_left_coset_suffix(word, params.left_coset)
                result .= params.operation(result, params.term(params.u, newSz))
                visited += 1
            end

            visit!(depth + 1, newSz)
            pop!(word)
        end
    end

    visit!(0, params.z)
    return pack_traversal_result(G, params, result, 0.0, visited, 0)
end

"""
    traverse_bogatyrev(G, params)

Bogatyrev-style adaptive traversal.

A vertex is evaluated first. Its descendants are skipped if

    K(letter) * tail_size(Sz) < eps.

The accumulated `error_bound` is the sum of these skipped descendant estimates.
"""
function traverse_bogatyrev(G::RealSchottkyGroup, params::TraversalParameters)
    bounds = params.bounds === nothing ? build_bounds(G) : params.bounds
    alphabet_g = alphabet(G)

    result = copy(params.id_transform_term)
    err = 0.0
    visited = 0
    skipped = 0

    # State: (current_depth, transformed_points, current_word)
    stack = Tuple{Int,Vector{ComplexF64},Vector{Int}}[]
    push!(stack, (0, params.z, Int[]))

    while !isempty(stack)
        depth, Sz, word = pop!(stack)

        depth >= params.max_depth && continue

        for j in reverse(alphabet_g)
            if !isempty(word) && j == -last(word)
                continue
            end

            if params.reduced && isempty(word) && j < 0
                continue
            end

            new_word = copy(word)
            push!(new_word, j)

            if has_right_coset_prefix(new_word, params.right_coset)
                continue
            end

            newSz = generator(G, j)(Sz)

            # The child vertex itself is evaluated here.
            if !has_left_coset_suffix(new_word, params.left_coset)
                result .= params.operation(result, params.term(params.u, newSz))
                visited += 1
            end

            # Descendant-only tail after the already evaluated child.
            tail = subtree_tail_bound(bounds, j, params.tail_size(newSz))
            next_depth = depth + 1

            if tail < params.eps
                # Good stop: the omitted descendants are smaller than eps.
                err += tail
                skipped += 1

            elseif next_depth >= params.max_depth
                # Hard stop: precision was not reached before max_depth.
                # Return current accumulated value and current accumulated error.
                err += tail
                skipped += 1

                @warn "max_depth reached before eps; returning partial traversal result" depth=next_depth tail=tail eps=params.eps word=new_word visited=visited skipped=skipped
                
                return pack_traversal_result(G, params, result, err, visited, skipped)

            else
                push!(stack, (next_depth, newSz, new_word))
            end
        end
    end

    return pack_traversal_result(G, params, result, err, visited, skipped)
end


"""
    traverse_bogatyrev_prebound(G, params)

Depth-first adaptive traversal with child pre-estimates.

For a child S_j T it estimates the whole child subtree by

    (1 + K[j]) * L[(j,t)] * |Tz - Tw|,

where t is the leftmost letter of T.
"""
function traverse_bogatyrev_prebound(G::RealSchottkyGroup, params::TraversalParameters)
    bounds = params.bounds === nothing ? build_bounds(G) : params.bounds
    alphabet_g = alphabet(G)

    result = copy(params.id_transform_term)
    err = 0.0
    visited = 0
    skipped = 0

    stack = Tuple{Int,Vector{ComplexF64},Vector{Int}}[]
    push!(stack, (0, params.z, Int[]))

    while !isempty(stack)
        depth, Sz, word = pop!(stack)

        depth >= params.max_depth && continue

        parent_left_letter = isempty(word) ? 0 : last(word)
        parent_size = params.tail_size(Sz)

        children = ordered_children_by_prebound(G, bounds, parent_left_letter)

        for j in children
            if !isempty(word) && j == -last(word)
                continue
            end

            if params.reduced && isempty(word) && j < 0
                continue
            end

            new_word = copy(word)
            push!(new_word, j)

            if has_right_coset_prefix(new_word, params.right_coset)
                continue
            end

            next_depth = depth + 1

            if parent_left_letter == 0
                # At root we cannot pre-estimate through a parent circle,
                # so we compute the child first.
                newSz = generator(G, j)(Sz)
                total_bound = child_subtree_bound(
                    bounds,
                    j,
                    params.tail_size(newSz),
                )

                if total_bound < params.eps
                    err += total_bound
                    skipped += 1
                    continue
                end

                if next_depth >= params.max_depth
                    err += total_bound
                    skipped += 1

                    @warn "max_depth reached before eps; returning partial traversal result" depth=next_depth tail=tail eps=params.eps word=new_word visited=visited skipped=skipped
                
                    return pack_traversal_result(G, params, result, err, visited, skipped)
                end

            else
                # Non-root case: pre-estimate child + its descendants.
                total_bound = child_subtree_prebound(
                    bounds,
                    j,
                    parent_left_letter,
                    parent_size,
                )

                if total_bound < params.eps
                    err += total_bound
                    skipped += 1
                    continue
                end

                if next_depth >= params.max_depth
                    err += total_bound
                    skipped += 1

                    @warn "max_depth reached before eps; returning partial traversal result" depth=next_depth tail=tail eps=params.eps word=new_word visited=visited skipped=skipped
                
                    return pack_traversal_result(G, params, result, err, visited, skipped)
                end

                newSz = generator(G, j)(Sz)
            end

            # The child vertex itself is evaluated here.
            if !has_left_coset_suffix(new_word, params.left_coset)
                result .= params.operation(result, params.term(params.u, newSz))
                visited += 1
            end

            push!(stack, (next_depth, newSz, new_word))
        end
    end

    return pack_traversal_result(G, params, result, err, visited, skipped)
end