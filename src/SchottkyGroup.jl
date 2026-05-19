# -----------------------------------------------------------------------------
# SchottkyGroup.jl
#
# Base real Schottky group: real Mobius generators + explicit 2g circles.
# S_j maps interior C_-j to exterior of C_j
#
#   * S_-j = inv(S_j) remains: this is notation, not symmetry.
# -----------------------------------------------------------------------------

struct RealSchottkyGroup
    g::Int
    generators::Vector{Mobius}        # S_1,...,S_g
    inverses::Vector{Mobius}          # S_-1,...,S_-g
    circles_plus::Vector{RealCircle}  # C_1,...,C_g
    circles_minus::Vector{RealCircle} # C_-1,...,C_-g
    alpha_plus::Vector{ComplexF64}    # attracting fixed point of S_j
    alpha_minus::Vector{ComplexF64}   # attracting fixed point of S_j^{-1}
    multipliers::Vector{ComplexF64}
    hyperelliptic_reduction::Bool     # true only for compatibility constructor

    function RealSchottkyGroup(generators::Vector{Mobius},
                               circles_plus::Vector{RealCircle},
                               circles_minus::Vector{RealCircle};
                               validate=true,
                               check_sides=true,
                               hyperelliptic_reduction=false)
        g = length(generators)
        length(circles_plus) == g || throw(ArgumentError("circles_plus must have length g"))
        length(circles_minus) == g || throw(ArgumentError("circles_minus must have length g"))

        inverses = inv.(generators)
        αp = ComplexF64[]
        αm = ComplexF64[]
        λ = ComplexF64[]
        for M in generators
            p, m = attracting_repelling_fixed_points(M)
            push!(αp, p)
            push!(αm, m)
            push!(λ, derivative(M, p))
        end

        G = new(g, generators, inverses, circles_plus, circles_minus, αp, αm, λ, hyperelliptic_reduction)
        validate && validate_disjoint_circles(G)
        validate && validate_pairings(G; check_sides=check_sides)
        return G
    end
end

"Convenience constructor from Dict indices -g:-1,1:g."
function RealSchottkyGroup(generators::Vector{Mobius}, circles::Dict{Int,RealCircle}; kwargs...)
    g = length(generators)
    plus = [circles[j] for j in 1:g]
    minus = [circles[-j] for j in 1:g]
    RealSchottkyGroup(generators, plus, minus; kwargs...)
end

# function RealSchottkyGroup(alpha_pts, mu; kwargs...)
#     #=alpha_pts_raw = attracting_repelling_fixed_points.(generators)
#     mu_raw = [(generators[i].a - alpha_pts_raw[i][1] * generators[i].c)/(generators[i].a - alpha_pts_raw[i][2] * generators[i].c) 
#                     for i in 1:length(generators)]
#     # "by" is to be replaced by the sorting criterion asserted by crazy schizophrenic
#     indices = sort(collect(1:length(generators)), by = ->alpha_pts_raw[i][1])
#     alpha_pts = Dict(Int, Float64)
#     for i in 1:length(generators)
#         alpha_pts[i] = alpha_pts_raw[indices[i]][1]
#         alpha_pts[-i] = alpha_pts_raw[indices[i]][2]
#     end
#     mu = mu_raw[indices]
#     generators_sorted = generators[indices]=#
#     circles = make_fundamental_domain_to_min_maxdiam_div_mindist(alpha_pts, mu)
#     generators = [from_fixed_points(alpha_pts[i], alpha_pts[-i], mu[i]) for i in keys(mu)]
#     RealSchottkyGroup(generators, circles; kwargs...)
# end

alphabet(G::RealSchottkyGroup) = vcat(collect(-G.g:-1), collect(1:G.g))

generator(G::RealSchottkyGroup, j::Integer) = j > 0 ? G.generators[j] : G.inverses[-j]

circle(G::RealSchottkyGroup, j::Integer) = j > 0 ? G.circles_plus[j] : G.circles_minus[-j]

# """Return circle C_j for letter j ∈ {-g,...,-1,1,...,g}."""
# function circle(G::RealSchottkyGroup, j::Int)
#     j == 0 && throw(ArgumentError("letter 0 has no circle"))
#     return j > 0 ? G.circles_plus[j] : G.circles_minus[-j]
# end

apply_generator(G::RealSchottkyGroup, j::Integer, z) = generator(G, j)(z)

"Apply a word [i1,...,in], interpreted as S_in ... S_i1."
function apply_word(G::RealSchottkyGroup, word::AbstractVector{<:Integer}, z)
    out = z
    for j in word
        out = apply_generator(G, j, out)
    end
    out
end

"Check all 2g closed disks are pairwise disjoint."
function validate_disjoint_circles(G::RealSchottkyGroup; tol=1e-12)
    labels = alphabet(G)
    for a in 1:length(labels), b in (a+1):length(labels)
        i, j = labels[a], labels[b]
        disjoint(circle(G,i), circle(G,j); tol=tol) ||
            throw(ArgumentError("circles C_$i and C_$j are not disjoint"))
    end
    return true
end

"""
    validate_pairings(G; check_sides=false)

Basic sanity check that S_j maps the real endpoints of C_-j to the endpoints of C_j.
This does not prove that the supplied data define a Schottky group; it catches
wrong matrix/circle pairings early.

If check_sides=true, also test one point inside C_-j.  The correct side convention
is project-dependent, so this remains a weak warning-style check for now.
"""
function validate_pairings(G::RealSchottkyGroup; tol=1e-8, check_sides=false)
    for j in 1:G.g
        M = generator(G, j)
        a, b = endpoints(circle(G, -j))
        c, d = endpoints(circle(G, j))
        imgs = sort(real.([M(a), M(b)]))
        target = sort([c, d])
        maximum(abs.(imgs .- target)) <= tol ||
            throw(ArgumentError("generator S_$j does not map endpoints of C_-$j to endpoints of C_$j"))

        # Side convention:
        #   S_j maps interior(C_-j) to exterior(C_j).
        if check_sides
            ok = check_interior_maps_to_exterior(
                M,
                circle(G, -j),
                circle(G, j),
            )

            if !ok
                throw(ArgumentError(
                    "side convention failed for j=$j: S_$j does not map interior(C_-$j) to exterior(C_$j)",
                ))
            end
        end
    end
    return true
end


# function circle_center_complex(C::RealCircle)
#     return complex(C.center, 0.0)
# end

# function is_inside(C::RealCircle, z::ComplexF64; tol=1e-10)
#     return abs(z - circle_center_complex(C)) < C.radius - tol
# end

# function is_outside(C::RealCircle, z::ComplexF64; tol=1e-10)
#     return abs(z - circle_center_complex(C)) > C.radius + tol
# end

function finite_complex(z::ComplexF64)
    return isfinite(real(z)) && isfinite(imag(z))
end

"""
Check the Schottky side convention:

    S_j(interior(C_-j)) ⊂ exterior(C_j)

This is intentionally a numerical check: since a Möbius map sends circles
to circles and endpoint pairing was already checked, testing one interior
point is enough in exact arithmetic. We try several points to avoid hitting
a pole accidentally.
"""
function check_interior_maps_to_exterior(
    S::Mobius,
    C_source::RealCircle,
    C_target::RealCircle;
    tol=1e-10,
)
    c = circle_center_complex(C_source)
    r = C_source.radius

    test_points = ComplexF64[
        c + 0.5im * r,
        c - 0.5im * r,
        c + 0.25 * r + 0.25im * r,
        c - 0.25 * r + 0.25im * r,
    ]

    for p in test_points
        if !is_inside(C_source, p; tol=tol)
            continue
        end

        q = S(p)

        if !finite_complex(q)
            continue
        end

        if is_outside(C_target, q; tol=tol)
            return true
        else
            return false
        end
    end

    throw(ArgumentError(
        "could not test side convention: all test points mapped to infinity or were unusable",
    ))
end

function side_pairing_mobius(Csrc::RealCircle, Ctgt::RealCircle)
    cs, rs = Csrc.center, Csrc.radius
    ct, rt = Ctgt.center, Ctgt.radius

    # z ↦ ct - rt*rs/(z - cs)
    #
    # Matrix:
    #   (ct*z - ct*cs - rt*rs) / (z - cs)
    return Mobius(ct, -ct*cs - rt*rs, 1.0, -cs)
end

"""Sum of diameters of all 2g boundary circles."""
function total_circle_diameter(G::RealSchottkyGroup)
    return sum(circle_diameter(circle(G, j)) for j in alphabet(G))
end
