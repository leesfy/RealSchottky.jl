# -----------------------------------------------------------------------------
# GeometryEstimates.jl
#
# Geometry-only error prefactors for Schottky traversal.
#
# Traversal returns a geometric tail estimate:
#
#     err ≈ sum_skipped |S z1 - S z2|.
#
# For Poincare sums of the form
#
#     sum_S (1/(u-Sz1) - 1/(u-Sz2)),
#
# this becomes an analytic error estimate after multiplying by
#
#     dist(u, boundary(F))^(-2).
#
# This file contains only the boundary-distance part.
# -----------------------------------------------------------------------------

"""Return all boundary circles C_±j of the Schottky fundamental domain."""
function all_circles(G::RealSchottkyGroup)
    circles = RealCircle[]

    append!(circles, G.circles_plus)
    append!(circles, G.circles_minus)

    return circles
end

"""Euclidean distance from `u` to the boundary ∂F of the fundamental domain."""
function boundary_distance(G::RealSchottkyGroup, u::ComplexF64)
    d = Inf

    for C in all_circles(G)
        d = min(d, distance_to_circle_boundary(C, u))
    end

    return d
end

"""Vectorized `boundary_distance(G,u)`."""
boundary_distance(G::RealSchottkyGroup, uvec::Vector{ComplexF64}) =
    [boundary_distance(G, u) for u in uvec]

"""Return `dist(u,∂F)^(-2)`; returns `Inf` if u is numerically on the boundary."""
function boundary_dist2_prefactor(G::RealSchottkyGroup, u::ComplexF64; tol=1e-14)
    d = boundary_distance(G, u)

    if d <= tol
        return Inf
    end

    return inv(d^2)
end

"""Vectorized `boundary_dist2_prefactor(G,u)`."""
boundary_dist2_prefactor(G::RealSchottkyGroup, uvec::Vector{ComplexF64}; tol=1e-14) =
    [boundary_dist2_prefactor(G, u; tol=tol) for u in uvec]