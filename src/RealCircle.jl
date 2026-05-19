# -----------------------------------------------------------------------------
# RealCircle.jl
#
# Circles with real centers.  In the intended Schottky input they are usually
# orthogonal to the real axis, so a circle is just center + radius.
# -----------------------------------------------------------------------------

"Circle with real center and positive radius."
struct RealCircle
    center::Float64
    radius::Float64

    function RealCircle(center::Real, radius::Real)
        r = Float64(radius)
        r > 0 || throw(ArgumentError("circle radius must be positive"))
        new(Float64(center), r)
    end
end

"Real-axis endpoints of a real-centered circle."
endpoints(C::RealCircle) = (C.center - C.radius, C.center + C.radius)

"Euclidean circle membership test."
on_circle(C::RealCircle, z; tol=1e-10) = abs(abs(as_complex(z) - C.center) - C.radius) <= tol

"Closed-disk membership test."
contains_point(C::RealCircle, z; tol=1e-12) = abs(as_complex(z) - C.center) <= C.radius + tol

"True if closed disks are disjoint."
disjoint(C1::RealCircle, C2::RealCircle; tol=1e-12) = abs(C1.center - C2.center) > C1.radius + C2.radius + tol

"""Diameter of a real circle."""
circle_diameter(C::RealCircle) = 2.0 * C.radius

"""Euclidean distance between two disjoint real-axis-centered circles."""
function circle_distance(C1::RealCircle, C2::RealCircle; tol=1e-14)
    d = abs(C1.center - C2.center) - C1.radius - C2.radius

    if d <= tol
        throw(ArgumentError(
            "circles are not separated enough: distance = $d",
        ))
    end

    return d
end

"""Euclidean distance from point `u` to the boundary circle `C`."""
function distance_to_circle_boundary(C::RealCircle, u::ComplexF64)
    center = complex(C.center, 0.0)
    return abs(abs(u - center) - C.radius)
end

function circle_center_complex(C::RealCircle)
    return complex(C.center, 0.0)
end

function is_inside(C::RealCircle, z::ComplexF64; tol=1e-10)
    return abs(z - circle_center_complex(C)) < C.radius - tol
end

function is_outside(C::RealCircle, z::ComplexF64; tol=1e-10)
    return abs(z - circle_center_complex(C)) > C.radius + tol
end

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
