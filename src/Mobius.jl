# -----------------------------------------------------------------------------
# Mobius.jl
#
# Real 2x2 Möbius transformations.
# -----------------------------------------------------------------------------

"""
    Mobius(a,b,c,d)

Real-coefficient Möbius transformation

    z → (a*z + b)/(c*z + d).

The four coefficients are projective: multiplying them by one non-zero constant
represents the same transformation.  We only check that det = a*d-b*c is nonzero.
"""
struct Mobius
    a::Float64
    b::Float64
    c::Float64
    d::Float64

    function Mobius(a::Real, b::Real, c::Real, d::Real)
        aa, bb, cc, dd = Float64(a), Float64(b), Float64(c), Float64(d)
        Δ = aa*dd - bb*cc
        Δ == 0.0 && throw(ArgumentError("singular Mobius matrix: det = 0"))
        new(aa, bb, cc, dd)
    end
end

mobius_det(m::Mobius) = m.a*m.d - m.b*m.c

orientation_sign(M::Mobius) = sign(mobius_det(M))

is_orientation_preserving(M::Mobius) = mobius_det(M) > 0

is_orientation_reversing(M::Mobius) = mobius_det(M) < 0

const CP1_INF = Inf + 0.0im
is_infinity(z) = isinf(real(z)) || isinf(imag(z))
as_complex(z) = ComplexF64(z)

"Apply a Möbius transform to a real/complex point, including projective infinity."
function (m::Mobius)(z)
    zc = as_complex(z)
    if is_infinity(zc)
        return m.c == 0.0 ? CP1_INF : ComplexF64(m.a/m.c)
    end
    denom = m.c*zc + m.d
    denom == 0.0 && return CP1_INF
    return (m.a*zc + m.b)/denom
end

(m::Mobius)(zs::AbstractVector) = [m(z) for z in zs]

"Composition m ∘ n."
function compose(m::Mobius, n::Mobius)
    Mobius(
        m.a*n.a + m.b*n.c,
        m.a*n.b + m.b*n.d,
        m.c*n.a + m.d*n.c,
        m.c*n.b + m.d*n.d,
    )
end

Base.inv(m::Mobius) = Mobius(m.d, -m.b, -m.c, m.a)
inverse(m::Mobius) = inv(m)

"Derivative at a finite point. At infinity this is only meaningful when infinity is fixed."
function derivative(m::Mobius, z)
    zc = as_complex(z)
    Δ = mobius_det(m)
    if is_infinity(zc)
        return m.c == 0.0 ? ComplexF64(m.d/m.a) : 0.0 + 0.0im
    end
    return Δ/(m.c*zc + m.d)^2
end

"Return the two fixed points as complex numbers."
function fixed_points(m::Mobius; tol=1e-12)
    # z = (a z + b)/(c z + d) => c z^2 + (d-a) z - b = 0
    A = m.c
    B = m.d - m.a
    C = -m.b
    scale = max(abs(A), abs(B), abs(C), 1.0)

    if abs(A) <= tol*scale
        if abs(B) <= tol*scale
            throw(ArgumentError("identity/degenerated Mobius map: fixed points not defined"))
        end
        return (ComplexF64(-C/B), CP1_INF)
    end

    D = ComplexF64(B*B - 4*A*C)
    root = sqrt(D)
    return ((-B + root)/(2*A), (-B - root)/(2*A))
end

"Return (attracting fixed point of m, attracting fixed point of inv(m))."
function attracting_repelling_fixed_points(m::Mobius; tol=1e-12)
    z1, z2 = fixed_points(m; tol=tol)
    d1 = abs(derivative(m, z1))
    d2 = abs(derivative(m, z2))
    if d1 < d2
        return (z1, z2)
    elseif d2 < d1
        return (z2, z1)
    else
        throw(ArgumentError("cannot distinguish attracting and repelling fixed points; generator may be non-hyperbolic"))
    end
end

"Return Mobius struct from two fixed points and multiplier"
function from_fixed_points(α₋ᵢ, αᵢ,μ)
    μ == 0 && return Mobius(1,0,0,1)
    A = αᵢ - μ*α₋ᵢ
    B = αᵢ*α₋ᵢ*(μ-1)
    C = 1 - μ
    D = μ*αᵢ - α₋ᵢ
    Mobius(Float64(A), Float64(B), Float64(C), Float64(D))
end

"Attracting multiplier m'(alpha_plus)."
function multiplier(m::Mobius; tol=1e-12)
    αplus, _ = attracting_repelling_fixed_points(m; tol=tol)
    derivative(m, αplus)
end

"Explicit formula for multiplier from fixed points and Mobius matrix"
function multiplier_2(m::Mobius; tol=1e-12)
    α_p, α_m = attracting_repelling_fixed_points(m)
    k = (m.a - m.c * α_p) / (m.a - m.c * α_m)
    inv_k = (m.a - m.c * α_m) / (m.a - m.c * α_p)

    mult = multiplier(m)

    if abs(k - mult) < tol
        println("k == mult")
        return k
    elseif abs(inv_k - mult) < tol
        println("inv_k == mult")
        return inv_k
    else
        println("mult differs")
        return mult
    end
end