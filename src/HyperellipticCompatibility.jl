# -----------------------------------------------------------------------------
# HyperellipticCompatibility.jl
#
# The only place where the old c,r,sigma parametrization appears.
# Use this only for tests/comparison with the old Fortran code.
# -----------------------------------------------------------------------------

"Old special generator S(u) = c - sigma*r^2/(u+c), represented as a Mobius matrix."
function legacy_hyperelliptic_mobius(c::Real, r::Real, sigma::Integer=1)
    Mobius(c, c*c - sigma*r*r, 1.0, c)
end

"Build the old symmetric hyperelliptic test case from c,r,sigma."
function from_hyperelliptic_crsigma(c::AbstractVector{<:Real},
                                    r::AbstractVector{<:Real},
                                    sigma::AbstractVector{<:Integer}=ones(Int, length(c));
                                    validate=false)
    g = length(c)
    length(r) == g || throw(ArgumentError("r must have length g"))
    length(sigma) == g || throw(ArgumentError("sigma must have length g"))

    gens = [legacy_hyperelliptic_mobius(c[j], r[j], sigma[j]) for j in 1:g]
    circles_plus = [RealCircle(c[j], r[j]) for j in 1:g]
    circles_minus = [RealCircle(-c[j], r[j]) for j in 1:g]

    RealSchottkyGroup(gens, circles_plus, circles_minus;
        validate=validate,
        check_sides=false,
        hyperelliptic_reduction=true)
end

"Apply old special generator or inverse.  Used in tests."
function legacy_hyperelliptic_apply(c::AbstractVector, r::AbstractVector, sigma::AbstractVector, j::Integer, z)
    jj = abs(j)
    1 <= jj <= length(c) || throw(ArgumentError("generator index out of range"))
    m = legacy_hyperelliptic_mobius(c[jj], r[jj], sigma[jj])
    j > 0 ? m(z) : inv(m)(z)
end
