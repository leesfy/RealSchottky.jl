module RealSchottky

using LinearAlgebra
using Plots
# using Optim 
# using SymPy


include("Mobius.jl")
include("RealCircle.jl")
include("SchottkyGroup.jl")
# include("velikin.jl")
include("GeometryEstimates.jl")
include("ChildrenOrder.jl")
include("Traversal.jl")
include("Differentials.jl")
include("HyperellipticCompatibility.jl")
include("Plotting.jl")

export Mobius,
       RealCircle,
       RealSchottkyGroup,
       det_projective,
       compose,
       derivative,
       fixed_points,
       attracting_repelling_fixed_points,
       multiplier,
       multiplier_2,
       generator,
       circle,
       endpoints,
       contains_point,
       on_circle,
       validate_pairings,
       validate_disjoint_circles,
       side_pairing_mobius,
       apply_generator,
       apply_word,
       alphabet,
       TraversalParameters,
       traverse,
       sum_op,
       product_op,
       eta,
       eta_zero_infty,
       zeta_j,
       zeta,
       exp_int_eta,
       exp_int_zeta_j,
       int_zeta_j,
       int_zeta,
       exp_period,
       period_matrix,
       # make_fundamental_domain_to_min_maxdiam_div_mindist,
       from_fixed_points,
       from_hyperelliptic_crsigma,
       legacy_hyperelliptic_mobius,
       legacy_hyperelliptic_apply,
       TraversalStats,
       plot_circles,
       plot_pairings,
       plot_pairing_images,
       plot_orbit,
       plot_orbit_pair,
       plot_word_image,
       plot_cayley_tree,
       save_plot

end
