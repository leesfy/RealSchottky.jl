# -----------------------------------------------------------------------------
# Plotting.jl
#
# Visualization helpers for real Schottky groups.
# -----------------------------------------------------------------------------

using Plots

# -----------------------------------------------------------------------------
# Basic helpers
# -----------------------------------------------------------------------------

"""Sample points on a real-axis-centered circle."""
function circle_points(C::RealCircle; n::Int=400)
    θ = range(0, 2π; length=n)
    x = C.center .+ C.radius .* cos.(θ)
    y = C.radius .* sin.(θ)
    return x, y
end

"""All boundary circles in plotting order: C_-g,...,C_-1,C_1,...,C_g."""
function plotting_circles(G::RealSchottkyGroup)
    return vcat(reverse(G.circles_minus), G.circles_plus)
end

"""Labels in the same order as `plotting_circles(G)`."""
function plotting_circle_labels(G::RealSchottkyGroup)
    labels_minus = ["C_$(-j)" for j in reverse(1:G.g)]
    labels_plus = ["C_$j" for j in 1:G.g]
    return vcat(labels_minus, labels_plus)
end

"""Automatic plot limits covering all circles."""
function circle_plot_limits(circles::Vector{RealCircle}; pad_factor::Float64=0.15)
    isempty(circles) && throw(ArgumentError("no circles to plot"))

    xmin = minimum(C.center - C.radius for C in circles)
    xmax = maximum(C.center + C.radius for C in circles)
    ymax = maximum(C.radius for C in circles)

    xspan = xmax - xmin
    yspan = 2ymax

    xpad = max(1e-12, pad_factor * max(xspan, 1.0))
    ypad = max(1e-12, pad_factor * max(yspan, 1.0))

    return (xmin - xpad, xmax + xpad), (-ymax - ypad, ymax + ypad)
end

"""Draw one circle on an existing plot."""
function plot_circle!(p, C::RealCircle;
                      label="",
                      n::Int=400,
                      fill_inside::Bool=false,
                      kwargs...)
    x, y = circle_points(C; n=n)

    if fill_inside
        plot!(p, x, y; label=label, fill=(0, 0.10), linewidth=1.5, kwargs...)
    else
        plot!(p, x, y; label=label, linewidth=1.5, kwargs...)
    end

    return p
end

"""Sample a quadratic Bezier arc from p1 to p2 with control point pc."""
function bezier_arc_points(p1::Tuple{Float64,Float64},
                           pc::Tuple{Float64,Float64},
                           p2::Tuple{Float64,Float64};
                           n::Int=120)
    t = range(0.0, 1.0; length=n)

    x = [(1-s)^2 * p1[1] + 2*(1-s)*s * pc[1] + s^2 * p2[1] for s in t]
    y = [(1-s)^2 * p1[2] + 2*(1-s)*s * pc[2] + s^2 * p2[2] for s in t]

    return x, y
end

"""Apply a reduced word [i1,...,in] as S_in ... S_i1 to points."""
function apply_word_to_points(G::RealSchottkyGroup,
                              word::Vector{Int},
                              zvec::Vector{ComplexF64})
    result = copy(zvec)

    for j in word
        result = generator(G, j).(result)
    end

    return result
end

"""Generate all reduced words grouped by level."""
function reduced_words_by_level(G::RealSchottkyGroup, depth::Int)
    depth >= 0 || throw(ArgumentError("depth must be non-negative"))

    levels = Vector{Vector{Vector{Int}}}()
    push!(levels, [Int[]])

    for _ in 1:depth
        prev = levels[end]
        current = Vector{Int}[]

        for word in prev
            for j in alphabet(G)
                if !isempty(word) && j == -last(word)
                    continue
                end

                child = copy(word)
                push!(child, j)
                push!(current, child)
            end
        end

        push!(levels, current)
    end

    return levels
end

"""Compact label for a word."""
function word_label(word::Vector{Int})
    isempty(word) && return "id"
    return join(["$j" for j in word], ",")
end

"""Key for dictionaries."""
word_key(word::Vector{Int}) = Tuple(word)

"""Default weight used in tree/orbit plots."""
function plot_word_weight(G::RealSchottkyGroup, word::Vector{Int}, zvec::Vector{ComplexF64})
    Sz = apply_word_to_points(G, word, zvec)
    return default_tail_size(Sz)
end

"""Map a positive weight to a linewidth."""
function weight_to_linewidth(weight::Float64,
                             min_weight::Float64,
                             max_weight::Float64;
                             min_lw::Float64=0.7,
                             max_lw::Float64=5.0)
    if !isfinite(weight) || weight <= 0
        return min_lw
    end

    if max_weight <= min_weight
        return (min_lw + max_lw) / 2
    end

    a = log(weight + eps())
    amin = log(min_weight + eps())
    amax = log(max_weight + eps())

    t = clamp((a - amin) / (amax - amin), 0.0, 1.0)
    return min_lw + t * (max_lw - min_lw)
end

"""Map a positive weight to a marker size."""
function weight_to_markersize(weight::Float64,
                              min_weight::Float64,
                              max_weight::Float64;
                              min_ms::Float64=3.0,
                              max_ms::Float64=8.0)
    if !isfinite(weight) || weight <= 0
        return min_ms
    end

    if max_weight <= min_weight
        return (min_ms + max_ms) / 2
    end

    a = log(weight + eps())
    amin = log(min_weight + eps())
    amax = log(max_weight + eps())

    t = clamp((a - amin) / (amax - amin), 0.0, 1.0)
    return min_ms + t * (max_ms - min_ms)
end



# -----------------------------------------------------------------------------
# Plotting helpers: finite points and bounding boxes
# -----------------------------------------------------------------------------

function is_finite_point(z::ComplexF64)
    return isfinite(real(z)) && isfinite(imag(z))
end

function finite_points(points::Vector{ComplexF64})
    return [z for z in points if is_finite_point(z)]
end

function bbox_circle(C::RealCircle)
    return (
        C.center - C.radius,
        C.center + C.radius,
        -C.radius,
        C.radius,
    )
end

function bbox_points(points::Vector{ComplexF64})
    pts = finite_points(points)

    isempty(pts) && throw(ArgumentError("no finite points to plot"))

    return (
        minimum(real.(pts)),
        maximum(real.(pts)),
        minimum(imag.(pts)),
        maximum(imag.(pts)),
    )
end

function merge_bbox(b1, b2)
    return (
        min(b1[1], b2[1]),
        max(b1[2], b2[2]),
        min(b1[3], b2[3]),
        max(b1[4], b2[4]),
    )
end

function bbox_circles(circles::Vector{RealCircle})
    isempty(circles) && throw(ArgumentError("no circles"))

    box = bbox_circle(circles[1])

    for C in circles[2:end]
        box = merge_bbox(box, bbox_circle(C))
    end

    return box
end

function bbox_to_limits(box; pad_factor::Float64=0.12)
    xmin, xmax, ymin, ymax = box

    xspan = max(xmax - xmin, 1.0)
    yspan = max(ymax - ymin, 1.0)

    xpad = pad_factor * xspan
    ypad = pad_factor * yspan

    return (xmin - xpad, xmax + xpad), (ymin - ypad, ymax + ypad)
end

function plot_real_axis!(p; xlims=nothing)
    hline!(p, [0.0]; label="", linewidth=1, linestyle=:dash)
    return p
end


# -----------------------------------------------------------------------------
# Fundamental circles and domain
# -----------------------------------------------------------------------------

"""
    plot_circles(G; kwargs...)

Plot all boundary circles C_±j.

If `show_fundamental_domain=true`, the removed disks are lightly filled, so the
visible unfilled region is the fundamental domain inside the plot window.
"""
function plot_circles(G::RealSchottkyGroup;
                      show_labels::Bool=true,
                      show_centers::Bool=true,
                      show_real_axis::Bool=true,
                      show_fundamental_domain::Bool=true,
                      n::Int=400,
                      xlims=nothing,
                      ylims=nothing,
                      title::AbstractString="Schottky circles",
                      kwargs...)
    circles = plotting_circles(G)
    labels = plotting_circle_labels(G)

    if isnothing(xlims) || isnothing(ylims)
        auto_xlims, auto_ylims = circle_plot_limits(circles)
        isnothing(xlims) && (xlims = auto_xlims)
        isnothing(ylims) && (ylims = auto_ylims)
    end

    p = plot(;
        aspect_ratio=:equal,
        legend=:outerright,
        xlabel="Re u",
        ylabel="Im u",
        title=title,
        xlims=xlims,
        ylims=ylims,
        kwargs...,
    )

    if show_real_axis
        hline!(p, [0.0]; label="", linewidth=1, linestyle=:dash)
    end

    for (C, lbl) in zip(circles, labels)
        plot_circle!(
            p,
            C;
            label=show_labels ? lbl : "",
            n=n,
            fill_inside=show_fundamental_domain,
        )

        if show_centers
            scatter!(p, [C.center], [0.0]; label="", markersize=3)
        end
    end

    return p
end


# -----------------------------------------------------------------------------
# Pairings
# -----------------------------------------------------------------------------

"""
    plot_pairings(G; kwargs...)

Plot side pairings C_-j -> C_j as visible arcs above the real axis.
"""
function plot_pairings(G::RealSchottkyGroup;
                       base_plot=nothing,
                       show_labels::Bool=true,
                       title::AbstractString="Schottky circle pairings",
                       arc_height_factor::Float64=0.22,
                       xlims=nothing,
                       ylims=nothing,
                       kwargs...)

    circles = plotting_circles(G)
    box = bbox_circles(circles)

    ymax_circle = maximum(C.radius for C in circles)

    arcs = []

    for j in 1:G.g
        Cminus = G.circles_minus[j]
        Cplus = G.circles_plus[j]

        p1 = (Float64(Cminus.center), Float64(Cminus.radius))
        p2 = (Float64(Cplus.center),  Float64(Cplus.radius))

        dx = abs(p2[1] - p1[1])

        arc_height =
            max(p1[2], p2[2]) +
            arc_height_factor * max(dx, 1.0) +
            0.75 * ymax_circle

        pc = ((p1[1] + p2[1]) / 2, arc_height)

        x, y = bezier_arc_points(p1, pc, p2)
        push!(arcs, (j, x, y, pc))

        arc_pts = ComplexF64.(x, y)
        box = merge_bbox(box, bbox_points(arc_pts))
    end

    auto_xlims, auto_ylims = bbox_to_limits(box; pad_factor=0.14)
    xlims = isnothing(xlims) ? auto_xlims : xlims
    ylims = isnothing(ylims) ? auto_ylims : ylims

    p = isnothing(base_plot) ?
        plot_circles(
            G;
            show_labels=true,
            show_fundamental_domain=false,
            title=title,
            xlims=xlims,
            ylims=ylims,
            kwargs...
        ) :
        base_plot

    plot!(p; xlims=xlims, ylims=ylims)

    for (j, x, y, pc) in arcs
        plot!(p, x, y; label="", linewidth=2.2, arrow=:arrow)

        if show_labels
            annotate!(p, pc[1], pc[2], text("S_$j", 9))
        end
    end

    return p
end

"""
    plot_pairing_images(G; j=nothing, kwargs...)

Plot source circle C_-j, image S_j(C_-j), and target circle C_j.

If the pairing is correct, the image S_j(C_-j) should lie on top of C_j.
"""
function plot_pairing_images(G::RealSchottkyGroup;
                             j=nothing,
                             n::Int=600,
                             show_all_base_circles::Bool=false,
                             title::AbstractString="Images of paired circles",
                             xlims=nothing,
                             ylims=nothing,
                             kwargs...)

    js = isnothing(j) ? collect(1:G.g) : [Int(j)]

    all_image_data = []
    box = nothing

    for k in js
        1 <= k <= G.g || throw(ArgumentError("j must be in 1:g"))

        Csrc = G.circles_minus[k]
        Ctgt = G.circles_plus[k]
        S = generator(G, k)

        xs, ys = circle_points(Csrc; n=n)
        source_pts = ComplexF64.(xs, ys)
        image_pts = finite_points(S.(source_pts))

        isempty(image_pts) && throw(ArgumentError(
            "S_$k(C_$(-k)) has no finite sampled points; pole may lie on the circle",
        ))

        push!(all_image_data, (k, Csrc, Ctgt, source_pts, image_pts))

        local_box = merge_bbox(bbox_circle(Csrc), bbox_circle(Ctgt))
        local_box = merge_bbox(local_box, bbox_points(image_pts))

        box = isnothing(box) ? local_box : merge_bbox(box, local_box)
    end

    if show_all_base_circles
        box = merge_bbox(box, bbox_circles(plotting_circles(G)))
    end

    auto_xlims, auto_ylims = bbox_to_limits(box; pad_factor=0.16)
    xlims = isnothing(xlims) ? auto_xlims : xlims
    ylims = isnothing(ylims) ? auto_ylims : ylims

    p = plot(;
        aspect_ratio=:equal,
        legend=:outerright,
        xlabel="Re u",
        ylabel="Im u",
        title=title,
        xlims=xlims,
        ylims=ylims,
        kwargs...,
    )

    plot_real_axis!(p)

    if show_all_base_circles
        for (C, lbl) in zip(plotting_circles(G), plotting_circle_labels(G))
            plot_circle!(p, C; label=lbl, linewidth=1.0, alpha=0.35)
        end
    end

    for (k, Csrc, Ctgt, source_pts, image_pts) in all_image_data
        # Source circle C_-j
        xs, ys = circle_points(Csrc; n=n)
        plot!(
            p,
            xs,
            ys;
            linestyle=:dot,
            linewidth=2.0,
            label="source C_$(-k)",
        )

        # Target circle C_j
        xt, yt = circle_points(Ctgt; n=n)
        plot!(
            p,
            xt,
            yt;
            linewidth=1.5,
            label="target C_$k",
        )

        # Image S_j(C_-j)
        plot!(
            p,
            real.(image_pts),
            imag.(image_pts);
            linestyle=:dash,
            linewidth=3.5,
            label="image S_$k(C_$(-k))",
        )

        # Mark a few sampled image points
        sample_ids = round.(Int, range(1, length(image_pts); length=min(7, length(image_pts))))
        scatter!(
            p,
            real.(image_pts[sample_ids]),
            imag.(image_pts[sample_ids]);
            markersize=3,
            label="",
        )
    end

    return p
end

# -----------------------------------------------------------------------------
# Orbits
# -----------------------------------------------------------------------------

"""
    plot_orbit(G, z; depth=3, kwargs...)

Plot the reduced-word orbit {S z : |S| <= depth}, colored by level.
"""
function plot_orbit(G::RealSchottkyGroup,
                    z;
                    depth::Int=3,
                    show_circles::Bool=true,
                    show_id::Bool=true,
                    title::AbstractString="Orbit of a point",
                    kwargs...)
    levels = reduced_words_by_level(G, depth)

    p = show_circles ?
        plot_circles(
            G;
            show_labels=false,
            show_fundamental_domain=false,
            title=title,
            kwargs...
        ) :
        plot(; aspect_ratio=:equal, title=title, xlabel="Re u", ylabel="Im u", kwargs...)

    z0 = ComplexF64(z)

    for (level, words) in enumerate(levels)
        pts = ComplexF64[]

        for word in words
            if isempty(word) && !show_id
                continue
            end
            push!(pts, apply_word_to_points(G, word, [z0])[1])
        end

        isempty(pts) && continue

        scatter!(
            p,
            real.(pts),
            imag.(pts);
            label="level $(level-1)",
            markersize=4,
        )
    end

    return p
end

"""
    plot_orbit_pair(G, z, w; depth=3, kwargs...)

Plot the pairs (S z, S w) up to a given depth and the segment joining them.
"""
function plot_orbit_pair(G::RealSchottkyGroup,
                         z, w;
                         depth::Int=3,
                         show_circles::Bool=true,
                         show_segments::Bool=true,
                         title::AbstractString="Orbit of a pair",
                         kwargs...)
    levels = reduced_words_by_level(G, depth)

    p = show_circles ?
        plot_circles(
            G;
            show_labels=false,
            show_fundamental_domain=false,
            title=title,
            kwargs...
        ) :
        plot(; aspect_ratio=:equal, title=title, xlabel="Re u", ylabel="Im u", kwargs...)

    z0 = ComplexF64(z)
    w0 = ComplexF64(w)

    for (level, words) in enumerate(levels)
        pts_z = ComplexF64[]
        pts_w = ComplexF64[]

        for word in words
            Sz = apply_word_to_points(G, word, [z0, w0])
            push!(pts_z, Sz[1])
            push!(pts_w, Sz[2])

            if show_segments
                plot!(
                    p,
                    [real(Sz[1]), real(Sz[2])],
                    [imag(Sz[1]), imag(Sz[2])];
                    linewidth=0.8,
                    alpha=0.5,
                    label="",
                )
            end
        end

        scatter!(
            p,
            real.(pts_z),
            imag.(pts_z);
            markersize=4,
            label="z, level $(level-1)",
        )

        scatter!(
            p,
            real.(pts_w),
            imag.(pts_w);
            markersize=4,
            markershape=:diamond,
            label="w, level $(level-1)",
        )
    end

    return p
end


# -----------------------------------------------------------------------------
# Images of the fundamental circles under a word
# -----------------------------------------------------------------------------

"""
    plot_word_image(G, word; kwargs...)

Plot images S(C_±j) of all fundamental circles under a word S.

`fit=:image` focuses on the image. Use `fit=:all` to include original circles too.
"""
function plot_word_image(G::RealSchottkyGroup,
                         word::Vector{Int};
                         n::Int=600,
                         show_original::Bool=false,
                         show_labels::Bool=false,
                         fit::Symbol=:image,
                         title::AbstractString="Image of fundamental circles",
                         xlims=nothing,
                         ylims=nothing,
                         kwargs...)

    fit in (:image, :all, :original) ||
        throw(ArgumentError("fit must be :image, :all, or :original"))

    circles = plotting_circles(G)
    labels = plotting_circle_labels(G)

    image_data = []
    image_box = nothing

    for (C, lbl) in zip(circles, labels)
        x, y = circle_points(C; n=n)
        pts = ComplexF64.(x, y)
        img = finite_points(apply_word_to_points(G, word, pts))

        isempty(img) && continue

        push!(image_data, (C, lbl, img))

        box = bbox_points(img)
        image_box = isnothing(image_box) ? box : merge_bbox(image_box, box)
    end

    image_box === nothing &&
        throw(ArgumentError("word image has no finite sampled points"))

    original_box = bbox_circles(circles)

    box =
        fit == :image   ? image_box :
        fit == :original ? original_box :
        merge_bbox(original_box, image_box)

    auto_xlims, auto_ylims = bbox_to_limits(box; pad_factor=0.18)
    xlims = isnothing(xlims) ? auto_xlims : xlims
    ylims = isnothing(ylims) ? auto_ylims : ylims

    p = plot(;
        aspect_ratio=:equal,
        legend=:outerright,
        xlabel="Re u",
        ylabel="Im u",
        title=title * " for word [" * word_label(word) * "]",
        xlims=xlims,
        ylims=ylims,
        kwargs...,
    )

    plot_real_axis!(p)

    if show_original
        for (C, lbl) in zip(circles, labels)
            plot_circle!(
                p,
                C;
                label=show_labels ? lbl : "",
                linewidth=1.0,
                alpha=0.35,
                linestyle=:dot,
            )
        end
    end

    for (_C, lbl, img) in image_data
        plot!(
            p,
            real.(img),
            imag.(img);
            linewidth=2.5,
            linestyle=:dash,
            label=show_labels ? "S($lbl)" : "",
        )
    end

    return p
end


# -----------------------------------------------------------------------------
# Cayley tree
# -----------------------------------------------------------------------------

"""
    plot_cayley_tree(G; depth=3, zvec=nothing, show_weights=false, show_weight_values=false, kwargs...)

Plot the reduced-word Cayley tree up to `depth`.

If `zvec` is given and `show_weights=true`, edge thickness and node size are
scaled using `default_tail_size(S(zvec))`. Numeric values are only shown if
`show_weight_values=true`.
"""
function plot_cayley_tree(G::RealSchottkyGroup;
                          depth::Int=3,
                          zvec=nothing,
                          show_word_labels::Bool=false,
                          show_weights::Bool=false,
                          show_weight_values::Bool=false,
                          title::AbstractString="Cayley tree",
                          kwargs...)
    levels = reduced_words_by_level(G, depth)

    positions = Dict{Tuple{Vararg{Int}},Tuple{Float64,Float64}}()

    for (level, words) in enumerate(levels)
        y = -(level - 1)
        n = length(words)

        for (i, word) in enumerate(words)
            x = n == 1 ? 0.0 : i - (n + 1) / 2
            positions[word_key(word)] = (x, y)
        end
    end

    p = plot(;
        legend=false,
        axis=false,
        grid=false,
        title=title,
        kwargs...,
    )

    weights = Dict{Tuple{Vararg{Int}},Float64}()

    if !isnothing(zvec)
        zc = as_complex_vector(zvec)

        for level_words in levels
            for word in level_words
                weights[word_key(word)] = plot_word_weight(G, word, zc)
            end
        end
    end

    min_weight = isempty(weights) ? 1.0 : minimum(values(weights))
    max_weight = isempty(weights) ? 1.0 : maximum(values(weights))

    # Edges
    for level in 2:length(levels)
        for word in levels[level]
            parent = word[1:end-1]

            x1, y1 = positions[word_key(parent)]
            x2, y2 = positions[word_key(word)]

            lw = 1.0
            if show_weights && haskey(weights, word_key(word))
                lw = weight_to_linewidth(weights[word_key(word)], min_weight, max_weight)
            end

            plot!(p, [x1, x2], [y1, y2]; linewidth=lw, label="")
        end
    end

    # Nodes
    for level_words in levels
        for word in level_words
            x, y = positions[word_key(word)]

            ms = 3.5
            if show_weights && haskey(weights, word_key(word))
                ms = weight_to_markersize(weights[word_key(word)], min_weight, max_weight)
            end

            scatter!(p, [x], [y]; markersize=ms, label="")
        end
    end

    # Labels: off by default to avoid clutter
    if show_word_labels || show_weight_values
        for level_words in levels
            for word in level_words
                x, y = positions[word_key(word)]
                label = ""

                if show_word_labels
                    label *= isempty(word) ? "id" : word_label(word)
                end

                if show_weight_values && haskey(weights, word_key(word))
                    label *= isempty(label) ? "" : "\n"
                    label *= string(round(weights[word_key(word)]; sigdigits=3))
                end

                !isempty(label) && annotate!(p, x, y + 0.18, text(label, 7))
            end
        end
    end

    return p
end


# -----------------------------------------------------------------------------
# IO helper
# -----------------------------------------------------------------------------

"""Save any plot object to file and return the filename."""
function save_plot(filename::AbstractString, p)
    savefig(p, filename)
    return filename
end