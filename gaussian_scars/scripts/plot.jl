include(joinpath(@__DIR__, "..", "src", "GaussianScars.jl"))
using .GaussianScars
using LaTeXStrings
using LinearAlgebra
using CairoMakie

function size_styles(sizes)
    color_cycle = [:red, :green, :blue, :orange, :purple, :brown, :black, :cyan]
    marker_cycle = [:circle, :square, :diamond, :utriangle, :dtriangle, :star5, :cross, :xcross]

    colors = Dict(L => color_cycle[mod1(i, length(color_cycle))] for (i, L) in enumerate(sizes))
    markers = Dict(L => marker_cycle[mod1(i, length(marker_cycle))] for (i, L) in enumerate(sizes))

    return colors, markers
end

function plot_initial_states(cfg::PlotConfig = PlotConfig(); default_theme = true)
    if !default_theme
        mytheme()
    end

    sizes = cfg.sizes
    z2_threshs = cfg.z2_thresholds
    ent_threshs = cfg.ent_thresholds
    solver = cfg.solver
    dir = cfg.output_dir

    initial_plus = Dict{Int, Vector}()
    initial_minus = Dict{Int, Vector}()
    energies = Dict{Int, Vector}()
    colors, markers = size_styles(sizes)

    for L in sizes
        E, V, _, _ = scar_tower(L; z2_threshold = z2_threshs[L], ent_threshold = ent_threshs[L])
        _, basis = build_pxp_hamiltonian(L)
        P_plus, P_minus = build_projectors(basis)
        reversal_map = build_reversal_map(L)

        if solver isa FullMatrixSolver
            all_c = [full_c(j, L) for j in 1:(L)]
            all_cd = [op' for op in all_c]
        else
            all_c = [bitwise_c(j, L) for j in 1:(L)]
            all_cd = [op' for op in all_c]
        end
        
        z2_in_even = iseven(L ÷ 2)

        function build_initial_sym_state(flip_first::Bool, sign::Float64)
            A, B = initial_params(L, flip_first)
            _, ψ = get_ground_state(solver, A, B, L; all_c = all_c, all_cd = all_cd)
            ψ = Vector(ψ)
            return symmetrize_state(ψ, reversal_map, sign)
        end

        olaps_plus = []
        olaps_minus = []

        for i in eachindex(E)
            ψ = V[:, i]
            ψ_plus = embed_to_full(P_plus * ψ, basis, L)
            ψ_minus = embed_to_full(P_minus * ψ, basis, L)
            ψ_f = embed_to_full(ψ, basis, L)
            πψ_f = similar(ψ_f)
            apply_spatial_inversion!(πψ_f, ψ_f, reversal_map)

            sign = norm(ψ_f + πψ_f) < 1e-14 ? -1.0 : 1.0

            ψ_init_plus = build_initial_sym_state(!z2_in_even, sign)
            ψ_init_minus = build_initial_sym_state(z2_in_even, sign)

            push!(olaps_plus, 2 * abs(dot(ψ_plus, ψ_init_plus))^2)
            push!(olaps_minus, 2 * abs(dot(ψ_minus, ψ_init_minus))^2)
        end

        initial_plus[L] = olaps_plus
        initial_minus[L] = olaps_minus
        energies[L] = E
    end

    fig = Figure(size = (600, 650), backgroundcolor=:white)
    
    ax1 = ax1 = Axis(
        fig[1,1],
        ylabel = L"|\langle \Psi_{\mathrm{init}}| \hat{P}_+ | \Psi_{\mathrm{scar}} \rangle|^2",
        xticklabelsvisible=false,
        xgridvisible=false,
        ygridvisible=false,
        limits=(-0.1, 10.1, -0.02, 1.02),
        yticks=latex_ticks(0:0.2:1.0),
        xticks=latex_ticks(0:2:10),
    )

    ax2 = Axis(
        fig[2,1],
        xlabel = L"E_{\mathrm{scar}}",
        ylabel = L"|\langle \Psi_{\mathrm{init}}| \hat{P}_- | \Psi_{\mathrm{scar}} \rangle|^2",
        xgridvisible=false,
        ygridvisible=false,
        limits=(-0.1, 10.1, -0.02, 1.02),
        yticks=latex_ticks(0:0.2:1.0),
        xticks=latex_ticks(0:2:10),
    )

    legend_handles = []
    labels = LaTeXString[]

    for L in sizes
        x = energies[L]

        lines!(ax1, x, initial_plus[L], color=colors[L], linewidth=2)
        scatter!(ax1, x, initial_plus[L], color=colors[L], markersize=8, marker=markers[L],
                 strokewidth=1.5, strokecolor=colors[L])

        lines!(ax2, x, initial_minus[L], color=colors[L], linewidth=2)
        scatter!(ax2, x, initial_minus[L], color=colors[L], markersize=8, marker=markers[L],
                 strokewidth=1.5, strokecolor=colors[L])

        push!(legend_handles, MarkerElement(marker=markers[L], color=colors[L],
                                            strokecolor=:black, strokewidth=1.5, markersize=12))
        push!(labels, L"L = %$L")
    end

    Legend(fig[1,1], legend_handles, labels, framevisible=false,
           tellheight=false, tellwidth=false, halign=:right, valign=:center,
           margin=(10, 10, 10, 10))

    rowgap!(fig.layout, 20)
    display(fig)
    save(joinpath(dir, "initial_overlaps.png"), fig)
end


function main()
    cfg = GaussianScars.PlotConfig(
        sizes = [8, 10, 12, 14],
        solver = FullMatrixSolver(),
        z2_thresholds = Dict(8=>1e-5, 10=>1e-5, 12=>1e-5, 14=>1e-5),
        ent_thresholds = Dict(8=>0.5, 10=>0.4, 12=>0.35, 14=>0.31),
        output_dir = "../data/",
    )
    plot_initial_states(cfg; default_theme=false)
end

main()