include(joinpath(@__DIR__, "..", "src", "GaussianScars.jl"))
using .GaussianScars
using LaTeXStrings
using LinearAlgebra
using CairoMakie
using JLD2

function size_styles(sizes)
    color_cycle = [:red, :green, :blue, :orange, :purple, :brown, :black, :cyan]
    marker_cycle = [:circle, :square, :diamond, :utriangle, :dtriangle, :star5, :cross, :xcross]

    colors = Dict(L => color_cycle[mod1(i, length(color_cycle))] for (i, L) in enumerate(sizes))
    markers = Dict(L => marker_cycle[mod1(i, length(marker_cycle))] for (i, L) in enumerate(sizes))

    return colors, markers
end

function get_initial_states(sizes, solver, z2_threshs, ent_threshs)
    initial_plus = Dict{Int, Vector}()
    initial_minus = Dict{Int, Vector}()
    energies = Dict{Int, Vector}()

    for L in sizes
        olaps_plus = []
        olaps_minus = []
        E, V, _, _ = scar_tower(L; z2_threshold = z2_threshs[L], ent_threshold = ent_threshs[L])
        _, basis = build_pxp_hamiltonian(L)
        P_plus, P_minus = build_projectors(basis)
        reversal_map = build_reversal_map(L)

        if solver isa FullMatrixSolver
            all_c = [full_c(j, L) for j in 1:(L)]
            all_cd = [op' for op in all_c]
        else
            all_c = nothing
            all_cd = nothing
        end

        function build_sym_state(flip_first::Bool, sign::Float64)
            A, B = initial_params(L, flip_first)
            _, ψ = get_ground_state(solver, A, B, L; all_c = all_c, all_cd = all_cd)
            ψ = Vector(ψ)
            return symmetrize_state(ψ, reversal_map, sign)
        end

        for i in eachindex(E)
            ψ = V[:, i]
            ψ_plus = embed_to_full(P_plus * ψ, basis, L)
            ψ_minus = embed_to_full(P_minus * ψ, basis, L)
            ψ_f = embed_to_full(ψ, basis, L)
            πψ_f = similar(ψ_f)
            apply_spatial_inversion!(πψ_f, ψ_f, reversal_map)

            sign = real(dot(ψ_f, πψ_f)) > 0 ? 1.0 : -1.0
            z2_in_even = iseven(L ÷ 2)

            ψ_init_plus = build_sym_state(!z2_in_even, sign)
            ψ_init_minus = build_sym_state(z2_in_even, sign)

            push!(olaps_plus, 2 * abs(dot(ψ_plus, ψ_init_plus))^2)
            push!(olaps_minus, 2 * abs(dot(ψ_minus, ψ_init_minus))^2)
        end
        initial_plus[L] = olaps_plus
        initial_minus[L] = olaps_minus
        energies[L] = E
    end
    return initial_plus, initial_minus, energies 
end

function get_optimized_states(cfg::PlotConfig; rerun_optimization = false, energies = Dict{Int, Vector}())
    optimized_plus = Dict{Int, Vector}()
    optimized_minus = Dict{Int, Vector}()
    isnothing(energies) && (energies = Dict{Int, Vector}())
    dir = cfg.output_dir

    for L in cfg.sizes
        if rerun_optimization
            opt_cfg = OptimizationConfig(
                L = L,
                λ = 0.0,
                max_nm = 20,
                max_lbfgs = 100,
                z2_threshold = cfg.z2_thresholds[L],
                ent_threshold = cfg.ent_thresholds[L],
                output_dir = cfg.output_dir,
                solver = cfg.solver,
            )
            energies[L], optimized_plus[L], optimized_minus[L] = run_optimization(opt_cfg)
        else
            if !haskey(energies, L)
                Es, _, _, _ = scar_tower(L; z2_threshold = cfg.z2_thresholds[L], ent_threshold = cfg.ent_thresholds[L])
                energies[L] = Es
            else
                Es = energies[L]
            end
            olaps_plus = Float64[]
            olaps_minus = Float64[]
            for idx in eachindex(Es)
                file_plus = joinpath(dir, "N$(L)_scar_$(idx)_even.jld2")
                file_minus = joinpath(dir, "N$(L)_scar_$(idx)_odd.jld2")

                olap_plus = isfile(file_plus) ? load(file_plus)["overlap"] : 0.0
                olap_minus = isfile(file_minus) ? load(file_minus)["overlap"] : 0.0
                push!(olaps_plus, olap_plus)
                push!(olaps_minus, olap_minus)
            end
            optimized_plus[L] = olaps_plus
            optimized_minus[L] = olaps_minus
            energies[L] = Es
        end
    end
    return optimized_plus, optimized_minus, energies
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

    colors, markers = size_styles(sizes)

    initial_plus, initial_minus, energies = get_initial_states(sizes, solver, z2_threshs, ent_threshs)

    fig = Figure(size = (600, 650), backgroundcolor=:white)
    
    ax1 = Axis(
        fig[1,1],
        ylabel = L"2|\langle \Psi_{\mathrm{init}}| \hat{P}_+ | \Psi_{\mathrm{scar}} \rangle|^2",
        xticklabelsvisible=false,
        xgridvisible=false,
        ygridvisible=false,
        limits=(-0.1, 10.1, 0, 1),
        yticks=latex_ticks(0:0.2:1.0),
        xticks=latex_ticks(0:2:10),
    )

    ax2 = Axis(
        fig[2,1],
        xlabel = L"E_{\mathrm{scar}}",
        ylabel = L"2|\langle \Psi_{\mathrm{init}}| \hat{P}_- | \Psi_{\mathrm{scar}} \rangle|^2",
        xgridvisible=false,
        ygridvisible=false,
        limits=(-0.1, 10.1, 0, 1),
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


function plot_optimized_states(cfg::PlotConfig = PlotConfig(); default_theme = true,
                                                               rerun_optimization = false)
    if !default_theme
        mytheme()
    end

    sizes = cfg.sizes
    dir = cfg.output_dir

    colors, markers = size_styles(sizes)

    optimized_plus, optimized_minus, energies = get_optimized_states(cfg; rerun_optimization=rerun_optimization)

    fig = Figure(size = (600, 650), backgroundcolor=:white)
    
    ax1 = ax1 = Axis(
        fig[1,1],
        ylabel = L"2|\langle \Psi_{\mathrm{opt}}| \hat{P}_+ | \Psi_{\mathrm{scar}} \rangle|^2",
        xticklabelsvisible=false,
        xgridvisible=false,
        ygridvisible=false,
        limits=(-0.1, 10.1, 0, 1),
        yticks=latex_ticks(0:0.2:1.0),
        xticks=latex_ticks(0:2:10),
    )

    ax2 = Axis(
        fig[2,1],
        xlabel = L"E_{\mathrm{scar}}",
        ylabel = L"2|\langle \Psi_{\mathrm{opt}}| \hat{P}_- | \Psi_{\mathrm{scar}} \rangle|^2",
        xgridvisible=false,
        ygridvisible=false,
        limits=(-0.1, 10.1, 0, 1),
        yticks=latex_ticks(0:0.2:1.0),
        xticks=latex_ticks(0:2:10),
    )

    legend_handles = []
    labels = LaTeXString[]

    for L in sizes
        x = energies[L]

        lines!(ax1, x, optimized_plus[L], color=colors[L], linewidth=2)
        scatter!(ax1, x, optimized_plus[L], color=colors[L], markersize=8, marker=markers[L],
                 strokewidth=1.5, strokecolor=colors[L])

        lines!(ax2, x, optimized_minus[L], color=colors[L], linewidth=2)
        scatter!(ax2, x, optimized_minus[L], color=colors[L], markersize=8, marker=markers[L],
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
    save(joinpath(dir, "optimized_overlaps.png"), fig)
end

function plot_initial_and_optimized(cfg::PlotConfig = PlotConfig(); default_theme = true)
    if !default_theme
        mytheme()
    end

    sizes = cfg.sizes
    z2_threshs = cfg.z2_thresholds
    ent_threshs = cfg.ent_thresholds
    solver = cfg.solver
    dir = cfg.output_dir

    optimized_plus = Dict{Int, Vector}()
    optimized_minus = Dict{Int, Vector}()
    energies = Dict{Int, Vector}()
    colors, markers = size_styles(sizes)

    initial_plus, initial_minus, energies = get_initial_states(sizes, solver, z2_threshs, ent_threshs)

    optimized_plus, optimized_minus, _ = get_optimized_states(cfg; rerun_optimization=false, energies = energies)

    fig = Figure(size = (1200, 650))

    ax11 = Axis(
        fig[1,1],
        ylabel = L"2|\langle \Psi_{\mathrm{init}}| \hat{P}_+ | \Psi_{\mathrm{scar}} \rangle|^2",
        xticklabelsvisible=false,
        xgridvisible=false,
        ygridvisible=false,
        limits=(-0.1, 10.1, 0, 1),
        yticks=latex_ticks(0:0.2:1.0),
        xticks=latex_ticks(0:2:10),
    )

    az21 = Axis(
        fig[2, 1],
        xlabel = L"E_{\mathrm{scar}}",
        ylabel = L"2|\langle \Psi_{\mathrm{init}}| \hat{P}_- | \Psi_{\mathrm{scar}} \rangle|^2",
        xgridvisible=false,
        ygridvisible=false,
        limits=(-0.1, 10.1, 0, 1),
        yticks=latex_ticks(0:0.2:1.0),
        xticks=latex_ticks(0:2:10),
    )

    ax12 = Axis(
        fig[1, 2],
        xticklabelsvisible=false,
        ylabel = L"2|\langle \Psi_{\mathrm{opt}}| \hat{P}_+ | \Psi_{\mathrm{scar}} \rangle|^2",
        xgridvisible=false,
        ygridvisible=false,
        limits=(-0.1, 10.1, 0, 1),
        yticks=latex_ticks(0:0.2:1.0),
        xticks=latex_ticks(0:2:10),
        yaxisposition=:right,
    )

    ax22 = Axis(
        fig[2, 2],
        xlabel = L"E_{\mathrm{scar}}",
        ylabel = L"2|\langle \Psi_{\mathrm{opt}}| \hat{P}_- | \Psi_{\mathrm{scar}} \rangle|^2",
        xgridvisible=false,
        ygridvisible=false,
        limits=(-0.1, 10.1, 0, 1),
        yticks=latex_ticks(0:0.2:1.0),
        xticks=latex_ticks(0:2:10),
        yaxisposition=:right,
    )

    legend_handles = []
    labels = LaTeXString[]

    for L in sizes
        x = energies[L]

        lines!(ax11, x, initial_plus[L], color=colors[L], linewidth=2)
        scatter!(ax11, x, initial_plus[L], color=colors[L], markersize=8, marker=markers[L],
                 strokewidth=1.5, strokecolor=colors[L])

        lines!(az21, x, initial_minus[L], color=colors[L], linewidth=2)
        scatter!(az21, x, initial_minus[L], color=colors[L], markersize=8, marker=markers[L],
                 strokewidth=1.5, strokecolor=colors[L])

        lines!(ax12, x, optimized_plus[L], color=colors[L], linewidth=2)
        scatter!(ax12, x, optimized_plus[L], color=colors[L], markersize=8, marker=markers[L],
                 strokewidth=1.5, strokecolor=colors[L])

        lines!(ax22, x, optimized_minus[L], color=colors[L], linewidth=2)
        scatter!(ax22, x, optimized_minus[L], color=colors[L], markersize=8, marker=markers[L],
                 strokewidth=1.5, strokecolor=colors[L])
        
        push!(legend_handles, MarkerElement(marker=markers[L], color=colors[L],
                                            strokecolor=:black, strokewidth=1.5, markersize=12))
        push!(labels, L"L = %$L")
    end

    Legend(fig[1, 2], legend_handles, labels, framevisible=false,
           tellheight=false, tellwidth=false, halign=:right, valign=:center,
           margin=(10, 10, 10, 10))

    hideydecorations!(ax12, label = false, ticks = false, ticklabels = false)
    hideydecorations!(ax22, label = false, ticks = false, ticklabels = false)

    ax12.yticksvisible = true
    ax12.yticklabelsvisible = true
    ax12.rightspinevisible = true

    ax22.yticksvisible = true
    ax22.yticklabelsvisible = true
    ax22.rightspinevisible = true

    colgap!(fig.layout, 20)
    rowgap!(fig.layout, 20)

    display(fig)
    save(joinpath(dir, "init+opt_overlaps.png"), fig)
end

clean_matrix(M; tol = 1e-14) = begin
    X = real.(Matrix(M))
    X[abs.(X) .< tol] .= 0.0
    return X
end

function style_matrix_axis!(ax)
    ax.xticksvisible = false
    ax.yticksvisible = false
    ax.xticklabelsvisible = false
    ax.yticklabelsvisible = false
    ax.xgridvisible = false
    ax.ygridvisible = false
    return ax
end

function plot_matrix!(ax, M; colormap, colorrange=(-1, 1))
    nrows, ncols = size(M)
    hm = heatmap!(
        ax,
        1:ncols, 1:nrows, M;
        interpolate = false,
        colormap = colormap,
        colorrange = colorrange,
    )

    xlims!(ax, 0.5, ncols + 0.5)
    ylims!(ax, nrows+0.5, 0.5)
    return hm
end

function plot_optimized_AB(A_plus, B_plus, A_minus, B_minus; 
                           default_theme = true, tol = 1e-14)
                    
    if !default_theme
        mytheme()
    end
    A_plus = clean_matrix(A_plus; tol=tol)
    B_plus = clean_matrix(B_plus; tol=tol)
    A_minus = clean_matrix(A_minus; tol=tol)
    B_minus = clean_matrix(B_minus; tol=tol)

    scale = maximum((
        maximum(abs, A_plus),
        maximum(abs, B_plus),
        maximum(abs, A_minus),
        maximum(abs, B_minus),
    ))

    A_plus ./= scale
    B_plus ./= scale
    A_minus ./= scale
    B_minus ./= scale

    # cmap = cgrad([
    #     RGBf(0.18, 0.15, 0.55),  # dark purple
    #     RGBf(0.96, 0.96, 0.96), # light gray
    #     RGBf(0.67, 0.28, 0.48),  # dark red
    # ])

    # Exact same colors from the paper, the previous ones were just slightly off
    cmap = cgrad([
        RGBf(0.21, 0.14, 0.52),  # dark purple
        RGBf(0.99, 0.99, 0.99), # light gray
        RGBf(0.60, 0.22, 0.44),  # dark red
    ])

    fig = Figure(size = (900, 700), fontsize = 24)

    ax11 = Axis(fig[1,1],
        title = L"A,\ \mathrm{parity\ sector}\ +1",
        aspect = DataAspect(),
    )
    ax12 = Axis(fig[1,2],
        title = L"B,\ \mathrm{parity\ sector}\ +1",
        aspect = DataAspect(),
    )
    ax21 = Axis(fig[2,1],
        title = L"A,\ \mathrm{parity\ sector}\ -1",
        aspect = DataAspect(),
    )
    ax22 = Axis(fig[2,2],
        title = L"B,\ \mathrm{parity\ sector}\ -1",
        aspect = DataAspect(),
    )

    for ax in (ax11, ax12, ax21, ax22)
        style_matrix_axis!(ax)
    end

    hm = plot_matrix!(ax11, A_plus; colormap=cmap)
    plot_matrix!(ax12, B_plus; colormap=cmap)
    plot_matrix!(ax21, A_minus; colormap=cmap)
    plot_matrix!(ax22, B_minus; colormap=cmap)

    Colorbar(fig[:, 3], hm;
             ticks = -1:0.2:1,
             height = Relative(0.95),)

    colgap!(fig.layout, 18)
    rowgap!(fig.layout, 18)

    return fig
end

function plot_AB_matrices(cfg::PlotConfig = PlotConfig(); default_theme = true)
    if !default_theme
        mytheme()
    end

    sizes = cfg.sizes
    dir = cfg.output_dir

    for L in sizes
        idx = 1
        while true
            file_plus = joinpath(dir, "N$(L)_scar_$(idx)_even.jld2")
            file_minus = joinpath(dir, "N$(L)_scar_$(idx)_odd.jld2")

            if !isfile(file_plus) || !isfile(file_minus)
                println("No more optimization results found for L=$(L) and idx=$(idx). Skipping AB matrix plot.")
                break
            end

            data_plus = load(file_plus)
            data_minus = load(file_minus)

            A_plus = data_plus["A"]
            B_plus = data_plus["B"]
            A_minus = data_minus["A"]
            B_minus = data_minus["B"]

            fig = plot_optimized_AB(A_plus, B_plus, A_minus, B_minus; default_theme=default_theme)
            # display(fig)
            save(joinpath(dir, "AB_matrices_N$(L)_$(idx).png"), fig)
            idx += 1
        end
    end
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
    plot_optimized_states(cfg; default_theme=false, rerun_optimization=false)
    plot_initial_and_optimized(cfg; default_theme=false)
    plot_AB_matrices(cfg; default_theme=false)
end

main()