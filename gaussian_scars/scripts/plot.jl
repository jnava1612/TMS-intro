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

            sign = real(dot(ψ_f, πψ_f)) > 0 ? 1.0 : -1.0

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
    z2_threshs = cfg.z2_thresholds
    ent_threshs = cfg.ent_thresholds
    solver = cfg.solver
    dir = cfg.output_dir

    optimized_plus = Dict{Int, Vector}()
    optimized_minus = Dict{Int, Vector}()
    energies = Dict{Int, Vector}()
    colors, markers = size_styles(sizes)

    for L in sizes

        if rerun_optimization

            opt_cfg = OptimizationConfig(
                L = L,
                λ = 0.0,
                max_nm = 20,
                max_lbfgs = 100,
                z2_threshold = z2_threshs[L],
                ent_threshold = ent_threshs[L],
                output_dir = dir,
                solver = solver,
            )

            energies[L], optimized_plus[L], optimized_minus[L] = run_optimization(opt_cfg)

        else
            Es, _, _, _ = scar_tower(L; z2_threshold = z2_threshs[L], ent_threshold = ent_threshs[L])
            olaps_plus = Float64[]
            olaps_minus = Float64[]
            for (idx, E) in enumerate(Es)
                file_plus = joinpath(dir, "N$(L)_scar_$(idx)_odd.jld2")
                file_minus = joinpath(dir, "N$(L)_scar_$(idx)_even.jld2")

                if isfile(file_plus) && isfile(file_minus)
                    data_plus = load(file_plus)
                    data_minus = load(file_minus)
                    push!(olaps_plus, data_plus["overlap"])
                    push!(olaps_minus, data_minus["overlap"])
                else
                    println("Warning: Missing optimization results for L=$(L), scar index=$(idx). Skipping.")
                    push!(olaps_plus, 0.0)
                    push!(olaps_minus, 0.0)
                end
            end
            optimized_plus[L] = olaps_plus
            optimized_minus[L] = olaps_minus
            energies[L] = Es
        end
    
    end

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

    initial_plus = Dict{Int, Vector}()
    initial_minus = Dict{Int, Vector}()
    optimized_plus = Dict{Int, Vector}()
    optimized_minus = Dict{Int, Vector}()
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

            sign = real(dot(ψ_f, πψ_f)) > 0 ? 1.0 : -1.0

            ψ_init_plus = build_initial_sym_state(!z2_in_even, sign)
            ψ_init_minus = build_initial_sym_state(z2_in_even, sign)

            push!(olaps_plus, 2 * abs(dot(ψ_plus, ψ_init_plus))^2)
            push!(olaps_minus, 2 * abs(dot(ψ_minus, ψ_init_minus))^2)
        end

        initial_plus[L] = olaps_plus
        initial_minus[L] = olaps_minus
        energies[L] = E
        
 
        olaps_plus = Float64[]
        olaps_minus = Float64[]
        for idx in eachindex(E)
            file_plus = joinpath(dir, "N$(L)_scar_$(idx)_odd.jld2")
            file_minus = joinpath(dir, "N$(L)_scar_$(idx)_even.jld2")
                
            if isfile(file_plus) && isfile(file_minus)
                data_plus = load(file_plus)
                data_minus = load(file_minus)
                push!(olaps_plus, data_plus["overlap"])
                push!(olaps_minus, data_minus["overlap"])
            else
                println("Warning: Missing optimization results for L=$(L), scar index=$(idx). Skipping.")
                push!(olaps_plus, 0.0)
                push!(olaps_minus, 0.0)
            end
        end
        optimized_plus[L] = olaps_plus
        optimized_minus[L] = olaps_minus
    end

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
end

main()