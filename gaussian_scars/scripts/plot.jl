include(joinpath(@__DIR__, "..", "src", "GaussianScars.jl"))
using .GaussianScars

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

    initial_plus = Dict{Int, Vector}()
    initial_minus = Dict{Int, Vector}()
    energies = Dict{Int, Vector}()
    colors, markers = size_styles(sizes)

    for (L_i, L) in enumerate(sizes)
        E, V, _, _ = scar_tower(L; z2_threshold = z2_threshs[L_i], ent_threshold = ent_threshs[L_i])
        H, basis = build_pxp_hamiltonian(L)
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
            return symmetrize_state(ψ, L, reversal_map, sign)
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
        limits=(0, 10, 0, 1.0),
        yticks=0:0.2:1.0,
    )

    ax2 = Axis(
        fig[2,1],
        xlabel = L"E_{\mathrm{scar}}",
        ylabel = L"|\langle \Psi_{\mathrm{init}}| \hat{P}_- | \Psi_{\mathrm{scar}} \rangle|^2",
        xgridvisible=false,
        ygridvisible=false,
        limits=(0, 10, 0, 1.0),
        yticks=0:0.2:1.0,
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

    rowgap!(fig.layout, 8)
    display(fig)
    save(cfg.output_file, fig)
end