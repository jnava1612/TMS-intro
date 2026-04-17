function get_ground_state(solver::GaussianSolver, A, B, L; kwargs...)
    error("get_ground_state not implemented for $(typeof(solver))")
end

function get_ground_state(::FullMatrixSolver, A, B, L; all_c, all_cd)
    if L > 16
        @warn "FullMatrixSolver is not efficient for L > 16. Consider using BitwiseSolver instead."
    end
    H = gaussian_hamiltonian(L, A, B, all_c, all_cd)
    e, v = eigs(H, nev=1, which=:SR)
    ψ0 = v[:, 1]
    return e[1], ψ0 ./ norm(ψ0)
end

function get_ground_state(solver::BitwiseSolver, A, B, L; ψstart=nothing)
    E0, ψ0, _ = bitwise_gaussian_groundstate(
                A, B;
                ψstart = ψstart,
                maxiter = solver.maxiter,
                tol = solver.tol
    )
    return E0, ψ0 ./ norm(ψ0)
end

function cost(params::Vector, L::Int, ψ_scar::Vector, reversal_map::Vector{Int},
              E_target::Float64, sign::Float64, solver::GaussianSolver;
              λ::Float64 = 0.0, verbose::Bool = true, all_c = nothing, all_cd = nothing, 
              ψstart = nothing, basis = nothing, H = nothing)
    
    A, B = unpack_params(params, L)
    
    E0, ψ0 = get_ground_state(solver, A, B, L; 
                              all_c=all_c, 
                              all_cd=all_cd, 
                              ψstart=ψstart)

    ψ_sym = symmetrize_state(ψ0, reversal_map, sign)
    
    olap = 2 * abs(dot(ψ_scar, ψ_sym))^2

    var = 0.0 # Placeholder for potential variance term if needed in the future

    if verbose
        println("Overlap with target scar: ", olap, " (E_target: ", E_target, ", E0: ", E0, ")")
    end

    return -olap + λ * var
end

function optimize_gaussian(ψ_scar::Vector, L::Int, E_target::Float64, reversal_map::Vector{Int},
                           sign::Float64, solver::GaussianSolver;
                           λ::Float64 = 0.0,
                           all_c = nothing, all_cd = nothing,
                           basis = nothing, H = nothing,
                           flip_first::Bool = false,
                           max_nm::Int = 500,
                           max_lbfgs::Int = 1000,
                           ψstart = nothing, verbose::Bool = true)
    
    A0, B0 = initial_params(L, flip_first)
    params0 = pack_params(A0, B0, L)

    obj(p) = cost(p, L, ψ_scar, reversal_map, E_target, sign, solver;
                  λ = λ, verbose = verbose, all_c = all_c, all_cd = all_cd, 
                  ψstart = ψstart, basis = basis, H = H)
                  
    res1 = optimize(obj, params0, NelderMead(), 
                    Optim.Options(iterations = max_nm, show_trace = true))
    
    res2 = optimize(obj, res1.minimizer, LBFGS(), 
                    Optim.Options(g_tol = 1e-12, 
                                  x_abstol = 1e-8, 
                                  f_abstol = 1e-8, 
                                  iterations = max_lbfgs, 
                                  show_trace = true))

    A_opt, B_opt = unpack_params(res2.minimizer, L)
    E_opt, ψ_opt = get_ground_state(solver, A_opt, B_opt, L; 
                                    all_c=all_c, all_cd=all_cd, ψstart=ψstart)

    ψ_opt_sym = symmetrize_state(ψ_opt, reversal_map, sign)
    olap_final = 2 * abs(dot(ψ_scar, ψ_opt_sym))^2

    verbose && println("Final overlap: ", olap_final, " | Final Gaussian energy: ", E_opt, " (target: ", E_target, ")")

    return A_opt, B_opt, ψ_opt_sym, olap_final
end

function run_optimization(cfg)
    # Extracts the scars and runs the optimization for each scar state, for a given length L 
    # and other parameters specified in cfg.
    L = cfg.L
    λ = cfg.λ
    solver = cfg.solver
    max_nm = cfg.max_nm
    max_lbfgs = cfg.max_lbfgs
    z2_threshold = cfg.z2_threshold
    ent_threshold = cfg.ent_threshold
    folder = cfg.output_dir

    z2_in_even = iseven(L ÷ 2)

    mkpath(folder)

    E, V, _, _ = scar_tower(L; z2_threshold=z2_threshold, ent_threshold=ent_threshold)

    H, basis = build_pxp_hamiltonian(L)
    P_plus, P_minus = build_projectors(basis)
    reversal_map = build_reversal_map(L)

    if solver isa FullMatrixSolver
        all_c = [full_c(j, L) for j in 1:L]
        all_cd = [op' for op in all_c]
    else
        all_c = all_cd = nothing
    end

    olaps_plus = []
    olaps_minus = []

    for (i, E_targ) in enumerate(E)
        ψ = V[:, i]

        ψ_plus = P_plus * ψ
        ψ_minus = P_minus * ψ

        ψ_plus_emb = embed_to_full(ψ_plus, basis, L)
        ψ_minus_emb = embed_to_full(ψ_minus, basis, L)

        ψ_f = embed_to_full(ψ, basis, L)
        πψ_f = similar(ψ_f)
        apply_spatial_inversion!(πψ_f, ψ_f, reversal_map)
        sign = norm(ψ_f + πψ_f) < 1e-14 ? -1.0 : 1.0

        A_p, B_p, ψ_p, olap_p = optimize_gaussian(ψ_plus_emb, L, E_targ, reversal_map, sign,
                                                  solver;
                                                  λ = λ, 
                                                  all_c = all_c, all_cd = all_cd,
                                                  basis = basis, 
                                                  H = H, 
                                                  ψstart = ψ_plus_emb,
                                                  flip_first = !z2_in_even,
                                                  max_nm = max_nm, 
                                                  max_lbfgs = max_lbfgs,
                                                  verbose = true)

        push!(olaps_plus, olap_p)
        jldsave("$(folder)/N$(L)_scar_$(i)_even.jld2"; A=A_p, B=B_p, ψ=ψ_p, overlap=olap_p)

        A_m, B_m, ψ_m, olap_m = optimize_gaussian(ψ_minus_emb, L, E_targ, reversal_map, sign,
                                                  solver;
                                                  λ = λ, 
                                                  all_c = all_c, all_cd = all_cd,
                                                  basis = basis, 
                                                  H = H, 
                                                  ψstart = ψ_minus_emb,
                                                  flip_first = z2_in_even,
                                                  max_nm = max_nm, 
                                                  max_lbfgs = max_lbfgs,
                                                  verbose = true)

        push!(olaps_minus, olap_m)
        jldsave("$(folder)/N$(L)_scar_$(i)_odd.jld2"; A=A_m, B=B_m, ψ=ψ_m, overlap=olap_m)

    end

    println("All optimizations completed.")
    println("Overlaps for parity-even sector: ", olaps_plus)
    println("Overlaps for parity-odd sector: ", olaps_minus)

    return E, olaps_plus, olaps_minus
end