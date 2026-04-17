include(joinpath(@__DIR__, "..", "src", "GaussianScars.jl"))
using .GaussianScars
using BenchmarkTools
using LinearAlgebra
using Arpack
using SparseArrays

function benchmark_gaussian_hamiltonian(cfg::BenchmarkConfig = BenchmarkConfig())
    L = cfg.L
    all_c = [full_c(j, L) for j in 1:(L)]
    all_cd = [op' for op in all_c]

    A, B = initial_params(L, false)

    ψ = randn(ComplexF64, 2^L)
    ψ ./= norm(ψ)

    Hfull = gaussian_hamiltonian(L, A, B, all_c, all_cd)

    println("\nBuild full Hamiltonian:")
    display(@benchmark gaussian_hamiltonian($L, $A, $B, $all_c, $all_cd))

    println("\nSolve from prebuilt full Hamiltonian:")
    display(@benchmark eigs($Hfull, nev=1, which=:SR))

    println("\nBuild + solve full Hamiltonian:")
    display(@benchmark begin
        H = gaussian_hamiltonian($L, $A, $B, $all_c, $all_cd)
        eigs(H, nev=1, which=:SR)
    end)

    println("\nBitwise action application:")
    display(@benchmark bitwise_gaussian_action($ψ, $A, $B))

    println("\nGround state from bitwise action:")
    display(@benchmark bitwise_gaussian_groundstate($A, $B; 
            ψstart=$ψ, maxiter=$(cfg.maxiter), tol=$(cfg.tol)))
end

function main()
    cfg = BenchmarkConfig(
        L = 14,
        maxiter = 1000,
        tol = 1e-6,
    )
    benchmark_gaussian_hamiltonian(cfg)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end