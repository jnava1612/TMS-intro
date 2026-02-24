using ITensors
using ITensorMPS
using HDF5
using Printf
using Statistics: mean, std
using CairoMakie
using Base.Threads

function build_O_eff(H::MPO, H2::MPO, E_targ, E_curr, a)
    b = 1.0 - a
    # Builds the effective operator O_eff = (a + b)*H2 - 2*(a*E_curr + b*E_targ)*H
    return (a + b)*H2 -2*(a*E_curr + b*E_targ)*H
end

function cost(ψ, H, H2, E_target, a)
    # Constructs the cost function C(ψ) = a*(<H^2> - <H>^2) + b*(E - E_target)^2
    # H2 is passed as an argument to avoid computing it repeatedly
    b = 1.0 - a
    E = inner(ψ', H, ψ)
    EH2 = inner(ψ', H2, ψ)
    σ = EH2 - E^2
    fold = EH2 -2*E_target*E + E_target^2
    C = a*σ + b*fold
    return C, E, σ, fold
end

function isScar(ψ, H, sites; σ_tol = 0.1)
    N = length(sites)
    E = inner(ψ', H, ψ)
    var = inner(H, ψ, H, ψ) - E^2

    S = SvN_ent(copy(ψ))
    S > log(2) * N/4 && return false

    return var < σ_tol

end

function PXP_Hamiltonian(sites)
    N = length(sites)
    ampo = OpSum()
    for j in 1:N
        prev = mod1(j - 1, N)
        next = mod1(j + 1, N)
        ampo += 1.0, "ProjDn", prev, "X", j, "ProjDn", next
    end
    return MPO(ampo, sites)
end

function Z2state(sites)
    states = [isodd(n) ? "Up" : "Dn" for n in 1:length(sites)]
    return productMPS(sites, states)
end

function SvN_ent(Ψ::MPS)
    N = length(Ψ)
    mid = N ÷ 2
    orthogonalize!(Ψ, mid)
    _, S, _ = svd(Ψ[mid], (linkind(Ψ, mid-1), siteind(Ψ, mid)))
    SV = diag(array(S))
    SV = SV[SV .> 1e-14]
    return -sum(s^2 * log(s^2) for s in SV)
end

function minimize(H, H2, prevStates, ψ0, E_target, a;
                    maxD = 200, n_iter = 30, n_sweeps = 4,
                    weight = 100.0, tol = 1e-8)

    ψ = copy(ψ0)
    normalize!(ψ)

    sweeps = Sweeps(n_sweeps)
    maxdim!(sweeps, maxD)
    cutoff!(sweeps, 1e-12)
    noise!(sweeps, 1e-8, 1e-9, 0.0)

    Cprev = Inf
    E_curr = inner(ψ', H, ψ)

    @printf("Initial energy: %.12f\n", E_curr)

    for iter in 1:n_iter
        O_eff = build_O_eff(H, H2, E_target, E_curr, a)
        if isempty(prevStates)
            _, ψ = dmrg(O_eff, ψ, sweeps; outputlevel = 0)
        else
            _, ψ = dmrg(O_eff, prevStates, ψ, sweeps; weight = weight, outputlevel = 0)
        end

        E_curr = inner(ψ', H, ψ)
        C, E, var, fold = cost(ψ, H, H2, E_target, a)
        @printf("Iter %d: E = %.12f, Var = %.2e, Fold = %.2e, Cost = %.2e\n", iter, E, var, fold, C)

        if abs(C - Cprev) < tol
            @printf("Converged after %d iterations.\n", iter)
            break
        end
        Cprev = C
    end
    return ψ
end

# Each one of the threads will run a different pair of (H2_shift, ψ)
# The ψsnap is a frozen copy of the scars that were already found,
# so that they can be used as initial states for the WUp step and is not
# modified by the threads.
function parallel_WUp(H, Id, sites, ψsnap, targets, sweeps_WUp, maxD, cut, weight)
    n = length(targets)
    ψ_wu = Vector{MPS}(undef, n)

    @threads for i in 1:n
        E_target = targets[i]
        
        H2_shift = apply(H - E_target*Id, H - E_target*Id; maxdim = maxD, cutoff = cut)

        ψ0 = Z2state(sites)

        if isempty(ψsnap)
            _, ψ = dmrg(H2_shift, ψ0, sweeps_WUp; outputlevel = 0)
        else
            _, ψ = dmrg(H2_shift, ψsnap, ψ0, sweeps_WUp; weight = weight, outputlevel = 0)
        end
        ψ_wu[i] = ψ
    end
    return ψ_wu
end

function parallel_scf(H, H2, ψ_snap, ψ_Wup, targets, a;
                    maxD = 200, n_iter = 30, n_sweeps = 4,
                    weight = 100.0, tol = 1e-8)

    n = length(targets)
    results = Vector{Tuple{MPS,Float64,Float64,Float64}}(undef, n)

    @threads for i in 1:n 
        E_target = targets[i]

        ψ = minimize(H, H2, ψ_snap, ψ_Wup[i], E_target, a;
                    maxD = maxD, n_iter = n_iter, n_sweeps = n_sweeps,
                    weight = weight, tol = tol)
        
        E_found = inner(ψ', H, ψ)
        var_found = inner(H, ψ, H, ψ) - E_found^2

        C_found, _, _, _ = cost(ψ, H, H2, E_target, a)

        results[i] = (ψ, E_found, var_found, C_found)

        @printf("Thread %d: E = %.12f, Var = %.2e, Cost = %.2e\n", threadid(), E_found, var_found, C_found)
    end

    return results
end

function merge!(ψs, Es, results, targets, H, H2, sites, a, folder;
                ΔEmin = 0.1, σ_tol = 0.1)
    
    N = length(sites)
    n = length(results)
    n_acc = 0
    unresolved = Int[]
    
    order = sortperm([results[i][2] for i in 1:n])  # Sort by energy

    for i in order
        ψ, E_found, var, C_found = results[i]

        if any(abs(E_found - Ep) < ΔEmin && var < σ_tol for Ep in Es)
            @printf("Thread %d: State too close or duplicated, skipping.\n", threadid())
            continue
        end

        if isScar(ψ, H, sites; σ_tol = σ_tol)
            @printf("Thread %d: ✓ Scar accepted: E=%.6f, Var=%.2e\n", threadid(), E_found, var)
            push!(ψs, ψ)
            push!(Es, E_found)
            f = h5open(joinpath(folder, "N$(N)_$(n_acc).h5"), "w")

            write(f, "ψ", ψ)
            write(f, "E", E_found)
            close(f)

            n_acc += 1
        else
            @printf("Thread %d: ✗ Not a scar: E=%.6f, Var=%.2e\n", threadid(), E_found, var)
            push!(unresolved, i)
        end
    end
    return n_acc, unresolved
end

function findScars()
    N = 20
    maxD = 200
    maxD_Op = 500
    cut = 1e-10

    a = 0.5

    E_min = -0.8 * N
    E_max = 0.8 * N
    n_targ = 40

    weight = 100.0
    ΔEmin = 0.1
    σ_thre = 0.1
    max_wave = 5

    folder = "data_parallel"
    mkpath(folder)

    sites = siteinds("S=1/2", N; conserve_qns = false)
    H = PXP_Hamiltonian(sites)
    H2 = apply(H, H; maxdim = maxD_Op, cutoff = cut)

    Id = OpSum()
    Id += 1.0,"Id",1
    Id = MPO(Id, sites)

    sweeps_WUp = Sweeps(5)
    maxdim!(sweeps_WUp, 10, 30, 80, 150, maxD)
    cutoff!(sweeps_WUp, 1e-8)
    noise!(sweeps_WUp, 1e-5, 1e-6, 1e-7, 1e-8, 0.0)

    Etarg = range(E_min, E_max, length=n_targ)

    ψs = MPS[]
    Es = Float64[]

    remaining_idx = collect(1:n_targ)

    for wave in 1:max_wave
        isempty(remaining_idx) && break
        targets = Etarg[remaining_idx]
        @printf("Wave %d: Running WUp for %d targets...\n", wave, length(targets))

        ψ_snap = copy(ψs)

        ψ_WUp = parallel_WUp(H, Id, sites, ψ_snap, targets, sweeps_WUp, maxD, cut, weight)

        results = parallel_scf(H, H2, ψ_snap, ψ_WUp, targets, a;
                                maxD = maxD, n_iter = 30, n_sweeps = 10,
                                weight = weight, tol = 1e-8)

        n_acc, unresolved = merge!(ψs, Es, results, targets, H, H2, sites, a, folder;
                                    ΔEmin = ΔEmin, σ_tol = σ_thre)

        @printf("Wave %d: Accepted %d scars, %d unresolved.\n", wave, n_acc, length(unresolved))

        remaining_idx = remaining_idx[unresolved]

        if n_acc == 0
            @printf("No new scars found in wave %d, stopping.\n", wave)
            break
        end
    end

    if isempty(Es)
        @printf("No scars found.\n")
    end

    perm = sortperm(Es)
    ψs = ψs[perm]
    Es = Es[perm]

    return ψs, Es
end