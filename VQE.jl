using ITensors
using ITensorMPS
using Printf
using Statistics: mean
using HDF5
using CairoMakie

function build_O_eff(H::MPO, H2::MPO, E_targ, E_curr, a, b)
    # Builds the effective operator O_eff = (a + b)*H2 - 2*(a*E_curr + b*E_targ)*H
    return (a + b)*H2 -2*(a*E_curr + b*E_targ)*H
end

function cost(ψ, H, H2, E_target, a, b)
    # Constructs the cost function C(ψ) = a*(<H^2> - <H>^2) + b*(E - E_target)^2
    # H2 is passed as an argument to avoid computing it repeatedly
    E = inner(ψ', H, ψ)
    EH2 = inner(ψ', H2, ψ)
    σ = EH2 - E^2
    fold = EH2 -2*E_target*E + E_target^2
    return a*σ + b*fold, E, σ, fold
end


function minimize(H, H2, Id, prevStates, ψ0, E_target, a, b;
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
        O_eff = build_O_eff(H, H2, E_target, E_curr, a, b)
        if isempty(prevStates)
            _, ψ = dmrg(O_eff, ψ, sweeps; outputlevel = 0)
        else
            _, ψ = dmrg(O_eff, prevStates, ψ, sweeps; weight = weight, outputlevel = 0)
        end

        E_curr = inner(ψ', H, ψ)
        C, E, var, fold = cost(ψ, H, H2, E_target, a, b)
        @printf("Iter %d: E = %.12f, Var = %.2e, Fold = %.2e, Cost = %.2e\n", iter, E, var, fold, C)

        if abs(C - Cprev) < tol
            @printf("Converged after %d iterations.\n", iter)
            break
        end
        Cprev = C
    end
    return ψ
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


function findScars()
    N = 20
    maxD = 500
    cut = 1e-10

    a = 0.5
    b = 1-a

    E_min = -0.8 * N
    E_max = 0.8 * N

    n_targ = 40
    weight = 100.0
    ΔEmin = 0.1
    σ_thre = 0.1

    idx = 1

    mkpath("data2")

    sites = siteinds("S=1/2", N, conserve_qns=false)
    H = PXP_Hamiltonian(sites)
    H2 = apply(H, H; maxdim = maxD, cutoff = cut)

    Id = OpSum()
    Id += 1.0,"Id",1
    Id = MPO(Id, sites)

    Etarg = range(E_min, E_max, length=n_targ)

    ψs = MPS[]
    Es = Float64[]

    sweeps_WUp = Sweeps(5)
    maxdim!(sweeps_WUp, 10, 30, 80, 150, maxD)
    cutoff!(sweeps_WUp, 1e-8)
    noise!(sweeps_WUp, 1e-5, 1e-6, 1e-7, 1e-8, 0.0)

    for (k, E_target) in enumerate(Etarg)
        # Warmup in shifted spectrum
        H2_shift = apply(H - E_target*Id, H - E_target*Id; maxdim = maxD, cutoff = cut)

        ψ0 = Z2state(sites)

        if isempty(ψs)
            _, ψ = dmrg(H2_shift, ψ0, sweeps_WUp; outputlevel = 0)
        else
            _, ψ = dmrg(H2_shift, ψs, ψ0, sweeps_WUp; weight = weight, outputlevel = 0)
        end

        # Main optimization
        ψ = minimize(H, H2, Id, ψs, ψ, E_target, a, b; maxD = maxD, 
                    n_iter = 30, n_sweeps = 10, weight = weight, tol = 1e-8)
        E_found = inner(ψ', H, ψ)
        var = inner(ψ', H2, ψ) - E_found^2

        C_found, _, _, _ = cost(ψ, H, H2, E_target, a, b)
        @printf("E_target = %.2f, E_found = %.2f, var = %.2e, C_found = %.2e\n", E_target, E_found, var, C_found)
        
        if any(abs(E_found - Ep) < ΔEmin && var < σ_thre for Ep in Es)
            @printf("State too close to previously found state, skipping.\n")
            continue
        end

        if isScar(ψ, H, sites; σ_tol = σ_thre)
            @printf("  ✓ Scar: E=%+.6f  Var=%.2e\n", E_found, var)
            push!(ψs, ψ)
            push!(Es, E_found)
            h5open("data2/N$(N)_$(idx).h5", "w") do f
                write(f, "ψ", ψ)
                write(f, "energy",   E_found)
                write(f, "variance", var)
            end
            idx += 1
        else
            @printf("  ✗ Not a scar: E=%+.6f  Var=%.2e\n", E_found, var)
        end
    end

    if isempty(ψs)
        @printf("No scars found in the specified energy range.\n")
        return nothing, nothing
    else
        @printf("Found %d scar states in total.\n", length(ψs))
    end

    perm = sortperm(Es)
    ψs = ψs[perm]
    Es = Es[perm]

    return ψs, Es

end

function plot(nFiles, N)
    Es = Float64[]
    Z2_olap = Float64[]

    for k in 1:nFiles
        f = h5open("data2/N$(N)_$(k).h5", "r")
        E = read(f, "energy")
        ψ = read(f, "ψ", MPS)
        close(f)

        Z2 = Z2state(siteinds(ψ))
        olap = inner(ψ, Z2)
        olap = abs(olap)^2
        olap = log10(olap)

        @printf("File %d: E=%.12f, Z2_overlap=%.6f\n", k, E/N, olap)

        push!(Es, E/N)
        push!(Z2_olap, olap)
    end

    fig = Figure(size=(800, 600))
    ax = Axis(fig[1, 1], xlabel=L"E/N", ylabel = L"\log_{10} |\langle Z_2 | \Psi_n \rangle|^2")
    scatter!(ax, Es, Z2_olap, color=:blue, markersize=8)
    # ylims!(ax, -10, 0)
    xlims!(ax, -0.8, 0.8)
    return fig
end
