using ITensors
using ITensorMPS
using Printf
using Statistics: mean
using HDF5
using CairoMakie

function PXP_Hamiltonian(sites)
    # This code considers only the periodic case
    N = length(sites)
    ampo = OpSum()
    for j in 1:N
        prev = mod1(j - 1, N)
        next = mod1(j + 1, N)
        ampo += 1.0, "ProjDn", prev, "X", j, "ProjDn", next
    end
    return MPO(ampo, sites)
end

function IdOp(sites)
    ampo = OpSum()
    for j in 1:length(sites)
        ampo += 1.0, "Id", j
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
    ψc = copy(Ψ)
    orthogonalize!(ψc, mid)
    linds = mid > 1 ? (linkind(ψc, mid-1), siteind(ψc, mid)) : (siteind(ψc, mid),)
    _, S, _ = svd(ψc[mid], linds)
    SV = [S[i,i] for i in 1:ITensors.dim(S,1)]
    SV = SV[SV .> 1e-14]
    return -sum(s^2 * log(s^2) for s in SV)
end

function restrictHS(ψ::MPS, sites, maxD::Int)
    # Restricts the Hilbert space by removing the "Up-Up" component on neighboring sites
    N = length(sites)
    ψ0 = copy(ψ)
    orthogonalize!(ψ0, 1)

    for j in 1:N-1
        ϕ= ψ0[j] * ψ0[j+1]
        ϕ0 = copy(ϕ)

        ϕ1 = mapprime(ϕ0*op("ProjUp", sites[j]), 1 => 0)
        ϕ2 = mapprime(ϕ1*op("ProjUp", sites[j+1]), 1 => 0)

        # Removing the "Up-Up" component
        ϕ3 = ϕ0 - ϕ2

        spec = replacebond!(ψ0, j, ϕ3; maxdim = maxD,
                            mindim =1,
                            cutoff = 1e-15,
                            eigen_perturbation = nothing,
                            ortho = "left",
                            normalize = true,
                            which_decomp = nothing,
                            svd_alg = nothing)
    end

    ampo = OpSum()
    ampo += -1.0, "ProjUp", N, "ProjUp", 1
    ampo += 1.0, "Id", 1

    H_proj = MPO(ampo, sites)
    ψ0 = contract(H_proj, ψ0; maxdim = maxD, normalize = true)

    setprime!(ψ0, 0; plev=1)
    normalize!(ψ0[1])

    return ψ0
end

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
    C = a*σ + b*fold
    return C, E, σ, fold
end

function isScar(ψ, H, sites; σ_tol = 0.1)
    N = length(sites)
    E = inner(ψ', H, ψ)
    var = inner(H, ψ, H, ψ) - E^2

    # Verify if the entanglement entropy is below the volume law upper bound
    S = SvN_ent(copy(ψ))
    S > log(2) * N/4 && return false

    #Verify if the stat is in the Rydberg subspace
    val = 0.0
    for j in 1:N
        jj = mod1(j+1, N)
        ampo = OpSum()
        ampo += 1.0, "ProjUp", j, "ProjUp", jj
        O = MPO(ampo, sites)
        val = max(val, abs(inner(ψ', O, ψ)))
    end
    val > 1e-8 && return false

    # Finally, check if the variance is below the threshold
    return true

end

function minimize(H, H2, sites, prevStates, ψ0, E_target, a, b;
                    maxD = 200, n_iter = 30, n_sweeps = 4,
                    weight = 100.0, tol = 1e-8, σ_thre = 1e-6)

    ψ = copy(ψ0)
    normalize!(ψ)

    sweeps = Sweeps(n_sweeps)
    maxdim!(sweeps, maxD)
    cutoff!(sweeps, 1e-12)
    noise!(sweeps, 1e-8, 1e-9, 0.0)

    Cprev = Inf
    ψ = restrictHS(ψ, sites, maxD)
    E_curr = inner(ψ', H, ψ)

    @printf("Minimization started: E_target = %.2f, initial E = %.2f\n", E_target, E_curr)

    for iter in 1:n_iter
        O_eff = build_O_eff(H, H2, E_target, E_curr, a, b)
        if isempty(prevStates)
            _, ψ = dmrg(O_eff, ψ, sweeps; outputlevel = 0)
        else
            _, ψ = dmrg(O_eff, prevStates, ψ, sweeps; weight = weight, outputlevel = 0)
        end

        ψ = restrictHS(ψ, sites, maxD)
        E_curr = inner(ψ', H, ψ)

        C, E, var, fold = cost(ψ, H, H2, E_target, a, b)
        @printf("Iter %d: E = %.12f, Var = %.2e, Fold = %.2e, Cost = %.2e\n", iter, E, var, fold, C)

        if abs(C - Cprev) < tol && var < σ_thre
            @printf("Converged after %d iterations.\n", iter)
            break
        end
        Cprev = C
    end
    return restrictHS(ψ, sites, maxD)
end

function findScars()
    N = 20
    maxD = 1200
    cut = 1e-10

    a = 0.05
    b = 0.25

    E_min = -0.6 * N
    E_max = 0.6 * N

    n_targ = N + 1
    weight = 100.0
    ΔEmin = 0.2
    σ_thre = 0.000001
    n_iter = 100

    folder = "data4"

    idx = 1

    mkpath(folder)

    sites = siteinds("S=1/2", N, conserve_qns=false)
    H = PXP_Hamiltonian(sites)
    Id = IdOp(sites)
    H2 = apply(H, H; maxdim = maxD, cutoff = cut)

    Etarg = range(E_min, E_max, length=n_targ)

    #reverse Etarg
    # Etarg = reverse(Etarg)
    @printf("Target energies: %s\n", string(Etarg))

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
        ψ = restrictHS(ψ, sites, maxD)

        ψ = minimize(H, H2, sites, ψs, ψ, E_target, a, b; maxD = maxD, 
                    n_iter = n_iter, n_sweeps = 10, weight = weight, tol = 1e-8, σ_thre = σ_thre)
        E_found = inner(ψ', H, ψ)
        var = inner(ψ', H2, ψ) - E_found^2

        C_found, _, _, _ = cost(ψ, H, H2, E_target, a, b)
        @printf("E_target = %.2f, E_found = %.2f, var = %.2e, C_found = %.2e\n", E_target, E_found, var, C_found)
        
        if any(abs(E_found - Ep) < ΔEmin for Ep in Es)
            @printf("State too close to previously found state, skipping.\n")
            continue
        end

        if isScar(ψ, H, sites; σ_tol = σ_thre)
            @printf("  ✓ Scar: E=%+.6f  Var=%.2e\n", E_found, var)
            push!(ψs, ψ)
            push!(Es, E_found)
            h5open("$folder/N$(N)_$(idx).h5", "w") do f
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
        return MPS[], Float64[]
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
        f = h5open("data3/N$(N)_$(k).h5", "r")
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

function plot(folder, N)
    Es = Float64[]
    Z2_olap = Float64[]

    for file in readdir(folder)
        f = h5open(joinpath(folder, file), "r")
        E = read(f, "energy")
        ψ = read(f, "ψ", MPS)
        close(f)

        Z2 = Z2state(siteinds(ψ))
        olap = inner(ψ, Z2)
        olap = abs(olap)^2
        olap = log10(olap)

        @printf("File %s: E=%.12f, Z2_overlap=%.6f\n", file, E/N, olap)

        push!(Es, E/N)
        push!(Z2_olap, olap)
    end

    fig = Figure(size=(800, 600))
    ax = Axis(fig[1, 1], xlabel=L"E/N", ylabel = L"\log_{10} |\langle Z_2 | \Psi_n \rangle|^2")
    scatter!(ax, Es, Z2_olap, color=:blue, markersize=8)
    ylims!(ax, -25, 0)
    xlims!(ax, -0.8, 0.8)
    return fig, ax
end
