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

function Z2proj(sites)
    N = length(sites)
    ampo = OpSum()
    ops_sites = []
    for j in 1:N
        push!(ops_sites, isodd(j) ? "ProjUp" : "ProjDn")
        push!(ops_sites, j)
    end
    ampo += (1.0, ops_sites...)

    return MPO(ampo, sites)
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
    orthogonalize!(ψ0, 1)
    normalize!(ψ0[1])

    return ψ0
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

function build_O_eff(H::MPO, H2::MPO, E_targ, E_curr, a, b)
    # Builds the effective operator O_eff = (a + b)*H2 - 2*(a*E_curr + b*E_targ)*H
    return (a + b)*H2 -2*(a*E_curr + b*E_targ)*H
end

function cost_3p(ψ, H, H2, PZ2, E_target, a, b, c)
    E = inner(ψ', H, ψ)
    EH2 = inner(ψ', H2, ψ)
    σ = EH2 - E^2
    ov2 = inner(ψ', PZ2, ψ)
    fold = EH2 -2*E_target*E + E_target^2
    C = a*σ + b*fold + c*ov2
    return C, E, σ, fold, ov2
end

function build_O_eff_3p(H::MPO, H2::MPO, PZ2::MPO, E_target, E_curr, a, b, c)
    return (a + b)*H2 - 2*(b*E_target + a*E_curr)*H - c*PZ2
end

function isScar(ψ, H, PZ2, sites; σ_tol = 0.1, otol = 1e-25)
    N = length(sites)
    E = inner(ψ', H, ψ)
    var = inner(H, ψ, H, ψ) - E^2
    ov2 = inner(ψ', PZ2, ψ)

    # Verify if the entanglement entropy is below the volume law upper bound
    S = SvN_ent(copy(ψ))
    S > 0.3*log(2) * N/2 && return false

    # Verify if it's an eigenstate by checking the variance of the energy
    var >= σ_tol && return false

    # Verify if the overlap with the Z2 state is non-negligible
    ov2 <= otol && return false

    # Finally, check if the variance is below the threshold
    return true

end

function minimize(H, H2, sites, prevStates, ψ0, E_target, a, b;
                    maxD = 200, n_iter = 30, n_sweeps = 4,
                    weight = 100.0, tol = 1e-8, σ_thre = 1e-6)

    ψ = copy(ψ0)
    normalize!(ψ)

    Cprev = Inf
    ψ = restrictHS(ψ, sites, maxD)
    E_curr = inner(ψ', H, ψ)
    
    @printf("Minimization started: E_target = %.2f, initial E = %.2f\n", E_target, E_curr)
    
    for iter in 1:n_iter
        sweeps = Sweeps(n_sweeps)
        if iter < 5
            maxdim!(sweeps, 20)
        else
            maxdim!(sweeps, maxD)
        end
        cutoff!(sweeps, 1e-12)
        noise!(sweeps, 1e-8, 1e-9, 0.0)
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

function minimize_3p(H, H2, PZ2, prevStates, ψ0, E_target, a, b, c;
                    maxD = 200, n_iter = 30, n_sweeps = 4,
                    weight = 100.0, tol = 1e-8, σ_thre = 1e-6, ov2_thre = 1e-25)
    
    @assert abs(a+b+c-1.0) < 1e-10 "Coefficients a, b, c must sum to 1.0, got $(a+b+c)"
    @assert c>=0 "Coefficient c must be non-negative"
    # @assert c<0.5 "Too large c>0.5: the optimization will focus too much on maximizing the overlap with the Z2 state and may miss scars with low overlap but low variance"

    ψ = copy(ψ0)
    normalize!(ψ)

    sweeps = Sweeps(n_sweeps)
    maxdim!(sweeps, maxD)
    cutoff!(sweeps, 1e-12)
    noise!(sweeps, 1e-8, 1e-9, 0.0)

    C_prev = Inf
    ψ = restrictHS(ψ, siteinds(ψ), maxD)
    E_curr = inner(ψ', H, ψ)

    @printf("Minimization started: E_target = %.2f, initial E = %.2f (a=%.2f, b=%.2f, c=%.2f)\n", E_target, E_curr, a, b, c)

    for iter in 1:n_iter
        O_eff = build_O_eff_3p(H, H2, PZ2, E_target, E_curr, a, b, c)
        if isempty(prevStates)
            _, ψ = dmrg(O_eff, ψ, sweeps; outputlevel = 0)
        else
            _, ψ = dmrg(O_eff, prevStates, ψ, sweeps; weight = weight, outputlevel = 0)
        end

        ψ = restrictHS(ψ, siteinds(ψ), maxD)
        E_curr = inner(ψ', H, ψ)

        C, E, var, fold, ov2 = cost_3p(ψ, H, H2, PZ2, E_target, a, b, c)

        # if var < σ_thre
        #     σ_thre = var
        # end
        # if ov2 > ov2_thre
        #     ov2_thre = ov2  
        #     @printf("  New best overlap with Z2 state: %.2e\n", ov2)
        # end

        @printf("Iter %d: E = %.12f, Var = %.2e, Fold = %.2e, Ov2 = %.2e, Cost = %.2e\n", iter, E, var, fold, ov2, C)

        if (ov2 > ov2_thre && var < σ_thre) || abs(C - C_prev) < tol
            @printf("Converged at iter %d ΔC = %.2e, Var = %.2e\n", iter, abs(C - C_prev), var)
            break
        end
        C_prev = C
    end

    return restrictHS(ψ, siteinds(ψ), maxD)
end

function findScars()
    N = 20
    maxD = 1200
    cut = 1e-10

    a = 0.02
    b = 0.31
    c = 0.67

    E_min = -11.2
    E_max = 11.2

    n_targ = 40
    weight = 100.0
    ΔEmin = 1.0

    folder = "data5"

    idx = 5

    mkpath(folder)

    println("Building Hamiltonian and operators...")

    sites = siteinds("S=1/2", N; conserve_qns=false)

    Etarg = range(E_min, E_max, length=n_targ)
    # Etarg = reverse(Etarg) # Start from the highest energy to find scars in the middle of the spectrum first, which are more likely to be missed if we start from the lowest energy

    #reverse Etarg
    # Etarg = reverse(Etarg)
    @printf("Target energies: %s\n", string(Etarg))

    ψs = MPS[]
    Es = Float64[]

    
    # Read existing scars from previous runs to use them as warmup states and avoid finding the same scar multiple times
    for file in readdir(folder)
        f = h5open(joinpath(folder, file), "r")
        E = read(f, "energy")
        ψ = read(f, "ψ", MPS)
        close(f)
        println("Loaded scar from file $file with energy $E")
        # println("Original site indices: ", siteinds(ψ))
        sites_psi = siteinds(ψ)
        for i in 1:length(ψ)
            ψ[i] = replaceind(ψ[i], sites_psi[i], sites[i])
        end
        # ψ = replaceinds(ψ, siteinds(ψ), sites)
        
        f = h5open(joinpath(folder, file), "r+")
        HDF5.delete_object(f, "ψ")
        write(f, "ψ", ψ)
        close(f)

        println("Replaced site indices in scar from file $file to match current sites.")
        
        push!(ψs, ψ)
        push!(Es, E)
    end
    
    println(siteinds(ψs[1]))
    println("Curent site indices: ", sites)
      
    Id = IdOp(sites)
    H = PXP_Hamiltonian(sites)
    H2 = apply(H, H; maxdim = maxD, cutoff = cut)
    PZ2 = Z2proj(sites)

    sweeps_WUp = Sweeps(5)
    maxdim!(sweeps_WUp, 10, 30, 80, 150, maxD)
    cutoff!(sweeps_WUp, 1e-8)
    noise!(sweeps_WUp, 1e-5, 1e-6, 1e-7, 1e-8, 0.0)
    
    # Emm = minimum(abs.(Es))
    # println("Minimum absolute energy of previously found scars: $Emm")

    for (k, E_target) in enumerate(Etarg)
        if abs(E_target ) > 6
            continue
        end
        σ_thre = 0.000001
        ov2_thre = 1e-6
        n_iter = 200
        println("Running WUp for target energy $E_target...")
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

        ψ = minimize_3p(H, H2, PZ2, ψs, ψ, E_target, a, b,c; maxD = maxD, 
                    n_iter = n_iter, n_sweeps = 15, weight = weight, tol = 1e-14,
                    σ_thre = σ_thre, ov2_thre = ov2_thre)
        E_found = inner(ψ', H, ψ)
        var = inner(ψ', H2, ψ) - E_found^2

        C_found, _, var, _, ov2 = cost_3p(ψ, H, H2, PZ2, E_target, a, b, c)
        # if var < σ_thre
        #     σ_thre = var
        # end
        # if ov2 > ov2_thre
        #     ov2_thre = ov2
        # end

        @printf("E_target = %.2f, E_found = %.2f, var = %.2e, C_found = %.2e\n", E_target, E_found, var, C_found)
        
        if any(abs(E_found - Ep) < ΔEmin for Ep in Es)
            @printf("State too close to previously found state, skipping.\n")
            continue
        end

        if isScar(ψ, H, PZ2, sites; σ_tol = 0.1, otol = 1e-6)
            @printf("  ✓ Scar: E=%+.6f  Var=%.2e\n", E_found, var)
            push!(ψs, ψ)
            push!(Es, E_found)
            h5open("$folder/N$(N)_$(idx).h5", "w") do f
                write(f, "ψ", ψ)
                write(f, "energy",   E_found)
                write(f, "variance", var)
                write(f, "ov2_Z2", ov2)
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
