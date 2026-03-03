using ITensors
using ITensorsMPS
using Printf
using Statistics:mean
using HDF5

function uniformSites(sites, folder; fname = nothing)
    if isnothing(fname)
        files = filter(f -> endswith(f, ".h5"), readdir(folder))
    else
        files = [fname]
    end
    
    for file in files
        println("Loading data from $file...")
        h5open(joinpath(folder, file), "r") do f
            ψ = read(f, "state")
        end
        sites_psi = siteinds(ψ)
        for i in 1:length(ψ)
            ψ[i] = replaceind(ψ[i], sites_psi[i], sites[i])
        end
        f = h5open(joinpath(folder, file), "r+")
        HDF5.delete_object(f, "ψ")
        write(f, "ψ", ψ)
        close(f)
    end
    return true
end

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

function buildOperators(sites; maxD = 200, cut = 1e-10)
    H = PXP_Hamiltonian(sites)
    H2 = apply(H, H; maxdim = maxD, cutoff = cut)
    Id = IdOp(sites)
    PZ2 = Z2proj(sites)
    return H, H2, Id, PZ2
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

function cost(ψ, H, H2, PZ2, E_target, a, b, c)
    E = inner(ψ', H, ψ)
    EH2 = inner(ψ', H2, ψ)
    σ = EH2 - E^2
    ov2 = inner(ψ', PZ2, ψ)
    fold = EH2 -2*E_target*E + E_target^2
    C = a*σ + b*fold + c*ov2
    return C, E, σ, fold, ov2
end

function build_O_eff(H::MPO, H2::MPO, PZ2::MPO, E_target, E_curr, a, b, c)
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