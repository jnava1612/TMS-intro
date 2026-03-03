using ITensors
using ITensorMPS
using Printf
using Statistics:mean
using HDF5

function uniformSites(sites, ψs, Es, folder; fname = nothing)
    if isnothing(fname)
        files = filter(f -> endswith(f, ".h5"), readdir(folder))
    else
        files = [fname]
    end
    
    for file in files
        println("Loading data from $file...")
        f = h5open(joinpath(folder, file), "r")
        ψ = read(f, "ψ", MPS)
        E = read(f, "energy")
        close(f)
        sites_psi = siteinds(ψ)
        for i in 1:length(ψ)
            ψ[i] = replaceind(ψ[i], sites_psi[i], sites[i])
        end
        f = h5open(joinpath(folder, file), "r+")
        HDF5.delete_object(f, "ψ")
        write(f, "ψ", ψ)
        close(f)
        push!(ψs, ψ)
        push!(Es, E)
    end
    return true
end

function checkMissing(energies; tolerance = 1.5, ω=1.33)
    if length(energies) < 2
        @printf("Not enough energies to check for missing states.\n")
        return -1
    end
    @assert energies == sort(energies) "Energies must be sorted in ascending order"
    spacings = diff(energies)
    gap = minimum([median(spacings), mean(spacings), ω])

    for (i, δ) in enumerate(spacings)
        if δ > tolerance * gap
            @printf("Warning: large gap detected between energies %.6f and %.6f (gap = %.2e)\n", 
                    energies[i], energies[i+1], δ)
            return i
        end
    end
    return 0
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

function minimize(H, H2, PZ2, prevStates, ψ0, E_target, a, b, c;
                    maxD = 200, n_iter = 30, n_sweeps = 4,
                    weight = 100.0, tol = 1e-8, σ_thre = 1e-6, ov2_thre = 1e-25)
    
    @assert abs(a+b+c-1.0) < 1e-10 "Coefficients a, b, c must sum to 1.0, got $(a+b+c)"
    @assert c>=0 "Coefficient c must be non-negative"

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
        O_eff = build_O_eff(H, H2, PZ2, E_target, E_curr, a, b, c)
        if isempty(prevStates)
            _, ψ = dmrg(O_eff, ψ, sweeps; outputlevel = 0)
        else
            _, ψ = dmrg(O_eff, prevStates, ψ, sweeps; weight = weight, outputlevel = 0)
        end

        ψ = restrictHS(ψ, siteinds(ψ), maxD)
        E_curr = inner(ψ', H, ψ)

        C, E, var, fold, ov2 = cost(ψ, H, H2, PZ2, E_target, a, b, c)

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

    prevRuns = true # Set to true to read scars from previous runs and use them as warmup states, false to start from scratch
    folder = "data5"
    fname = nothing # Set to a specific filename to read only that file, or nothing to read all scars from the folder
    idx = 5 # Starting index for saving scars, set to 1 if starting from scratch, or to the next available index if reading from previous runs

    # Coefficients for the cost function, must sum to 1.0
    a = 0.02
    b = 0.31
    c = 0.67

    n_iter = 200 # Maximum number of iterations for the optimization, can be increased for better convergence but will take more time

    E_min = -11.2
    E_max = 11.2

    n_targ = 40
    weight = 100.0
    ΔEmin = 1.0

    σ_thre = 0.000001
    ov2_thre = 1e-6
    σ_inc = 5 # The factor by which the variance threshold is increased if no scars are found in a given iteration, to allow for finding scars with higher variance that may be missed with a too strict threshold. Can be adjusted based on the expected variance of the scar states.
    #ov2_dec = 10 # The factor by which the overlap threshold is decreased if no scars are found in a given iteration, to allow for finding scars with lower overlap that may be missed with a too strict threshold. Can be adjusted based on the expected overlap of the scar states.

    mkpath(folder) # Create folder if it doesn't exist

    sites = siteinds("S=1/2", N; conserve_qns=false)
 
    ψs = MPS[]
    Es = Float64[]

    if prevRuns
        println("Loading scars from previous runs in folder $folder...")
        uniformSites(sites, ψs, Es, folder; fname)
        perm = sortperm(Es)
        ψs = ψs[perm]
        Es = Es[perm]
    end

    println("Building Hamiltonian and operators...")

    H, H2, Id, PZ2 = buildOperators(sites; maxD = maxD, cut = cut)

    Etarg = range(E_min, E_max, length=n_targ)

    @printf("Target energies: %s\n", string(Etarg))

    
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

    println("Starting scar search with VQE optimization... (a=%.2f, b=%.2f, c=%.2f)\n", a, b, c)
    println("First sweep, may miss some scars")

    for (k, E_target) in enumerate(Etarg)

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

        ψ = minimize(H, H2, PZ2, ψs, ψ, E_target, a, b,c; maxD = maxD, 
                    n_iter = n_iter, n_sweeps = 15, weight = weight, tol = 1e-14,
                    σ_thre = σ_thre, ov2_thre = ov2_thre)
        E_found = inner(ψ', H, ψ)

        C_found, _, var, _, ov2 = cost_3p(ψ, H, H2, PZ2, E_target, a, b, c)
        
        @printf("E_target = %.2f, E_found = %.2f, var = %.2e, C_found = %.2e\n", E_target, E_found, var, C_found)
        
        if any(abs(E_found - Ep) < ΔEmin for Ep in Es)
            @printf("State too close to previously found state, skipping.\n")
            continue
        end

        if isScar(ψ, H, PZ2, sites; σ_tol = σ_thre, otol = ov2_thre)
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
    n_targ = 10
    while true
        targ_miss = checkMissing(Es; tolerance = 1.5, ω=1.33)
        if targ_miss > 0
            σ_thre *= σ_inc

            Etarg2 = range(Es[targ_miss] + ΔEmin, Es[targ_miss+1] - ΔEmin, length=n_targ)
            @printf("Refining search between E=%.2f and E=%.2f with relaxed thresholds (σ_thre=%.2e, ov2_thre=%.2e)...\n", 
                    Es[targ_miss], Es[targ_miss+1], σ_thre, ov2_thre)

            for E_target in Etarg2
                ψ0 = Z2state(sites)
                H2_shift = apply(H - E_target*Id, H - E_target*Id; maxdim = maxD, cutoff = cut)
            
                _, ψ = dmrg(H2_shift, ψs, ψ0, sweeps_WUp; weight = weight, outputlevel = 0)
                ψ = restrictHS(ψ, sites, maxD)
                ψ = minimize(H, H2, PZ2, ψs, ψ, E_target, a, b,c; maxD = maxD, 
                            n_iter = n_iter, n_sweeps = 15, weight = weight, tol = 1e-14,
                            σ_thre = σ_thre, ov2_thre = ov2_thre)
                E_found = inner(ψ', H, ψ)

                C_found, _, var, _, ov2 = cost_3p(ψ, H, H2, PZ2, E_target, a, b, c)
                @printf("E_target = %.2f, E_found = %.2f, var = %.2e, C_found = %.2e\n", E_target, E_found, var, C_found)

                if any(abs(E_found - Ep) < ΔEmin for Ep in Es)
                    @printf("State too close to previously found state, skipping.\n")
                    continue
                end

                if isScar(ψ, H, PZ2, sites; σ_tol = σ_thre, otol = ov2_thre)
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

                perm = sortperm(Es)
                ψs = ψs[perm]
                Es = Es[perm]
            end
        elseif targ_miss == 0
            @printf("No large gaps detected in the found scar energies.\n")
            break
        elseif targ_miss == -1
            @printf("Not enough energies to check for missing states.\n")
            return MPS[], Float64[]
        end
    end

    return ψs, Es

end
