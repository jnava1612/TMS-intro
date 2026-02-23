using ITensors
using ITensorMPS
using LinearAlgebra
using Random
using Printf
using HDF5
using CairoMakie
using Statistics

# This version of the code is not yet capable of reproducing the tower of scars that is expected on the PXP model.
# It is only capable of finding the lowest energy scar state and succesfully reproducing its values.
# Further changes implementd on the code will ensure the ability to find the whole tower of scar states, and not just the lowest energy one.

function main()
    N = 20
    scars = MPS[]
    # E_min, E_max = -0.6*N, 0.6*N
    # # for i in 1:N+1
    # #     E0 = E_den[i]
    # #     @printf("Running for E0=%.12f\n", E0)
    # #     run(E0, N, i,scars)
    # # end

    # run(E_min, E_max, N)
    
    nn = 40
    Z2_overlap = zeros(nn)
    Elist = zeros(nn)
    for k in 1:nn
        f = h5open("data/N$(N)_$(k).h5", "r")
        ψ = read(f, "ψ", MPS)
        close(f)

        sites = siteinds(ψ)
        H = PXP_Hamiltonian(sites)
        Z2 = Z2state(sites)

        Elist[k] = inner(ψ', H, ψ)
        Z2_overlap[k] = inner(ψ, Z2)
    end
    @printf("Energies and Z2 overlaps for found states:\n")
    for k in 1:nn
        @printf("k=%d, E=%.12f, Z2_overlap=%.6f\n", k, Elist[k], Z2_overlap[k])
    end

    @printf("Now time for the plot\n")

    Z2_overlap = abs.(Z2_overlap).^2
    Z2_overlap = log10.(Z2_overlap)
    Elist = Elist ./ N

     @printf("Energies and Z2 overlaps for found states:\n")
    for k in 1:nn
        @printf("k=%d, E=%.12f, Z2_overlap=%.6f\n", k, Elist[k], Z2_overlap[k])
    end

    fig = Figure(resolution=(800, 600))
    ax = Axis(fig[1, 1],
                xlabel = L"E/L",
                ylabel = L"\log_{10} |\langle Z_2 | \Psi_n \rangle|^2",
                title = "Overlap of TMS states with Z2 state")
    scatter!(ax, Elist, Z2_overlap, color=:blue, markersize=10)
    ylims!(ax, -10, 0)
                # save("data/TMS.pdf", fig)

    return fig
end

function extractScars(sites, N, E_arr)
    maxD = 100
    minD = 20
    max_op = 100 # maximum number of iterations

    ξthre = 0.1 # threshold for updating ξ

    ϵ = 1e-10 # convergence threshold for energy

    H = PXP_Hamiltonian(sites)

    ψ_z2 = Z2state(sites)
    found_scars = MPS[]

    for i in 1:N+1
        idmpo = OpSum()
        idmpo += 1.0,"Id",1
        Id = MPO(idmpo, sites)
        H_targ = H - E_arr[i]*Id
        varray = zeros(1, max_op)

    #     ψ0 = copy(ψ_z2)
    #     ψ = copy(ψ0)
    #     for k in 1:max_op
    #         @show k
    #         if k<5
    #             maxDim = minD
    #         else
    #             maxDim = maxD
    #         end
    #         sweeps = Sweeps(5)
    #         maxdim!(sweeps, maxDim, maxDim)
    #         cutoff!(sweeps, 1e-16)
    #         @show sweeps


    #         _, ψ = dmrg(H_targ, found_scars, ψ, sweeps)
    #         ψ0 = copy(ψ)
    #         orthogonalize!(ψ0, 1)

    #         for j in 1:N-1
    #             # @show "Updating bond $j"
    #             ϕ = ψ0[j] * ψ0[j+1]
    #             ϕ0 = copy(ϕ)
    #             ϕ1 = mapprime(ϕ0*op("ProjUp",sites[j]), 1 => 0)
    #             ϕ2 = mapprime(ϕ1*op("ProjUp",sites[j+1]), 1 => 0)
    #             ϕ3 = ϕ0 - ϕ2

    #             spec = replacebond!(ψ0, j, ϕ3; maxdim = maxD,
    #                                             mindim = 1,
    #                                             cutoff = 1e-15,
    #                                             eigen_perturbation = nothing,
    #                                             ortho = "left",
    #                                             normalize = true,
    #                                             which_decomp = nothing,
    #                                             svd_alg = "recursive")
    #         end

    #         ampo1 = OpSum()
    #         ampo1 += -1.0, "ProjUp", N, "ProjUp", 1
    #         ampo1 += 1.0,"Id",1
    #         H_temp = MPO(ampo1, sites)

    #         temp = contract(H_temp, ψ0, maxdim = maxDim, normalize = true)
    #         setprime!.(temp,0; plev=1)

    #         ψ0 = copy(temp)
    #         orthogonalize!(ψ0, 1)
    #         normalize!(ψ0[1])

    #         ψ = copy(ψ0)

    #         Elist = inner(ψ', H, ψ)
    #         varlist = inner(H, ψ, H, ψ) - Elist^2
    #         @printf("PXP E variance=%.12f,    E=%.12f\n",varlist ,Elist)
    #         varray[k] = varlist
    #         if varlist <= ξthre
    #             H2 = H - (Elist-(-1)^k*3*varlist)*H0
    #             # H2 = H - Elist*H0
    #             ξthre = varlist
    #             @printf("Updating ξ to %.12f\n", ξthre)
    #         end

    #         # mkpath("data")
    #         # f = h5open("data/N$(N)_$(k).h5", "w")
    #         # write(f, "ψ", ψ)
    #         # close(f)

    #         if varlist < ϵ
    #             @show "Converged with variance = $varlist < $ϵ"
    #             break
    #         end

    #         # @printf("------------------------------\n")
    #         # for kk in 1:k
    #         #     @printf("varray[%d] = %.12f\n", kk, varray[kk])
    #         # end
    #     end

        sweeps = Sweeps(10)
        maxdim!(sweeps, 20, 50, 100)
        cutoff!(sweeps, 1e-16)
        _, ψ = dmrg(H_targ, found_scars, ψ_z2, sweeps; weight = 100.0)

        Elist = inner(ψ', H, ψ)
        varlist = inner(H, ψ, H, ψ) - Elist^2
        @printf("Final PXP E variance=%.12f,    E=%.12f\n",varlist ,Elist)
        mkpath("data")
        f = h5open("data/N$(N)_$(i).h5", "w")
        write(f, "ψ", ψ)
        close(f)
        push!(found_scars, ψ)
    end
end

# function run(E0, N, iter, found_scars)
function run(Emin, Emax, N)
    maxD = 100
    minD = 20
    max_op = 100 # maximum number of iterations

    ξthre = 0.1 # threshold for updating ξ

    ϵ = 1e-10 # convergence threshold for energy

    sites = siteinds("S=1/2", N, conserve_qns=false)

    @show sites[1]

    varray = zeros(1, max_op)

    extractScars(sites, N, range(Emin, Emax, length=N+1))
    # @show H
    # @show "Building Z2 state"
    # ψ0 = Z2state(sites)
    # # states = [isodd(n) ? "Up" : "Dn" for n in 1:length(sites)]
    # # ψ0 = randomMPS(sites, states, 2)
    # @show ψ0

    # ampo0 = OpSum()
    # ampo0 += 1.0,"Id",1
    # H0 = MPO(ampo0, sites)
    # H2 = H - E0*H0

    # ψ = copy(ψ0)
    # for k in 1:max_op
    #     @show k
    #     if k<5
    #         maxDim = minD
    #     else
    #         maxDim = maxD
    #     end
    #     sweeps = Sweeps(5)
    #     maxdim!(sweeps, maxDim, maxDim)
    #     cutoff!(sweeps, 1e-16)
    #     @show sweeps


    #     _, ψ = dmrg(H2, found_scars, ψ, sweeps)
    #     ψ0 = copy(ψ)
    #     orthogonalize!(ψ0, 1)

    #     for j in 1:N-1
    #         # @show "Updating bond $j"
    #         ϕ = ψ0[j] * ψ0[j+1]
    #         ϕ0 = copy(ϕ)
    #         ϕ1 = mapprime(ϕ0*op("ProjUp",sites[j]), 1 => 0)
    #         ϕ2 = mapprime(ϕ1*op("ProjUp",sites[j+1]), 1 => 0)
    #         ϕ3 = ϕ0 - ϕ2

    #         spec = replacebond!(ψ0, j, ϕ3; maxdim = maxD,
    #                                           mindim = 1,
    #                                           cutoff = 1e-15,
    #                                           eigen_perturbation = nothing,
    #                                           ortho = "left",
    #                                           normalize = true,
    #                                           which_decomp = nothing,
    #                                           svd_alg = "recursive")
    #     end

    #     ampo1 = OpSum()
    #     ampo1 += -1.0, "ProjUp", N, "ProjUp", 1
    #     ampo1 += 1.0,"Id",1
    #     H_temp = MPO(ampo1, sites)

    #     temp = contract(H_temp, ψ0, maxdim = maxDim, normalize = true)
    #     setprime!.(temp,0; plev=1)

    #     ψ0 = copy(temp)
    #     orthogonalize!(ψ0, 1)
    #     normalize!(ψ0[1])

    #     ψ = copy(ψ0)

    #     Elist = inner(ψ', H, ψ)
    #     varlist = inner(H, ψ, H, ψ) - Elist^2
    #     @printf("PXP E variance=%.12f,    E=%.12f\n",varlist ,Elist)
    #     varray[k] = varlist
    #     if varlist <= ξthre
    #         H2 = H - (Elist-(-1)^k*3*varlist)*H0
    #         # H2 = H - Elist*H0
    #         ξthre = varlist
    #         @printf("Updating ξ to %.12f\n", ξthre)
    #     end

    #     # mkpath("data")
    #     # f = h5open("data/N$(N)_$(k).h5", "w")
    #     # write(f, "ψ", ψ)
    #     # close(f)

    #     if varlist < ϵ
    #         @show "Converged with variance = $varlist < $ϵ"
    #         break
    #     end

    #     # @printf("------------------------------\n")
    #     # for kk in 1:k
    #     #     @printf("varray[%d] = %.12f\n", kk, varray[kk])
    #     # end
    # end

    # Elist = inner(ψ', H, ψ)
    # varlist = inner(H, ψ, H, ψ) - Elist^2
    # @printf("Final PXP E variance=%.12f,    E=%.12f\n",varlist ,Elist)
    # mkpath("data")
    # f = h5open("data/N$(N)_$(iter).h5", "w")
    # write(f, "ψ", ψ)
    # close(f)
    # push!(found_scars, ψ)

    # # return ψ, Elist, varlist

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

function verifyScar(site, H, Ψs, Es)
    N = length(site)
    n = length(Ψs)
    S_vol = log(2)*N/2

    println("Verifying scar states...")
    if n > 1
        println("\n[1] Energies and spacings (should be ≈ constant ≈ 1.33 for N→∞):")
        spacings = diff(Es)
        mean_spacing = mean(spacings)
        @printf("  Mean spacing = %.6f  (expected ≈ %.4f for N=%d)\n",
                mean_spacing, 2π/N * N/2, N)

        # Verifying variance
        println("\n[2] Variance ⟨H²⟩ − ⟨H⟩² (should be ≈ 0 for true eigenstates):")
        for (k,ψ) in enumerate(Ψs)
            E   = inner(ψ', H, ψ)
            var = inner(H, ψ, H, ψ) - E^2
            flag = var < 1e-4 ? "✓" : (var < 1e-2 ? "~" : "✗")
            @printf("    State %2d:  E = %+.6f  Var = %.2e  %s\n", k, E, var, flag)
        end

        # 3. Mutual orthogonality
        println("\n[3] Overlap matrix |⟨ψ_i|ψ_j⟩|  (should be ≈ identity):")
        for i in 1:n
            print("    ")
            for j in 1:n
                @printf("%6.4f  ", abs(inner(Ψs[i], Ψs[j])))
            end
            println()
        end

        println("\n[4] Half-chain entanglement entropy (volume law upper bound = $(S_vol)):")
        # Verifying Entropy
        for (k, ψ) in enumerate(Ψs)
            S     = SvN_ent(copy(ψ))
            ratio = S / S_vol
            flag  = ratio < 0.3 ? "✓ sub-volume" : (ratio < 0.5 ? "~ marginal" : "✗ volume-law")
            @printf("    State %2d:  S = %.4f  (%.1f%% of vol-law)  %s\n",
                    k, S, 100*ratio, flag)
        end

        # 5. Z2 overlap
        println("\n[5] Overlap |⟨Z2|ψ_k⟩|²  (all scar states should be non-negligible):")
        ψZ2 = Z2state(sites)
        total = 0.0
        for (k, ψ) in enumerate(Ψs)
            ov2 = abs2(inner(ψZ2, ψ))
            total += ov2
            flag = ov2 > 1e-3 ? "✓" : "✗"
            @printf("    State %2d:  |⟨Z2|ψ_k⟩|² = %.6f  %s\n", k, ov2, flag)
        end
        @printf("    Total overlap with found states = %.4f\n", total)
    end
end

function isScar(ψ, H, sites; σ_tol = 0.1, S_tol = nothing, olap_tol = 1e-3)
    N = length(sites)
    E = inner(ψ', H, ψ)
    var = inner(H, ψ, H, ψ) - E^2
    if var > σ_tol
        return false
    end
    
    S_tol = isnothing(S_tol) ? log(2)*N/4 : S_tol
    S = SvN_ent(copy(ψ))
    if S > S_tol
        @printf("State rejected due to high entanglement entropy: S=%.4f > S_tol=%.4f\n", S, S_tol)
        return false
    end

    # ψZ2 = Z2state(sites)
    # ov2 = abs2(inner(ψZ2, ψ))
    # if ov2 < olap_tol
    #     @printf("State rejected due to low overlap with Z2 state: |⟨Z2|ψ⟩|²=%.6f < olap_tol=%.6f\n", ov2, olap_tol)
    #     return false
    # end

    return true
end

function dmrg2(H, prevStates, ψ0, sweeps; weight=100.0)
    if isempty(prevStates)
        _, ψ = dmrg(H, ψ0, sweeps)
    else
        _, ψ = dmrg(H, prevStates, ψ0, sweeps; weight=weight)
    end
    return ψ
end


function extractScars()
    N = 20
    maxD = 400
    minD = 20
    cut = 1e-10

    ntarg = 50

    Emin = -0.8*N
    Emax = 0.8*N
    
    maxop = 200

    varray = zeros(1, maxop)

    σ_thre = 0.1
    ol_thre = 1e-3
    ϵ = 0.0000000001

    ΔEmin = 0.05

    weight = 50.0

    sites = siteinds("S=1/2", N, conserve_qns=false)

    H = PXP_Hamiltonian(sites)
    
    Id = OpSum()
    Id += 1.0,"Id",1
    Id = MPO(Id, sites)

    ψs = MPS[]
    Es = Float64[]

    Etarg = range(Emin, Emax, length=ntarg)
    idx = 1

    for (k,E) in enumerate(Etarg)

        @printf("Extracting scar state %d with target energy E=%.12f\n", k, E)
        H2 = H - E*Id

        H2sq = apply(H2, H2; maxdim = maxD, cutoff = cut)
        
        ψ0 = Z2state(sites)

        sweeps_WUp = Sweeps(5)
        maxdim!(sweeps_WUp, 10, 30, 80, 150, maxD)
        cutoff!(sweeps_WUp, 1e-8)
        noise!(sweeps_WUp, 1e-5, 1e-6, 1e-7, 1e-8, 0.0)

        sweeps = Sweeps(20)
        maxdim!(sweeps, maxD)
        cutoff!(sweeps, 1e-12)
        noise!(sweeps, 1e-8, 1e-9, 0.0)

        ψ = dmrg2(H2sq, ψs, ψ0, sweeps_WUp; weight=weight)

        ψ = dmrg2(H2sq, ψs, ψ, sweeps; weight=weight)

        E_found = inner(ψ', H, ψ)
        var_found = inner(H, ψ, H, ψ) - E_found^2

        if any(abs(E_found - Ep) < ΔEmin for Ep in Es)
            @printf("  Skipping state with E=%.12f due to proximity to existing scar states\n", E_found)
            continue
        end

        if isScar(ψ, H, sites; σ_tol=σ_thre, olap_tol=ol_thre)
            @printf("  Found scar state with E=%.12f, variance=%.2e\n", E_found, var_found)
            push!(ψs, ψ)
            push!(Es, E_found)

            f = h5open("data/N$(N)_$(idx).h5", "w")
            write(f, "ψ", ψ)
            close(f)
            
            idx += 1
        else
            @printf("  State with E=%.12f does not meet scar criteria (variance=%.2e)\n", E_found, var_found)
            continue
        end

        
    end

    perm = sortperm(Es)
    Es = Es[perm]
    ψs = ψs[perm]

    verifyScar(sites, H, ψs, Es)

    return ψs, Es

end


function checkData(N)
    Z2_overlap = zeros(N+1)
    varlist = zeros(N+1)
    Elist = zeros(N+1)
    S_ent = zeros(N+1)
    for k in 1:5
        f =h5open("data/N$(N)_$(k).h5", "r")
        ψ = read(f, "ψ", MPS)
        close(f)

        sites = siteinds(ψ)
        H = PXP_Hamiltonian(sites)
        Z2 = Z2state(sites)

        Elist[k] = inner(ψ', H, ψ)
        varlist[k] = inner(H, ψ, H, ψ) - Elist[k]^2
        Z2_overlap[k] = inner(ψ, Z2)

        mid = Int(N/2)
        orthogonalize!(ψ, mid)
        ϕ = ψ[mid] * ψ[mid+1]
        _, S = svd(ϕ, (linkind(ψ, mid-1), siteind(ψ, mid)))

        SvN_initial = 0
        for n in 1:dim(S, 1)
            p = S[n, n]^2
            if p>1e-10
                SvN_initial -= p*log(p)
            end
        end
        S_ent[k] = SvN_initial
    end
    for k in 1:5
        @printf("k=%d, E=%.12f, var=%.12f, Z2_overlap=%.6f, S_ent=%.6f\n", k, Elist[k], varlist[k], Z2_overlap[k], S_ent[k])
    end
end



function test()
    N = 20
    maxD = 200
    minD = 20
    Eini = -1.1
    U = 1000
    maxop = 200

    ind = 1

    varray = zeros(1, maxop)

    σ_thre = 0.01
    ϵ = 0.0000000001

    sites = siteinds("S=1/2", N, conserve_qns=false)

    H = PXP_Hamiltonian(sites)

    ψ0 = Z2state(sites)

    Id = OpSum()
    Id += 1.0,"Id",1
    Id = MPO(Id, sites)
    H2 = H - Eini*Id 

    ψ = copy(ψ0)
    @show ψ

    for k in 1:maxop
        @show k
        if k<5
            maxDim = minD
        else
            maxDim = maxD
        end

        sweeps = Sweeps(5)
        maxdim!(sweeps, maxDim, maxDim)
        cutoff!(sweeps, 1e-16)
        # @show sweeps

        _, ψ = dmrg(H2, ψ, sweeps)
        ψ0 = copy(ψ)
        orthogonalize!(ψ0, 1)
        # @show ψ0

        for j in 1:N-1
            ϕ = ψ0[j] * ψ0[j+1]
            ϕ0 = copy(ϕ)
            ϕ1 = mapprime(ϕ0*op("ProjUp",sites[j]), 1 => 0)
            ϕ2 = mapprime(ϕ1*op("ProjUp",sites[j+1]), 1 => 0)
            ϕ3 = ϕ0 - ϕ2

            spec = replacebond!(ψ0, j, ϕ3; maxdim = maxD,
                                            mindim = 1,
                                            cutoff = 1e-15,
                                            eigen_perturbation = nothing,
                                            ortho = "left",
                                            normalize = true,
                                            which_decomp = nothing,
                                            svd_alg = "recursive")
        end

        ampo1 = OpSum()
        ampo1 += -1.0, "ProjUp", N, "ProjUp", 1
        ampo1 += 1.0,"Id",1
        H_temp = MPO(ampo1, sites)

        temp = contract(H_temp, ψ0, maxdim = maxDim, normalize = true)
        setprime!.(temp,0; plev=1)

        ψ0 = copy(temp)
        orthogonalize!(ψ0, 1)
        normalize!(ψ0[1])
        ψ = copy(ψ0)

        Elist = inner(ψ', H, ψ)
        varlist = inner(H, ψ, H, ψ) - Elist^2

        varray[k] = varlist

        if varlist <= σ_thre && varlist < 0.001
            H2 = H - (Elist-(-1)^k*varlist)*Id
            # H2 = H - Elist*Id
            σ_thre = varlist
            @printf("Updating σ_thre to %.12f\n", σ_thre)
        end

        
        if varlist < σ_thre || k == maxop
            f = h5open("data/$(N)_$(ind).h5", "w")
            write(f, "ψ", ψ)
            close(f)
            @show Elist
            if varlist < ϵ
                @show "Converged with variance = $varlist < $ϵ"
                break
            end
        end

    end

end