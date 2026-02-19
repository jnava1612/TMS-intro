using ITensors
using LinearAlgebra
using Random
using Printf
using ITensors.HDF5
using CairoMakie

# This version of the code is not yet capable of reproducing the tower of scars that is expected on the PXP model.
# It is only capable of finding the lowest energy scar state and succesfully reproducing its values.
# Further changes implementd on the code will ensure the ability to find the whole tower of scar states, and not just the lowest energy one.

function main()
    E_den = -0.6:0.05:0.6
    N = 80
    len = size(E_den)[1]
    # for i in 1:len
    #     E0 = E_den[i]*N
    #     @show "Running TMS with E0 = $E0"
    #     run(E0, N, i)
    # end
    
    Z2_overlap = zeros(len)
    Elist = zeros(len)
    for k in 1:len
        f = h5open("data/N$(N)_$(k).h5", "r")
        ψ = read(f, "ψ", MPS)
        close(f)

        sites = siteinds(ψ)
        H = PXP_Hamiltonian(sites)
        Z2 = Z2state(sites)

        Elist[k] = inner(ψ', H, ψ)
        Z2_overlap[k] = inner(ψ, Z2)
    end

    @printf("Now time for the plot\n")

    Z2_overlap = abs.(Z2_overlap).^2
    Elist = Elist ./ 20

    fig = Figure(resolution=(800, 600))
    ax = Axis(fig[1, 1],
                xlabel = L"E/L",
                ylabel = L"\log_{10} |\langle Z_2 | \Psi_n \rangle|^2",
                title = "Overlap of TMS states with Z2 state")
                scatter!(ax, E_den, log10.(Z2_overlap))
                # save("data/TMS.pdf", fig)

    return fig
end

function run(E0, N, iter)
    maxD = 100
    minD = 20
    max_op = 100 # maximum number of iterations

    ξthre = 0.1 # threshold for updating ξ

    ϵ = 1e-10 # convergence threshold for energy

    sites = siteinds("S=1/2", N, conserve_qns=false)

    @show sites[1]

    varray = zeros(1, max_op)

    @show "Building PXP Hamiltonian"
    H = PXP_Hamiltonian(sites)
    @show H
    @show "Building Z2 state"
    # ψ0 = Z2state(sites)
    states = [isodd(n) ? "Up" : "Dn" for n in 1:length(sites)]
    ψ0 = randomMPS(sites, states, 2)
    @show ψ0

    ampo0 = OpSum()
    ampo0 += 1.0,"Id",1
    H0 = MPO(ampo0, sites)
    H2 = H - E0*H0

    ψ = copy(ψ0)
    for k in 1:max_op
        @show k
        if k<5
            maxDim = minD
        else
            maxDim = maxD
        end
        sweeps = Sweeps(1)
        maxdim!(sweeps, maxDim, maxDim)
        cutoff!(sweeps, 1e-16)
        @show sweeps


        _, ψ = dmrg(H2, ψ, sweeps)
        ψ0 = copy(ψ)
        orthogonalize!(ψ0, 1)

        for j in 1:N-1
            # @show "Updating bond $j"
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
        @printf("PXP E variance=%.12f,    E=%.12f\n",varlist ,Elist)
        varray[k] = varlist
        if varlist <= ξthre
            H2 = H - (Elist-(-1)^k*3*varlist)*H0
            # H2 = H - Elist*H0
            ξthre = varlist
            @printf("Updating ξ to %.12f\n", ξthre)
        end

        # mkpath("data")
        # f = h5open("data/N$(N)_$(k).h5", "w")
        # write(f, "ψ", ψ)
        # close(f)

        if varlist < ϵ
            @show "Converged with variance = $varlist < $ϵ"
            break
        end

        # @printf("------------------------------\n")
        # for kk in 1:k
        #     @printf("varray[%d] = %.12f\n", kk, varray[kk])
        # end
    end

    Elist = inner(ψ', H, ψ)
    varlist = inner(H, ψ, H, ψ) - Elist^2
    @printf("Final PXP E variance=%.12f,    E=%.12f\n",varlist ,Elist)
    mkpath("data")
    f = h5open("data/N$(N)_$(iter).h5", "w")
    write(f, "ψ", ψ)
    close(f)

    # return ψ, Elist, varlist

end

function PXP_Hamiltonian(sites; periodic =true)
    ampo = OpSum()
    N = length(sites)
    for j in 1:N
        if periodic
            left = (j == 1 ? N : j - 1)
            right = (j == N ? 1 : j + 1)
            ampo += 1.0, "ProjDn", left, "X", j, "ProjDn", right
        else
            left = j - 1
            right = j + 1
            if left >= 1 && right <= N
                ampo += 1.0, "ProjDn", left, "X", j, "ProjDn", right
            end
        end
    end
    H = MPO(ampo, sites)
    return H
end

function Z2state(sites)
    states = [isodd(n) ? "Up" : "Dn" for n in 1:length(sites)]
    ψ = productMPS(sites, states)
    return ψ
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
