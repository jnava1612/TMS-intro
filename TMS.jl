using ITensors
using LinearAlgebra
using Random
using Printf
using ITensors.HDF5

function main()
    N = 20
    maxD = 10
    minD = 20

    @show "Generating random MPS with N = $N, maxD = $maxD, minD = $minD"

    E0 = -1.1
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
    ψ0 = Z2state(sites)
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

        mkpath("data")
        f = h5open("data/N$(N)_$(k).h5", "w")
        write(f, "ψ", ψ)
        close(f)

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

