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

    @show "Building PXP Hamiltonian"
    H = PXP_Hamiltonian(sites)
    @show H

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