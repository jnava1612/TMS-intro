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

end