using ITensors
using ITensorMPS
using Printf
using Statistics: mean

function O_eff(H2::MPO, H::MPO, E_targ, E_curr, a, b)
    # Builds the effective operator O_eff = (a + b)*H2 - 2*(a*E_curr + b*E_targ)*H
    return (a + b)*H2 -2*(a*E_curr + b*E_targ)*H
end

function cost(ψ, H, H2, E_target, a, b)
    # Constructs the cost function C(ψ) = a*(<H^2> - <H>^2) + b*(E - E_target)^2
    E = inner(ψ', H, ψ)
    EH2 = inner(ψ', H2, ψ)
    σ = EH2 - E^2
    fold = EH2 -2*E_target*E + E_target^2
    return a*σ + b*fold, E, σ, fold
end
