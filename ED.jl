using LinearAlgebra
using SparseArrays
using Printf
using Statistics: mean, std
using DataFrames
using CSV
using CairoMakie
using ITensors
using ITensorMPS
using HDF5

function constrainedBasis(N::Int)
    basis = Int[]
    for state in 0:(2^N - 1)
        valid = true
        for j in 1:N-1
            # Check if both site j and j+1 are "Up" (i.e., bit is 1)
            if (state >> j)&1 == 1 && (state >> mod(j+1, N))&1 ==1
                valid = false
                break
            end
        end
        valid && push!(basis, state)
    end
    index = Dict(state => i for (i, state) in enumerate(basis))
    return basis, index
end

function Z2state(sites)
    states = [isodd(n) ? "Up" : "Dn" for n in 1:length(sites)]
    return productMPS(sites, states)
end

function sparsePXP(N::Int, basis::Vector{Int}, index::Dict{Int,Int})
    D = length(basis)
    rows = Int[]
    cols = Int[]
    vals = Float64[]

    for (col, state) in enumerate(basis)
        for j in 0:N-1
            prev = mod(j-1, N)
            next = mod(j+1, N)
            if (state >> prev)&1 == 0 && (state >> next)&1 == 0
                new_state = state ⊻ (1 << j) # Flip bit j
                if haskey(index, new_state)
                    push!(rows, index[new_state])
                    push!(cols, col)
                    push!(vals, 1.0)
                end
            end
        end
    end
    return sparse(rows, cols, vals, D, D)
end

function SvN_ent(ψ::Vector{Float64}, basis::Vector{Int}, N::Int)
    Nhalf = N ÷ 2
    mask_L = (1 << Nhalf) - 1   # bitmask for left sites 0..Nhalf-1
    mask_R = ~mask_L             # bitmask for right sites Nhalf..N-1

    # Collect unique left and right configurations
    left_states  = Dict{Int,Int}()
    right_states = Dict{Int,Int}()

    for state in basis
        L = state & mask_L
        R = (state >> Nhalf) & mask_L   # shift right half to low bits
        haskey(left_states,  L) || (left_states[L]  = length(left_states)  + 1)
        haskey(right_states, R) || (right_states[R] = length(right_states) + 1)
    end

    nL = length(left_states)
    nR = length(right_states)

    # Build the coefficient matrix M_{iL, iR} = ⟨iL, iR|ψ⟩
    # in the Schmidt basis of left and right configurations
    M = zeros(Float64, nL, nR)
    for (k, state) in enumerate(basis)
        L  = state & mask_L
        R  = (state >> Nhalf) & mask_L
        iL = left_states[L]
        iR = right_states[R]
        M[iL, iR] += ψ[k]
    end

    # SVD of M: singular values are Schmidt coefficients λ_i
    # Reduced density matrix eigenvalues are λ_i²
    sv = svdvals(M)
    sv = sv[sv .> 1e-14]

    return -sum(s^2 * log(s^2) for s in sv)
end

function overlapZ2(ψ::Vector{Float64}, basis::Vector{Int}, N::Int, index::Dict{Int,Int})
    # Computes |⟨Z2|ψ⟩|²
    z2 = sum(1 << j for j in 0:2:N-1)

    haskey(index, z2) || return 0.0

    return ψ[index[z2]]^2
end


function identifyScars(evals::Vector{Float64}, evecs::Matrix{Float64},
                    basis::Vector{Int}, N::Int, index::Dict{Int,Int}; 
                    overlap_tol = 1e-5, S_thr = 0.3)

    D = length(evals)
    S_vol = log(2) * N/2

    entropies = zeros(Float64, D)
    overlaps = zeros(Float64, D)

    for i in 1:D
        i % 500 == 0 && @printf("  %d / %d\n", i, D)
        ψ = evecs[:, i]
        entropies[i] = SvN_ent(ψ, basis, N)
        overlaps[i] = overlapZ2(ψ, basis, N, index)
    end

    scars_idx = Int[]
    for i in 1:D
        if entropies[i] < S_thr*S_vol && overlaps[i] > overlap_tol
            push!(scars_idx, i)
        end
    end
    
    diagnostic = Dict(
        "entropies" => entropies,
        "overlaps" => overlaps,
        "S_vol" => S_vol,
        "S_ev" => S_thr*S_vol,
        "overlap_tol" => overlap_tol
    )

    return scars_idx, diagnostic
end


function runED(N::Int; S_thr = 0.3, ov_thr = 1e-5, verbose = true)
    basis, index = constrainedBasis(N)
    H_sp = sparsePXP(N, basis, index)
    D = length(basis)

    @printf("Diagonalizing %d x %d matrix...\n", size(H_sp)...)

    H_dense = Matrix(H_sp)
    t0 = time()
    F = eigen(Symmetric(H_dense))
    @printf("Diagonalization completed in %.2f seconds.\n", time() - t0)

    evals = F.values
    evecs = F.vectors

    scar_idx, diagnostic = identifyScars(evals, evecs, basis, N, index; 
                                    overlap_tol = ov_thr, S_thr = S_thr)

    @printf("Found %d possible scar states.\n", length(scar_idx))

    if verbose
        for i in scar_idx
            @printf("  E=%.6f  S=%.2f  O=%.2e\n", evals[i], diagnostic["entropies"][i], diagnostic["overlaps"][i])
        end
    end

    entropies = diagnostic["entropies"]
    overlaps = diagnostic["overlaps"]

    df = DataFrame(Energy = evals, Entropy = entropies, Overlap = overlaps)

    CSV.write("full_spectrum2.csv", df)

    df2 = DataFrame(Energy = evals[scar_idx], Entropy = entropies[scar_idx], Overlap = overlaps[scar_idx])

    println("Plotting $(length(scar_idx)) scar candidates...")

    fig  = Figure(size=(800, 600))
    ax = Axis(fig[1, 1], xlabel=L"E/N", ylabel = L"\log_{10} |\langle Z_2 | \Psi_n \rangle|^2")
    scatter!(ax, df2.Energy, log10.(df2.Overlap), color=:blue, markersize=8)
    # ylims!(ax, -25, 0)
    xlims!(ax, -0.8, 0.8)
    # title!(ax, "PXP Scar Candidates: N=$N, S_thr=$(S_thr), O_thr=$(ov_thr)")
    display(fig)

    CSV.write("scar_states2.csv", df2)

    evecs2 = evecs[:, scar_idx]
    h5open("scar_states2.h5", "w") do file
        write(file, "energies", evals[scar_idx])
        write(file, "entropies", entropies[scar_idx])
        write(file, "overlaps", overlaps[scar_idx])
        write(file, "evecs", evecs2)
    end

    return scar_idx, evals, evecs, df2
end

function eig_to_MPS(v::Vector{Float64}, basis::Vector{Int}, sites;
                     maxD=200, cutoff=1e-12, E_ED=nothing, H_mpo=nothing)
    N = length(sites)
    Ψ_tensor = ITensor(sites)
    for (i, state_int) in enumerate(basis)
        config = [sites[j] => 2 - ((state_int >> (j-1)) & 1) for j in 1:N]
        Ψ_tensor[config...] = v[i]
    end
    Ψ_tensor /= norm(Ψ_tensor)
    ψ = MPS(Ψ_tensor, sites; maxdim=maxD, cutoff=cutoff)
    done = true

    if !isnothing(H_mpo) && !isnothing(E_ED)
        E_mps = inner(ψ', H_mpo, ψ)
        @printf("  E_ED = %+.8f   E_MPS = %+.8f   ΔE = %.2e\n",
                E_ED, E_mps, abs(E_mps - E_ED))
        
        if abs(E_mps - E_ED) > 1e-4
            @warn "Energy mismatch — check basis ordering convention"
            done = false
        end
    end

    return ψ, done
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

function scars_to_MPS(scar_idx, evecs, basis, sites, index::Dict{Int,Int};
                        maxD=200, cutoff=1e-12, evals=nothing, H_mpo=nothing)
    scar_mps = MPS[]
    Z2 = Z2state(sites)
    for (idx, i) in enumerate(scar_idx)
        v = evecs[:, i]
        E_ED = isnothing(evals) ? nothing : evals[i]
        ψ, done = eig_to_MPS(v, basis, sites; maxD=maxD, cutoff=cutoff, E_ED=E_ED, H_mpo=H_mpo)
        if done
            push!(scar_mps, ψ)
            h5open("scar_$(idx).h5", "w") do f
                write(f, "energy", E_ED)
                write(f, "mps_tensor", ψ)
            end
            o2 = inner(Z2, ψ)^2
            ovelap = overlapZ2(v, basis, length(sites), index)
            @printf("  Scar %d: E=%.6f  O_Z2=%.2e  O_Z2_MPS=%.2e\n", idx, E_ED, ovelap, o2)
        
        else
            @warn "Skipping MPS conversion for scar index $i due to energy mismatch."
        end
    end
    return scar_mps
end
