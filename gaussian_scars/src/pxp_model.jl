function constrainedBasis(N::Int)
    # Builds the constrained basis for the PXP model, 
    # where no two adjacent sites can both be in the excited state (1).
    # It has a size that grows as the Fibonacci sequence, which is much smaller than the full 2^N basis.
    basis = Int[]
    for state in 0:(2^N - 1)
        valid = true
        for j in 0:N-1
            jnext = mod(j+1, N)
            if (state >> j)&1 == 1 && (state >> jnext)&1 == 1
                valid = false
                break
            end
        end
        valid && push!(basis, state)
    end
    return basis
end

function build_reversal_map(L::Int)
    # Builds a map from any state to the state obtained by reversing the order of the sites.
    dim = 2^L
    map = Vector{Int}(undef, dim)
    for i in 0:(dim - 1)
        rev_i = 0
        for j in 0:(L - 1)
            if (i >> j) & 1 == 1
                rev_i |= (1 << (L - 1 - j))
            end
        end
        map[i + 1] = rev_i + 1
    end
    return map
end


function build_pxp_hamiltonian(L::Int)
    # Builds the PXP Hamiltonian in the constrained basis. 
    # The Hamiltonian has off-diagonal elements corresponding to flipping a site 
    # from 0 to 1 or from 1 to 0, but only if both neighboring sites are 0.
    rows = Int[]
    cols = Int[]
    vals = Float64[]

    basis = constrainedBasis(L)
    dim = length(basis)

    index = Dict(basis[i] => i for i in 1:dim)
    for (i, state) in enumerate(basis)
        for j in 0:(L-1)
            left  = mod(j-1, L)
            right = mod(j+1, L)
            if ((state >> left) & 1 == 0) && ((state >> right) & 1 == 0)
                new_state = state ⊻ (1 << j)
                if haskey(index, new_state)
                    push!(rows, index[new_state])
                    push!(cols, i)
                    push!(vals, 1.0)
                end
            end
        end
    end
    return sparse(rows, cols, vals, dim, dim), basis
end

function build_projectors(basis::Vector{Int})
    #Builds the projectors onto the even and odd parity sectors of the constrained basis.
    diag_plus = Float64[iseven(count_ones(state)) ? 1.0 : 0.0 for state in basis]
    diag_minus = 1.0 .- diag_plus
    return spdiagm(0 => diag_plus), spdiagm(0 => diag_minus)
end