# Implementation of the Gaussian Hamiltonian and its ground state computation 
# using a bitwise representation of the state vector.

function build_term_lists(A, B)
    L = size(A, 1)

    Aterms = Tuple{Int, Int, eltype(A)}[]
    Bterms = Tuple{Int, Int, eltype(B)}[]

    for j in 1:L, k in 1:L
        if !iszero(A[j, k])
            push!(Aterms, (j, k, ComplexF64(A[j, k])))
        end
        if !iszero(B[j, k]) 
            push!(Bterms, (j, k, ComplexF64(B[j, k])))
        end
    end
    return Aterms, Bterms
end

function _bitwise_gaussian_action_chunk(state_range, ψ, Aterms, Bterms)
    T = promote_type(eltype(ψ), Float64)
    out = zeros(T, length(ψ))

    @inbounds for idx in state_range
        amp = ψ[idx]
        iszero(amp) && continue

        state = UInt64(idx - 1)
        for (j, k, A_jk) in Aterms
            ok, new_state, sign = apply_cdag_c(state, j, k)
            if ok
                out[Int(new_state) + 1] += A_jk * sign * amp
            end
        end
        for (j, k, B_jk) in Bterms
            ok, new_state, sign = apply_cdag_cdag(state, j, k)
            if ok
                out[Int(new_state) + 1] += 0.5 * B_jk * sign * amp
            end

            ok, new_state, sign = apply_c_c(state, j, k)
            if ok
                out[Int(new_state) + 1] -= 0.5 * conj(B_jk) * sign * amp
            end
        end
    end

    return out
end

function bitwise_gaussian_action(ψ, A, B)
    L = size(A, 1)
    dim = length(ψ)
    @assert dim == 2^L "State vector length must be 2^L"
    @assert L ≤ 63 "Current bitwise implementation supports up to 64 sites"

    Aterms, Bterms = build_term_lists(A, B)

    nchunks = min(dim, nthreads())
    chunk_size = cld(dim, nchunks)
    ranges = [s:min(s + chunk_size - 1, dim) for s in 1:chunk_size:dim]

    tasks = [
        Threads.@spawn _bitwise_gaussian_action_chunk(rng, ψ, Aterms, Bterms)
        for rng in ranges
    ]
    partials = fetch.(tasks)
    out = zero(partials[1])
    for p in partials
        out .+= p
    end
    return out
end

function bitwise_gaussian_groundstate(A, B; ψstart=nothing, maxiter=1000, tol=1e-10)
    L = size(A, 1)
    dim = 2^L

    if isnothing(ψstart)
        ψstart = randn(ComplexF64, dim)
    else
        ψstart = ComplexF64.(copy(ψstart))
    end
    ψstart ./= norm(ψstart)

    action = ψ -> bitwise_gaussian_action(ψ, A, B)

    vals, vecs, info = eigsolve(action, ψstart, 1, :SR; maxiter=maxiter, tol=tol)

    ψg = Vector(vecs[1])
    ψg ./= norm(ψg)

    return real(vals[1]), ψg, info
end


# Implementation of the Gaussian Hamiltonian and its ground state computation using a dense matrix representation.

function full_c(idx::Int, L::Int)
    σz = sparse([1 0; 0 -1])
    c = sparse([0.0 1.0; 0.0 0.0])
    ops = Vector{SparseMatrixCSC{Float64, Int}}(undef, L)
    for site in 1:L
        if site < idx
            ops[site] = σz
        elseif site == idx
            ops[site] = c
        else
            ops[site] = sparse(I, 2, 2)
        end
    end
    op = ops[1]
    for site in 2:L
        op = kron(ops[site], op)
    end
    return op
end

function _gaussian_hamiltonian_chunk(j_range, L, dim, A, B, all_c, all_cd)
    H_chunk = spzeros(ComplexF64, dim, dim)
    for j in j_range
        for k in 1:L
            if !iszero(A[j,k])
                H_chunk += A[j,k] * (all_cd[j] * all_c[k])
            end
            if !iszero(B[j,k])
                H_chunk += 0.5 * (
                    B[j,k] * (all_cd[j] * all_cd[k]) -
                    conj(B[j,k]) * (all_c[j] * all_c[k])
                )
            end
        end
    end
    return H_chunk
end

function gaussian_hamiltonian(L::Int, A::AbstractMatrix, B::AbstractMatrix,
                              all_c, all_cd)
    @assert size(A) == (L, L)
    @assert size(B) == (L, L)

    dim = 2^L

    nchunks = min(L, Threads.nthreads())
    chunk_size = cld(L, nchunks)
    chunks = [s:min(s + chunk_size - 1, L) for s in 1:chunk_size:L]

    tasks = [
        Threads.@spawn _gaussian_hamiltonian_chunk(r, L, dim, A, B, all_c, all_cd)
        for r in chunks
    ]

    partials = fetch.(tasks)

    H = spzeros(ComplexF64, dim, dim)
    for part in partials
        H += part
    end
    return H
end

# Function to verify that the bitwise implementation of the Gaussian Hamiltonian action 
# matches the dense matrix version for random A and B.

function verify_bitwise_action(L::Int)
    all_c = [full_c(j, L) for j in 1:L]
    all_cd = [op' for op in all_c]

    A = randn(L, L); A = 0.5 .* (A + A')
    B = randn(L, L); B = 0.5 .* (B - B')

    ψ = randn(ComplexF64, 2^L)
    ψ ./= norm(ψ)
    y = copy(ψ)

    H = gaussian_hamiltonian(L, A, B, all_c, all_cd)
    y_matrix = H * ψ
    y_bitwise = bitwise_gaussian_action(y, A, B)

    println("||Hψ(matrix) - Hψ(bitwise)|| = ", norm(y_matrix - y_bitwise))
end
