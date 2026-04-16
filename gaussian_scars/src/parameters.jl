function pack_params(A, B, L::Int)
    params = Float64[]
    for i in 1:L, j in i:L;     push!(params, A[i, j]); end
    for i in 1:L, j in (i+1):L; push!(params, B[i, j]); end
    return params
end

function unpack_params(params::Vector{Float64}, L::Int)
    A   = zeros(L, L)
    B   = zeros(L, L)
    idx = 1
    for i in 1:L, j in i:L
        A[i, j] = A[j, i] = params[idx]; idx += 1
    end
    for i in 1:L, j in (i+1):L
        B[i, j]  =  params[idx]
        B[j, i]  = -params[idx]
        idx += 1
    end
    return A, B
end


function initial_params(L::Int, flip::Bool = false; mode = :zero)
    diag_A = [isodd(i) ? -1.0 : 1.0 for i in 1:L]
    flip && (diag_A[1] = 1.0) # Flip the first element to break symmetry if needed
    if mode == :zero
        B = spzeros(L, L) 
    elseif mode == :random
         B = randn(L, L) * 0.01
         B = B - B'  # Ensure B is antisymmetric
    else
        error("Unknown mode: $mode. Use :zero or :random.")
    end
    return Diagonal(diag_A) |> Matrix, B
end
