using LinearAlgebra
using SparseArrays
using Arpack
using Base.Threads
using Printf
using Optim
using JLD2
using CairoMakie

#==========================================================================================#
# Builder Functions
#==========================================================================================#


function constrainedBasis(N::Int)
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

function build_bdg_state(U::Matrix{Float64}, V::Matrix{Float64}, L::Int)
    dim = 2^L
    Z = V * pinv(U)
    ψ = zeros(Float64, dim)

    @threads for i in 0:(dim-1)
        occupied = Int[]
        for j in 0:(L-1)
            (i>>j) & 1 == 1 && push!(occupied, j+1)
        end
        n = length(occupied)
        iseven(n) || continue
        if n == 0
            @inbounds ψ[i+1] = 1.0
            continue
        end
        Zsub = Z[occupied, occupied]
        @inbounds ψ[i+1] = pfaffian(Matrix(Zsub))
    end

    nm = norm(ψ)
    nm < 1e-14 && error("Norm is too small, likely due to numerical instability in pfaffian calculation.")
    ψ ./= nm
    return ψ
end

function build_projectors(basis::Vector{Int}, L::Int)
    sigma_z = sparse([1 0; 0 -1])
    chiral = sigma_z
    for j in 2:L
        chiral = kron(sigma_z, chiral)
    end
    
    values, vectors = eigen(Matrix(chiral))
    
    positive_vectors = [vectors[:,i] for i in 1:length(values) if values[i]>0.0]
    negative_vectors = [vectors[:,i] for i in 1:length(values) if values[i]<0.0]

    P_plus = spzeros(Float64, size(basis, 1), size(basis, 1))
    for i in eachindex(positive_vectors)
        vector_projected = sparse(project_to_fib(positive_vectors[i], basis))
        if !isnan(vector_projected[1])
            P_plus += kron(vector_projected', vector_projected)
        end
    end

    P_minus = spzeros(Float64, size(basis, 1), size(basis, 1))
    for i in eachindex(negative_vectors)
        vector_projected = sparse(project_to_fib(negative_vectors[i], basis))
        if !isnan(vector_projected[1])
            P_minus += kron(vector_projected', vector_projected)
        end
    end
    return P_plus, P_minus
end


#==========================================================================================#
# State Modifier Functions
#==========================================================================================#

function apply_reversal!(ψ_out::AbstractVector, ψ_in::AbstractVector, reversal_map::Vector{Int})
    @assert length(ψ_out) == length(ψ_in) == length(reversal_map)
    fill!(ψ_out, 0.0)  # safety
    @inbounds for i in eachindex(ψ_in)
        ψ_out[reversal_map[i]] = ψ_in[i]
    end
end

function symmetrize_state(ψ0::Vector{Float64}, L::Int, reversal_map::Vector{Int})
    Pψ0 = similar(ψ0)
    apply_reversal!(Pψ0, ψ0, reversal_map)

    if norm(ψ0 + Pψ0) < 1e-14
        ψ = ψ0 - Pψ0
    else norm(ψ0 - Pψ0) < 1e-14
        ψ = ψ0 + Pψ0
    end

    return ψ ./ norm(ψ)
end

function embed_to_full(ψ_restricted::Vector, basis::Vector{Int}, L::Int)
    dim = 2^L
    ψ_full = zeros(Float64, dim)
    for (i, state) in enumerate(basis)
        ψ_full[state+1] = ψ_restricted[i]
    end
    return ψ_full./ norm(ψ_full)
end

function project_to_fib(ψ_full::Vector, basis::Vector{Int})
    ψ_proj = similar(basis, Float64)
    for (i, state) in enumerate(basis)
        ψ_proj[i] = ψ_full[state + 1]
    end
    return ψ_proj./ norm(ψ_proj)
end



#==========================================================================================#
# Scar Tower Extraction Functions
#==========================================================================================#

function entropy(ψ, basis, L)
    Nhalf = L ÷ 2
    dimL = 2^Nhalf
    dimR = 2^(L - Nhalf)

    ψ_full = zeros(Float64, 2^L)

    # embed into full space
    for (i, state) in enumerate(basis)
        ψ_full[state+1] = ψ[i]
    end

    ψ_mat = reshape(ψ_full, dimL, dimR)

    s = svdvals(ψ_mat)
    p = s.^2
    return -sum(p .* log.(p .+ 1e-12))
end
 
function scar_tower(L::Int; z2_threshold::Float64=1e-5, ent_threshold::Float64=0.31)
    S_vol = L * log(2) / 2
    H, basis  = build_pxp_hamiltonian(L)
    Z2_i = sum(1 << i for i in 0:2:(L-1))
    Z2 = zeros(Float64, length(basis))

    for (i, state) in enumerate(basis)
        if state == Z2_i
            Z2[i] = 1.0
        end
    end
    E, V      = eigen(Matrix(H))
    olaps = [abs(dot(Z2, V[:,k]))^2 for k in 1:length(E)]
    entropies = [entropy(V[:,k], basis, L) for k in 1:length(E)]
    scar_idx  = [k for k in eachindex(E) if olaps[k] > z2_threshold && entropies[k] < ent_threshold*S_vol && E[k] > 0.0]
    println("Identified scar states at indices: ", scar_idx)
    E = E[scar_idx]
    V = V[:, scar_idx]
    entropies = entropies[scar_idx]
    olaps = olaps[scar_idx]
    perm = sortperm(E)
    return E[perm], V[:, perm]
end

function view_scars(L::Int; z2_threshold=1e-5, ent_threshold=0.31)
    E, V, entropies, olaps = scar_tower(L; z2_threshold=z2_threshold, ent_threshold=ent_threshold)
    fig  = Figure(size=(800, 600))
    ax = Axis(fig[1, 1], xlabel=L"E/N", ylabel = L"\log_{10} |\langle Z_2 | \Psi_n \rangle|^2")
    scatter!(ax, E, log10.(olaps), color=:blue, markersize=8)
    display(fig)
end



#==========================================================================================#
# Cost Function Components and Helpers
#==========================================================================================#


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

function initial_params(L::Int, flip::Bool = false)
    diag_A = [isodd(i) ? -1.0 : 1.0 for i in 1:L]
    flip && (diag_A[1] = 1.0) # Flip the first element to break symmetry if needed
    B = randn(L, L) * 0.01
    B = B - B'  # Ensure B is antisymmetric
    return Diagonal(diag_A) |> Matrix, B
end

function cost_function(params::Vector, L::Int, ψ_scar::Vector, reversal_map::Vector{Int}, E_target::Float64; 
                       λ::Float64 = 0.0, basis = nothing, H = nothing)

    A, B = unpack_params(params, L)

    ψ0 = bdg_GS(A, B, L)
    
    ψ_sym = symmetrize_state(ψ0, L, reversal_map)
    
    olap = 4 * abs(dot(ψ_scar, ψ_sym))^2
    if olap < 1e-14
        println("Overlap is very small: ", olap)
        return 1e6  # Large penalty for zero overlap
    end
    var = 0.0
    if !isnothing(basis) && !isnothing(H) && λ > 0.0
        ψ_red = project_to_fib(ψ_sym, basis)

        E_current = real(dot(ψ_red, H * ψ_red))
        var = (E_current - E_target)^2
        
        println("Overlap with scar: ", olap, " | Energy: ", E_current, " (target: ", E_target, ")")
    else
        println("Overlap with scar: ", olap, "(E_target: ", E_target, ")")
    end

    return -olap + λ * var
end

function optimize_gaussian(ψ_scar::Vector, L::Int, E_target::Float64, reversal_map::Vector{Int};
                           λ::Float64           = 0.0,
                           basis = nothing,
                           H = nothing,
                           flip_first::Bool     = false,
                           max_nm::Int          = 500,
                           max_lbfgs::Int       = 1000)

    A0, B0 = initial_params(L, flip_first)
    params0 = pack_params(A0, B0, L)
    
    obj(p) = cost_function(p, L, ψ_scar, reversal_map, E_target; λ = λ, basis = basis, H = H)

    res1 = optimize(obj, params0, NelderMead(), 
                    Optim.Options(iterations = max_nm, show_trace = true))

    println("Nelder-Mead result: ", res1)
    
    res2 = optimize(obj, res1.minimizer, LBFGS(), 
                    Optim.Options(g_tol = 1e-20, 
                                  x_abstol = 1e-12, 
                                  f_abstol = 1e-12, 
                                  iterations = max_lbfgs, 
                                  show_trace = true))

    println("LBFGS result: ", res2)

    A_opt, B_opt = unpack_params(res2.minimizer, L)
    ψ_opt = bdg_GS(A_opt, B_opt, L)
    ψ_opt_sym = symmetrize_state(ψ_opt, L, reversal_map)

    olap_final = 4 * abs(dot(ψ_scar, ψ_opt_sym))^2

    if !isnothing(H)
        ψ_red = project_to_fib(ψ_opt_sym, basis)
        E_final = real(dot(ψ_red, H * ψ_red))
        println("Final overlap: ", olap_final, " | Final energy: ", E_final, " (target: ", E_target, ")")
    else
        println("Final overlap: ", olap_final, " (E_target: ", E_target, ")")
    end

    return A_opt, B_opt, ψ_opt_sym, olap_final
end


#==========================================================================================#
# Optimization Functions
#==========================================================================================#


function pfaffian(M::Matrix{Float64})
    n = size(M, 1)
    @assert iseven(n) "Pfaffian requires even-dimensional matrix"
    n == 0 && return 1.0
    n == 2 && return M[1, 2]
 
    A = copy(M)
    pf = 1.0
    for k in 1:2:(n-1)
        # pivot: largest |A[k, j]| for j > k
        _, piv_j = findmax(abs.(A[k, k+1:end]))
        piv_j += k                        # absolute index
 
        if piv_j != k + 1
            # swap rows/cols k+1 and piv_j
            A[[k+1, piv_j], :] = A[[piv_j, k+1], :]
            A[:, [k+1, piv_j]] = A[:, [piv_j, k+1]]
            pf = -pf                      # each swap flips sign
        end
 
        pivot = A[k, k+1]
        abs(pivot) < 1e-14 && return 0.0 # Numerical instability, treat as zero
        pf *= pivot
 
        # Schur complement update
        for i in (k+2):n, j in (k+2):n
            A[i, j] += (A[k, j] * A[i, k+1] - A[k, i] * A[j, k+1]) / pivot
        end
    end
    return pf
end

function bdg_GS(A::Matrix{Float64}, B::Matrix{Float64}, L::Int)
    M = [A B; -B -A]
    vals, vecs = eigen(Matrix(M))
    vals = real.(vals); vecs = real.(vecs)
    occ = findall(vals .< 0)
    length(occ) != L && error("Expected exactly L negative eigenvalues, got ", length(occ))
    U = vecs[1:L, occ]
    V = vecs[L+1:2L, occ]

    for i in 1:L
        nrm = sqrt(norm(U[:,i])^2 + norm(V[:,i])^2)
        U[:,i] /= nrm
        V[:,i] /= nrm
    end

    return build_bdg_state(U, V, L)
end



#==========================================================================================#
# Main Function
#==========================================================================================#

function main()
    L = 14
    λ = 0.0
    flip_first = false
    max_nm = 50
    max_lbfgs = 1000


    folder = "dataN$(L)"
    mkpath(folder)
    E, V = scar_tower(L; z2_threshold=1e-5, ent_threshold=0.31)
    H, basis = build_pxp_hamiltonian(L)
    P_plus, P_minus = build_projectors(basis, L)
    reversal_map = build_reversal_map(L)

    for (i, E_targ) in enumerate(E)
        ψ = V[:, i]
        ψ_plus = P_plus * ψ
        ψ_minus = P_minus * ψ

        ψ_plus_emb = embed_to_full(ψ_plus, basis, L)
        ψ_minus_emb = embed_to_full(ψ_minus, basis, L)
        
        A_p, B_p, ψ_p, olap_p = optimize_gaussian(ψ_plus, L, E_targ, reversal_map;
                                                  basis = basis, H = H, λ = λ, 
                                                  flip_first = flip_first,
                                                  max_nm = max_nm, max_lbfgs = max_lbfgs)

        println("Optimized overlap for parity-even sector: ", olap_p)

        jldsave("$(folder)/scar_$(i)_even.jld2"; A=A_p, B=B_p, ψ=ψ_p, overlap=olap_p)

        A_m, B_m, ψ_m, olap_m = optimize_gaussian(ψ_minus, L, E_targ, reversal_map;
                                                  basis = basis, H = H, λ = λ, 
                                                  flip_first = flip_first,
                                                  max_nm = max_nm, max_lbfgs = max_lbfgs)
        println("Optimized overlap for parity-odd sector: ", olap_m)

        jldsave("$(folder)/scar_$(i)_odd.jld2"; A=A_m, B=B_m, ψ=ψ_m, overlap=olap_m)

    end
    
    println("All optimizations completed.")
end
