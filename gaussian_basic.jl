using LinearAlgebra
using SparseArrays
using Arpack
using Base.Threads
using Printf
using Optim
using JLD2

const Id2 = sparse(I, 2, 2)
const c = sparse([0 1; 0 0]) # fermionic annihilation operator

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
    index = Dict(state => i for (i, state) in enumerate(basis))
    return basis, index
end

function apply_reversal!(ψ_out::AbstractVector, ψ_in::AbstractVector, L::Int)
    dim = length(ψ_in)

    Threads.@threads for i in 0:(dim-1)
        rev_i = 0
        for j in 0:(L-1)
            if (i >> j) & 1 == 1
                rev_i |= (1 << (L - 1 - j))
            end
        end
        
        @inbounds ψ_out[rev_i + 1] = ψ_in[i + 1]
    end
end

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
        abs(pivot) < 1e-14 && continue   # zero block, contributes 0 to Pf
        pf *= pivot
 
        # Schur complement update
        for i in (k+2):n, j in (k+2):n
            A[i, j] += (A[k, j] * A[i, k+1] - A[k, i] * A[j, k+1]) / pivot
        end
    end
    return pf
end

function build_bdg_state(U::Matrix{Float64}, V::Matrix{Float64}, L::Int)
    dim = 2^L
    ϵ = 1e-12 
    Z = V / (U + ϵ * I)
    Z = 0.5 * (Z - Z') # Ensure strict antisymmetry
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

function bdg_GS(A::Matrix{Float64}, B::Matrix{Float64}, L::Int)
    M = [A B; -B -A]
    vals, vecs = eigen(Matrix(M))
    vals = real.(vals); vecs = real.(vecs)
    ord = sortperm(vals)
    occ = ord[1:L]
    U = vecs[1:L, occ]
    V = vecs[L+1:2L, occ]

    return build_bdg_state(U, V, L)
end

function symmetrize_state(ψ0::Vector{Float64}, L::Int, sector_sign::Float64)
    Pψ0 = similar(ψ0)
    apply_reversal!(Pψ0, ψ0, L)

    ψ = ψ0 + sector_sign * Pψ0
    n = norm(ψ)
    n < 1e-14 && error("Norm is too small after symmetrization, likely due to destructive interference.")
    return ψ ./ n
end

function parity_project(ψ::Vector, sector::Int, L::Int)
    dim = length(ψ)
    ψ_proj = similar(ψ, Float64)
    @threads for i in 0:(dim-1)
        parity_val = iseven(count_ones(i)) ? 1.0 : -1.0
        @inbounds ψ_proj[i+1] = 0.5 * (1.0 + sector * parity_val) * ψ[i+1]
    end
    return ψ_proj
end

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

function cost_function(params::Vector, L::Int, ψ_scar::Vector, sector_sign::Float64, parity::Int, E_target::Float64; λ = 0.0)

    A, B = unpack_params(params, L)

    ψ0 = bdg_GS(A, B, L)
    ψ_sym = symmetrize_state(ψ0, L, sector_sign)
    ψ_proj = parity_project(ψ_sym, parity, L) # Project onto even parity sector
    n = norm(ψ_proj)
    n < 1e-14 && (println("Olap = 0.0"); return 0.0)
    olap = abs(dot(ψ_scar, ψ_proj))^2
    println("Overlap with scar: ", olap)
    return -olap
end

function optimize_gaussian(ψ_scar::Vector, L::Int, E_target::Float64;
                           sector_sign::Float64 = 1.0,
                           parity_sector::Int   = 1,
                           λ::Float64           = 0.0,
                           flip::Bool           = false,
                           max_nm::Int          = 10000,
                           max_lbfgs::Int       = 20000)

    ψ_scar_proj = parity_project(ψ_scar, parity_sector, L)
    proj_norm = norm(ψ_scar_proj)
    @printf("  ‖P̂_%+d |ψ_scar⟩‖ = %.6f  (expect ≈ 0.707)\n", parity_sector, proj_norm)
    ψ_scar_proj ./= proj_norm

    A0, B0 = initial_params(L, flip)
    params0 = pack_params(A0, B0, L)
    obj(p) = cost_function(p, L, ψ_scar_proj, sector_sign, parity_sector, E_target; λ = λ)

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
    ψ_opt_sym = symmetrize_state(ψ_opt, L, sector_sign)
    ψ_opt_proj = parity_project(ψ_opt_sym, parity_sector, L)

    olap_final = 4 * abs(dot(ψ_scar, ψ_opt_proj))^2
    return A_opt, B_opt, ψ_opt_proj, olap_final
end

function main()
    L = 14
    E_targ = 1.33
    idx =  7 + floor(Int, E_targ)
    folder = "dataN$(L)"
    file = "$(folder)/scar_$(idx)_E$(floor(Int, E_targ)).jld2"
    # sector_sign = 1.0
    # parity_sector = 1
    ψ = jldopen(file, "r") do f
        read(f, "state")
    end
    Pψ  = similar(ψ); apply_reversal!(Pψ, ψ, L)
    sector_sign   = real(dot(ψ, Pψ)) > 0 ? 1.0 : -1.0
    parity_sector = (L % 4 == 0) ? 1 : -1
    flip    = (parity_sector == -1)
    println("Loaded scar state from ", file, " with size: ", length(ψ))
    println("E = ", E_targ, " | Sector sign: ", sector_sign, " | Parity sector: ", parity_sector)
    A_opt, B_opt, ψ_opt, olap_final = optimize_gaussian(ψ, L, E_targ; 
                                                        sector_sign = sector_sign, 
                                                        parity_sector = parity_sector, 
                                                        λ = 0.0, 
                                                        flip = flip)

    println("Final overlap with scar: ", olap_final)
    jldsave("$(folder)/gaussian_N$(L)_E$(floor(Int, E_targ)).jld2"; A=A_opt, B=B_opt, ψ_opt=ψ_opt)
    return A_opt, B_opt, ψ_opt, olap_final
end
