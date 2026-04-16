function apply_spatial_inversion!(ψ_out::AbstractVector, ψ_in::AbstractVector, reversal_map::Vector{Int})
    @assert length(ψ_out) == length(ψ_in) == length(reversal_map)
    fill!(ψ_out, 0.0)  # safety
    @inbounds for i in eachindex(ψ_in)
        ψ_out[reversal_map[i]] = ψ_in[i]
    end
end

function symmetrize_state(ψ0::AbstractVector, reversal_map::Vector{Int}, sign::Float64)
    πψ0 = similar(ψ0)
    apply_spatial_inversion!(πψ0, ψ0, reversal_map)
    ψ = ψ0 + sign * πψ0
    return ψ ./ norm(ψ)
end

function embed_to_full(ψ_restricted::Vector, basis::Vector{Int}, L::Int)
    # Embeds a vector defined on a restricted basis into the full Hilbert space of dimension 2^L.
    dim = 2^L
    ψ_full = zeros(Float64, dim)
    for (i, state) in enumerate(basis)
        ψ_full[state+1] = ψ_restricted[i]
    end
    return ψ_full
end

function project_to_fib(ψ_full::Vector, basis::Vector{Int})
    # Projects a state defined in the full Hilbert space onto the restricted basis 
    #subspace defined by the basis.
    ψ_proj = similar(basis, Float64)
    for (i, state) in enumerate(basis)
        ψ_proj[i] = ψ_full[state + 1]
    end
    return ψ_proj
end