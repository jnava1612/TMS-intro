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

function scar_tower(L::Int; z2_threshold::Float64=1e-5, ent_threshold::Float64=0.31, 
                    save_data::Bool=false, path::String="")
    S_vol = L * log(2) / 2
    H, basis  = build_pxp_hamiltonian(L)
    
    Z2_i = sum(1 << i for i in 0:2:(L-1))
    Z2 = zeros(Float64, length(basis))

    for (i, state) in enumerate(basis)
        if state == Z2_i
            Z2[i] = 1.0
            break
        end
    end

    E, V      = eigen(Matrix(H))
    olaps = [abs(dot(Z2, V[:,k]))^2 for k in 1:length(E)]
    entropies = [entropy(V[:,k], basis, L) for k in 1:length(E)]
    
    scar_idx  = [
        k for k in eachindex(E) 
        if olaps[k] > z2_threshold && 
           entropies[k] < ent_threshold*S_vol && 
           E[k] > 1e-10]

    println("Identified scar states at indices: ", scar_idx)

    E = E[scar_idx]
    V = V[:, scar_idx]
    entropies = entropies[scar_idx]
    olaps = olaps[scar_idx]
    
    perm = sortperm(E)

    E = E[perm]
    V = V[:, perm]
    entropies = entropies[perm]
    olaps = olaps[perm]

    if save_data
        @save joinpath(path, "scar_data_L$(L).jld2") E V entropies olaps
    end

    return E, V, entropies, olaps
end

function view_scars(L::Int; z2_threshold=1e-5, ent_threshold=0.31)
    # This function is for visualizing the identified scar states in terms of 
    # their energy, overlap with the Z2 state, and entanglement entropy.
    # Helps to verify if the thresholds are correctly identifying scar states 
    # and to understand their properties.
    E, _, _, olaps = scar_tower(L; z2_threshold=z2_threshold, ent_threshold=ent_threshold)
    fig  = Figure(size=(800, 600))
    ax = Axis(fig[1, 1], xlabel=L"E/N", ylabel = L"\log_{10} |\langle Z_2 | \Psi_n \rangle|^2")
    scatter!(ax, E, log10.(olaps), color=:blue, markersize=8)
    display(fig)
end
