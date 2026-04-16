using LinearAlgebra
using SparseArrays
using Arpack
using Base.Threads


#==========================================================================================#
# Fermionic operators and their application to states represented as UInt64 bitstrings.
#==========================================================================================#

@inline function fermion_sign(state::UInt64, site::Int)
    lower_mask = (UInt64(1) << (site - 1)) - UInt64(1)
    return isodd(count_ones(state & lower_mask)) ? -1.0 : 1.0
end

@inline function annihilate(state::UInt64, site::Int)
    bit = UInt64(1) << (site - 1)
    if (state & bit) == 0
        return false, UInt64(0), 0.0
    end
    return true, state ⊻ bit, fermion_sign(state, site)
end

@inline function create(state::UInt64, site::Int)
    bit = UInt64(1) << (site - 1)
    if (state & bit) != 0
        return false, UInt64(0), 0.0
    end
    return true, state | bit, fermion_sign(state, site)
end

@inline function apply_cdag_cdag(state::UInt64, j::Int, k::Int)
    ok1, state1, sign1 = create(state, k)
    ok1 || return false, UInt64(0), 0.0

    ok2, state2, sign2 = create(state1, j)
    ok2 || return false, UInt64(0), 0.0

    return true, state2, sign1 * sign2
end

@inline function apply_c_c(state::UInt64, j::Int, k::Int)
    ok1, state1, sign1 = annihilate(state, k)
    ok1 || return false, UInt64(0), 0.0

    ok2, state2, sign2 = annihilate(state1, j)
    ok2 || return false, UInt64(0), 0.0

    return true, state2, sign1 * sign2
end

@inline function apply_cdag_c(state::UInt64, j::Int, k::Int)
    ok1, state1, sign1 = annihilate(state, k)
    ok1 || return false, UInt64(0), 0.0

    ok2, state2, sign2 = create(state1, j)
    ok2 || return false, UInt64(0), 0.0

    return true, state2, sign1 * sign2
end