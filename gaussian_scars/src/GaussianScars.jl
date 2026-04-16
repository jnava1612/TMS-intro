module GaussianScars

using LinearAlgebra
using SparseArrays
using Arpack
using Base.Threads
using Printf
using Optim
using JLD2
using KrylovKit
using LaTeXStrings
using CairoMakie
using BenchmarkTools

include("operator_actions.jl")
# include("operators.jl")
# include("gaussian_hamiltonian.jl")
# include("pxp_model.jl")
# include("states.jl")
# include("scars.jl")
# include("parameters.jl")
# include("optimization.jl")
# include("config.jl")

# export fermion_sign, annihilate, create, apply_cdag_cdag, apply_c_c, apply_cdag_c
# export build_term_lists, bitwise_gaussian_action, gaussian_groundstate_bitwise
# export verify_bitwise_action, full_c, gaussian_hamiltonian
# export constrainedBasis, build_reversal_map, build_pxp_hamiltonian, build_projectors
# export apply_spatial_inversion!, symmetrize_state, embed_to_full, project_to_fib
# export entropy, scar_tower, view_scars
# export pack_params, unpack_params, initial_params
# export cost, optimize_gaussian
# export OptimizationConfig, PlotConfig, BenchmarkConfig
# export run_optimization, plot_initial_states, benchmark_gaussian_hamiltonian


end