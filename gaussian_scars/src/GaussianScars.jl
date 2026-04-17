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

include("config.jl")
include("gaussian_hamiltonian.jl")
include("operator_actions.jl")
include("optimization.jl")
include("parameters.jl")
include("plotting.jl")
include("pxp_model.jl")
include("scar_extraction.jl")
include("states.jl")



export OptimizationConfig, PlotConfig, BenchmarkConfig 
export GaussianSolver, FullMatrixSolver, BitwiseSolver

export build_term_lists, bitwise_gaussian_action, bitwise_gaussian_groundstate
export verify_bitwise_action, full_c, gaussian_hamiltonian

export fermion_sign, annihilate, create, apply_cdag_cdag, apply_c_c, apply_cdag_c

export cost, optimize_gaussian, get_ground_state

export pack_params, unpack_params, initial_params

export mytheme, latex_ticks

export constrainedBasis, build_reversal_map, build_pxp_hamiltonian, build_projectors

export entropy, scar_tower, view_scars

export apply_spatial_inversion!, symmetrize_state, embed_to_full, project_to_fib

export run_optimization, plot_initial_states, benchmark_gaussian_hamiltonian


end