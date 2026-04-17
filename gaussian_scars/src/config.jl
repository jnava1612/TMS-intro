abstract type GaussianSolver end

struct FullMatrixSolver <: GaussianSolver end

Base.@kwdef struct BitwiseSolver <: GaussianSolver
    maxiter::Int = 1000
    tol::Float64 = 1e-10
end

Base.@kwdef struct OptimizationConfig
    L::Int = 14
    λ::Float64 = 0.0
    solver::GaussianSolver = FullMatrixSolver()
    max_nm::Int = 50
    max_lbfgs::Int = 50
    z2_threshold::Float64 = 1e-5
    ent_threshold::Float64 = 1e-5
    output_dir::String = "data"
end

Base.@kwdef struct PlotConfig
    sizes::Vector{Int} = [8, 10, 12, 14]
    z2_thresholds::Vector{Float64} = [1e-5, 1e-5, 1e-5, 1e-5]
    ent_thresholds::Vector{Float64} = [0.5, 0.4, 0.35, 0.31]
    output_dir::String = "data"
end

Base.@kwdef struct BenchmarkConfig
    L::Int = 20
    maxiter::Int = 1000
    tol::Float64 = 1e-10
end