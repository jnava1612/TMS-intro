include(joinpath(@__DIR__, "..", "src", "GaussianScars.jl"))
using .GaussianScars

cfg = OptimizationConfig(
    L = 14,
    λ = 0.0,
    max_nm = 100,
    max_lbfgs = 200,
    z2_threshold = 1e-5,
    ent_threshold = 0.31,
    output_dir = "../data/",
)

run_optimization(cfg)