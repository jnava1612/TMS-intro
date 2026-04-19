include(joinpath(@__DIR__, "..", "src", "GaussianScars.jl"))
using .GaussianScars

cfg = OptimizationConfig(
    L = 12,
    λ = 0.0,
    max_nm = 20,
    max_lbfgs = 100,
    z2_threshold = 1e-5,
    ent_threshold = 0.35,
    output_dir = "../data/",
    solver = BitwiseSolver(),
)
run_optimization(cfg)