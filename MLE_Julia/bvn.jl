using Pkg, Plots, LinearAlgebra, Distributions, Statistics
using Random, JuMP, Ipopt, Test, Cbc

"""
    bvn_mle()
Compute the maximum likelihood estimate (MLE) of the parameters of a bivariate normal distribution
using nonlinear optimization.
"""

function bvn_mle(n; verbose = false, plot = false)

    Random.seed!(420)

    bvn = MvNormal([1.5, 3.0], [1.0 0.85; 0.85 1.0])  # True values for distribution
    data = rand(bvn, n)

    X1 = data[1, :]
    X2 = data[2, :]

    if plot

        display(scatter(X1, X2))

    end

    mvn = Model(with_optimizer(Ipopt.Optimizer))
    MOI.set(mvn, MOI.RawParameter("print_level"), 1)

    set_silent(mvn)

    @variable(mvn, μ_x, start = 0.0)
    @variable(mvn, μ_y, start = 0.0)

    @variable(mvn, σ_x >= 0.0, start = 0.0)
    @variable(mvn, σ_y >= 0.0, start = 0.0)

    @variable(mvn, -1.0 <= ρ <= 1.0, start = 0.0)

    @NLobjective(mvn, Max,
                 -n * (log(σ_x) + log(σ_y) + 0.5 * log(1 - ρ^2)) -
                 0.5/(1 - ρ^2) * sum(((X1[i] - μ_x)^2)/σ_x^2 + ((X2[i] - μ_y)^2)/σ_y^2 -
                 2.0 * ρ * ((X1[i] - μ_x) * (X2[i] - μ_y))/(σ_x * σ_y) for i in 1:n))

    optimize!(mvn)

    if verbose

        println("μ_x = ", JuMP.value(μ_x))
        println("μ_t = ", JuMP.value(μ_y))

        println("σ_x = ", JuMP.value(σ_y))
        println("σ_y = ", JuMP.value(σ_y))
        println("ρ = ", JuMP.value(ρ))

        println("MLE objective: ", JuMP.objective_value(mvn))

    end

    @test JuMP.value(μ_x) ≈ mean(X1) atol = 1e-3
    @test JuMP.value(μ_y) ≈ mean(X2) atol = 1e-3
    @test JuMP.value(σ_x) ≈ var(X1) atol = 1e-2
    @test JuMP.value(σ_y) ≈ var(X2) atol = 1e-2

end

n_ = parse(Int64, ARGS[1])
verbose_ = parse(Bool, ARGS[2])
plot_ = parse(Bool, ARGS[3])

bvn_mle(n_, verbose = verbose_, plot = plot_)
