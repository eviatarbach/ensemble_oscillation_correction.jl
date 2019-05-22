using Distributed
using DifferentialEquations
using Plots
using Distributions
using LinearAlgebra

#addprocs()

m = 20  # ensemble size

σ = 10
β = 8/3
ρ = 28

function lorenz(du, u, p, t)
 du[1] = σ*(u[2] - u[1])
 du[2] = u[1]*(ρ - u[3]) - u[2]
 du[3] = u[1]*u[2] - β*u[3]
end

abstract type Model end

struct ETKF
    ens::Array{Float64, 2}
    m::Int64
    H::Array{Float64, 2}
    R::Array{Float64, 2}
    R_inv::Array{Float64, 2}
    model::Model
    function ETKF(ens::Array{Float64, 2}, m::Int64, H::Array{Float64, 2},
                  R::Array{Float64, 2}, model::Model)
        R_inv = inv(R)
    end
end

x0 = [1.0; 0.0; 0.0]
tspan = (0.0, 100.0)
prob = ODEProblem(lorenz, x0, tspan)
sol = solve(prob)

function run_da()
    Δt = 0.1
    H = Diagonal(ones(3))
    R = Diagonal(ones(3))
    R_inv = inv(R)
    obs_err = MvNormal(zeros(3), diag(R))

    E = hcat(sol.u[end-(m-1)*10:10:end]...)
    ens = [init(ODEProblem(lorenz, E[:, i], (0.0, 1.0)), Tsit5()) for i=1:m]

    x_true = sol.u[end]
    x_free = x_true + rand(obs_err)

    int_true = init(ODEProblem(lorenz, x_true, tspan), Tsit5())
    int_free = init(ODEProblem(lorenz, x_free, tspan), Tsit5())

    errs = []

    errs_free = []

    B = zeros(3, 3)

    for t=1:100
        y = x_true + rand(obs_err)
        x_m = mean(E, dims=2)

        X = (E .- x_m)/sqrt(m - 1)
        B = B + X*X'
        Y = (H*E .- H*x_m)/sqrt(m - 1)
        Ω = real((I + Y'*R_inv*Y)^(-1))
        w = Ω*Y'*R_inv*(y - H*x_m)

        E = real(x_m .+ X*(w .+ sqrt(m - 1)*Ω^(1/2)))
        err = sqrt(mean((mean(E, dims=2) .- x_true).^2))
        append!(errs, err)
        append!(errs_free, sqrt(mean((x_free .- x_true).^2)))

        E = hcat([solve(ODEProblem(lorenz, E[:, i], (0.0, Δt))).u[end] for i=1:m]...)

        step!(int_true, Δt, true)
        step!(int_free, Δt, true)

        x_true = int_true.u
        x_free = int_free.u
    end
    return errs, errs_free
end
#for t = 1:100
#    y = x + rand(obs_err)
#end
#plot(sol, vars=(1, 2))
