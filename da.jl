using Distributed
using Distributions
using LinearAlgebra

#addprocs()

m = 20  # ensemble size

@everywhere function lorenz(t, u)
    σ = 10
    β = 8/3
    ρ = 28
    du = zeros(3)
    du[1] = σ*(u[2] - u[1])
    du[2] = u[1]*(ρ - u[3]) - u[2]
    du[3] = u[1]*u[2] - β*u[3]
    return du
end

@everywhere function rk4(f::Function, y0, t0::Float64, t1::Float64, h::Float64)
    y = y0
    n = (t1 - t0)÷h
    t = t0
    for i in 1:n
        k1 = h * f(t, y)
        k2 = h * f(t + 0.5*h, y + 0.5*k1)
        k3 = h * f(t + 0.5*h, y + 0.5*k2)
        k4 = h * f(t + h, y + k3)
        y = y + (k1 + 2*k2 + 2*k3 + k4)/6
        t = t0 + i*h
    end
    return y
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
E = hcat([rk4(lorenz, x0, 0.0, last, 0.1) for last=range(10.0, stop=100.0, length=m)]...)

function run_da(E)
    Δt = 0.1
    H = Diagonal(ones(3))
    R = Diagonal(ones(3))
    R_inv = inv(R)
    obs_err = MvNormal(zeros(3), diag(R))

    x_true = E[:, end]
    x_free = x_true + rand(obs_err)

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

        E = hcat(pmap((col)->rk4(lorenz, col, 0.0, Δt, Δt), E[:, i] for i=1:m)...)

        x_true = rk4(lorenz, x_true, 0.0, Δt, Δt)
        x_free = rk4(lorenz, x_free, 0.0, Δt, Δt)
    end
    return errs, errs_free
end
