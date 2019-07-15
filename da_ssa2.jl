module DA_SSA2

include("integrators.jl")
include("embedding.jl")
using .Integrators
using .Embedding

#using Distributed
using Distributions
using LinearAlgebra

#addprocs()

function ETKF(E::Array{Float64, 2}, model::Function, model_err::Function,
              R::Symmetric{Float64, Array{Float64, 2}}, m::Int64, tree, osc,
              pcs, k, D, M, oracle; H=I, Δt::Float64=0.1, window::Float64=1.0, cycles::Int64=100,
              ave_window=120, outfreq=5)
    if H != I
        p, n = size(H)
    else
        p, n = size(R)
    end

    x_hist = zeros(cycles, D)

    R_inv = inv(R)
    obs_err = MvNormal(zeros(p), R)

    x_true = E[:, end]
    x_free = x_true + randn(n)

    errs = []

    errs_free = []

    B = zeros(n, n)

    for cycle=1:cycles
        println(cycle)
        y = zeros(p)
        y[1:3] = (H*x_true + rand(obs_err))[1:3]
        x_m = mean(E, dims=2)[:, 1]
        #proj = project(tree, (x_m'*pcs)', osc, k, 1)[2, :]
        if cycle == 1
            x_old = zeros(D)[:, 1:1]
        else
            x_old = x_hist[max(1, cycle - ave_window - 1):(cycle - 1), :]'
        end
        y[4:6] = oracle[cycle, :]
        #y[D+1:end] = mean(x_old, dims=2) + proj

        X = (E .- x_m)/sqrt(m - 1)
        B = B + X*X'
        Y = (H*E .- H*x_m)/sqrt(m - 1)
        Ω = real((I + Y'*R_inv*Y)^(-1))
        w = Ω*Y'*R_inv*(y - H*x_m)

        E = real(x_m .+ X*(w .+ sqrt(m - 1)*Ω^(1/2)))
        err = sqrt(mean((mean(E, dims=2) .- x_true).^2))
        append!(errs, err)
        append!(errs_free, sqrt(mean((x_free .- x_true).^2)))
        x_hist[cycle, :] = mean(E, dims=2)

        E = hcat(map((col)->rk4_inplace(model_err, col, 0.0, window, Δt), E[:, i] for i=1:m)...)

        x_true = rk4_inplace(model, x_true, 0.0, window, Δt)
        x_free = rk4_inplace(model_err, x_free, 0.0, window, Δt)
    end
    return errs, errs_free, x_hist, B
end

end
