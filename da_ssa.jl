module DA_SSA

export ETKF_SSA

include("integrators.jl")
include("embedding.jl")
using .Integrators
using .Embedding

using Distributions
using LinearAlgebra
using NearestNeighbors

function ETKF_SSA(E::Array{Float64, 2}, model, model_err,
                  R::Symmetric{Float64, Array{Float64, 2}}, m::Int64, D, M,
                  r1, r2, tree1, tree2; psrm=true, H=I, Δt::Float64=0.1,
                  window::Float64=0.4, cycles::Int64=1000, outfreq=40,
                  inflation=1.0, integrator=Integrators.rk4, osc_vars=1:D)
    if H != I
        p = size(H)[1]
    else
        p = size(R)[1]
    end

    H_osc = zeros(length(osc_vars), p)
    H_osc[[CartesianIndex(i) for i in zip(1:length(osc_vars), osc_vars)]] .= 1

    full_x_hist = []
    x_true_hist = []
    x_free_hist = []

    R_inv = inv(R)
    obs_err = MvNormal(zeros(p), R)

    x_true = E[:, end]
    x_free = x_true + randn(D)/5

    errs = []

    errs_free = []

    B = zeros(D, D)

    for cycle=1:cycles
        println(cycle)
        y = H*x_true + rand(obs_err)
        x_m = mean(E, dims=2)

        X = (E .- x_m)/sqrt(m - 1)
        X = x_m .+ inflation*(X .- x_m)
        #B = B*(cycle - 1) + X*X'
        #B = B/cycle
        Y = (H*E .- H*x_m)/sqrt(m - 1)
        Ω = real((I + Y'*R_inv*Y)^(-1))
        w = Ω*Y'*R_inv*(y - H*x_m)

        E = real(x_m .+ X*(w .+ sqrt(m - 1)*Ω^(1/2)))
        err = sqrt(mean((mean(E, dims=2) .- x_true).^2))
        append!(errs, err)
        append!(errs_free, sqrt(mean((x_free .- x_true).^2)))

        for i=1:m
            E[:, i] = integrator(model_err, E[:, i], 0.0, window, Δt)
            if psrm
                inc = -r2[knn(tree2, E[osc_vars, i], 1)[1][1], :] + r1[knn(tree1, E[osc_vars, i], 1)[1][1], :]
                E[:, i] += (H')*(H_osc')*inc
            end
        end

        x_true = integrator(model, x_true, 0.0, window, Δt)
        x_free = integrator(model_err, x_free, 0.0, window, Δt)
    end
    return errs, errs_free, B
end

end
