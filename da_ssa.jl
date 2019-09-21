module DA_SSA

export ETKF_SSA

using Statistics
using LinearAlgebra
using Distributed
using SharedArrays

using NearestNeighbors
using Distributions

struct DA_Info
    errs
    errs_free
    spread
    B
end

function ETKF_SSA(; E::Array{Float64, 2}, model, model_err, integrator,
                  R::Symmetric{Float64, Array{Float64, 2}}, m::Int64,
                  Δt::Float64, window::Float64, cycles::Int64, outfreq::Int64,
                  D::Int64, M::Int64, r1, r2, tree1, tree2, psrm=true, H=I,
                  inflation=1.0, osc_vars=1:D, cov=false)
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
    spread = []

    if cov
        B = zeros(D, D)
    else
        B = nothing
    end

    for cycle=1:cycles
        println(cycle)
        y = H*x_true + rand(obs_err)
        x_m = mean(E, dims=2)

        X = (E .- x_m)/sqrt(m - 1)
        X = x_m .+ inflation*(X .- x_m)
        if cov
            B = B*(cycle - 1) + X*X'
            B = B/cycle
        end
        Y = (H*E .- H*x_m)/sqrt(m - 1)
        Ω = real((I + Y'*R_inv*Y)^(-1))
        w = Ω*Y'*R_inv*(y - H*x_m)

        E = real(x_m .+ X*(w .+ sqrt(m - 1)*Ω^(1/2)))
        append!(errs, sqrt(mean((mean(E, dims=2) .- x_true).^2)))
        append!(errs_free, sqrt(mean((x_free .- x_true).^2)))
        append!(spread, mean(std(E, dims=2)))

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
    return DA_Info(errs, errs_free, spread, B)
end

end
