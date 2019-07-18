module DA_SSA3

export ETKF_SSA

include("integrators.jl")
include("embedding.jl")
using .Integrators
using .Embedding

using Distributions
using LinearAlgebra

function ETKF_SSA(E::Array{Float64, 2}, model::Function, model_err::Function,
                  R::Symmetric{Float64, Array{Float64, 2}}, m::Int64, tree, osc,
                  pcs, k, D, M, oracle, obs_operator_err, x_hist; H=I, Δt::Float64=0.1,
                  window::Float64=0.4, cycles::Int64=1000, outfreq=40)
    if H != I
        p, n = size(H)
    else
        p, n = size(R)
    end

    full_x_hist = []
    x_true_hist = []
    x_free_hist = []

    R_inv = inv(R)
    obs_err = MvNormal(zeros(p), R)

    x_true = E[:, end]
    x_free = x_true + randn(D)/5

    errs = []

    errs_free = []

    B = zeros(n, n)

    for cycle=1:cycles
        println(cycle)
        y = H*x_true + rand(obs_err)
        x_m = mean(E, dims=2)

        X = (E .- x_m)/sqrt(m - 1)
        #B = B + X*X'
        Y = (H*E .- H*x_m)/sqrt(m - 1)
        Ω = real((I + Y'*R_inv*Y)^(-1))
        w = Ω*Y'*R_inv*(y - H*x_m)

        E = real(x_m .+ X*(w .+ sqrt(m - 1)*Ω^(1/2)))
        err = sqrt(mean((mean(E, dims=2) .- x_true).^2))
        append!(errs, err)
        append!(errs_free, sqrt(mean((x_free .- x_true).^2)))

        x_hist = reshape(cat(reshape(x_hist', m, M - 1, D)[:, 2:end, :], reshape(E', m, 1, D), dims=2), m, D*(M-1))'

        for i=1:m
            curr_osc = zeros(2*(M - 1) + 1, D)
            curr_osc[M+1:end, :] = rk4(model_err, E[:, i], 0.0, window*(M-1), Δt, outfreq)
            curr_osc[M, :] = E[:, i]
            curr_osc[1:M-1, :] = reshape(x_hist'[i, :], M - 1, D)
            curr_osc = reshape(curr_osc, D*(2*(M - 1) + 1))
            E[:, i] = rk4(model_err, E[:, i], 0.0, window, Δt, outfreq)
#            E[:, i] -= obs_operator_err'*curr_osc
#            E[:, i] += oracle[cycle, :]
        end

        x_true = rk4_inplace(model, x_true, 0.0, window, Δt)
        x_free = rk4_inplace(model_err, x_free, 0.0, window, Δt)
    end
    return errs, errs_free, full_x_hist
end

end
