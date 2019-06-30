module DA_SSA

export ETKF_SSA

include("integrators.jl")
include("embedding.jl")
using .Integrators
using .Embedding

using Distributions
using LinearAlgebra

function ETKF_SSA(E::Array{Float64, 2}, model::Function, model_err::Function,
                  R::Symmetric{Float64, Array{Float64, 2}}, m::Int64, tree, osc,
                  pcs, k, D, M; H=I, Δt::Float64=0.1,
                  window::Float64=0.4, cycles::Int64=1000, outfreq=40)
    if H != I
        p, n = size(H)
    else
        p, n = size(R)
    end

    full_x_hist = []
    x_hist = reshape(reshape(E', m, 2*(M - 1) + 1, D)[:, 1:(M-1), :], m, D*(M-1))'

    R_inv = inv(R)
    obs_err = MvNormal(zeros(p), R)

    x_true = reshape(E', m, 2*(M - 1) + 1, D)[end, M, :]
    x_free = x_true + randn(D)/10

    x_m = mean(E, dims=2)

    errs = []

    errs_free = []

    B = zeros(n, n)

    for cycle=1:cycles
        println(cycle)
        y = zeros(p)
        y[1:D] = x_true + rand(obs_err)[1:D]
        y[D+1:end] = project(tree, (reshape(x_m, 2*(M - 1) + 1, D)[M-1, :]'*pcs)', osc, k, 1)[2, :]
        x_m = mean(E, dims=2)

        X = (E .- x_m)/sqrt(m - 1)
        #B = B + X*X'
        Y = (H*E .- H*x_m)/sqrt(m - 1)
        Ω = real((I + Y'*R_inv*Y)^(-1))
        w = Ω*Y'*R_inv*(y - H*x_m)

        E_new = real(x_m .+ X*(w .+ sqrt(m - 1)*Ω^(1/2)))
        E = reshape(E', m, 2*(M-1) + 1, D)
        E[:, M, :] = reshape(E_new', m, 2*(M-1) + 1, D)[:, M, :]  # apply analysis increment only to present
        E = reshape(E, m, (2*(M-1) + 1)*D)'
        x_hist = reshape(cat(reshape(x_hist', m, M - 1, D)[:, 2:end, :], reshape(E', m, 2*(M-1)+1, D)[:, M:M, :], dims=2), m, D*(M-1))'
        append!(full_x_hist, reshape(mean(E, dims=2), 2*(M - 1) + 1, D)[M, :]')
        err = sqrt(mean((reshape(mean(E, dims=2), 2*(M - 1) + 1, D)[M, :] .- x_true).^2))
        append!(errs, err)
        append!(errs_free, sqrt(mean((x_free .- x_true).^2)))

        for i=1:m
            E_i = E[:, i]
            E_i = reshape(E_i, 2*(M - 1) + 1, D)
            #E_i[M, :] = rk4_inplace(model_err, E_i[M, :], 0.0, window, Δt)
            E_i[M:end, :] = rk4(model_err, E_i[M, :], 0.0, window*M, Δt, outfreq)
            E_i[1:M-1, :] = reshape(x_hist'[i, :], M - 1, D)
            E_i = reshape(E_i, D*(2*(M - 1) + 1))
            E[:, i] = E_i
        end

        x_true = rk4_inplace(model, x_true, 0.0, window, Δt)
        x_free = rk4_inplace(model_err, x_free, 0.0, window, Δt)
    end
    return errs, errs_free, full_x_hist, B
end

end
