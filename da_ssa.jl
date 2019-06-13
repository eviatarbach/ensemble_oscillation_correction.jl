module DA_SSA

export ETKF_SSA

include("integrators.jl")
include("embedding.jl")
using .Integrators
using .Embedding

using Distributions
using LinearAlgebra

function ETKF_SSA(E::Array{Float64, 2}, model::Function,
                  R::Symmetric{Float64, Array{Float64, 2}}, m::Int64, tree, osc,
                  data, k, D, M, x_hist; H=I, Δt::Float64=0.1,
                  window::Float64=0.4, cycles::Int64=1000, outfreq=40)
    if H != I
        p, n = size(H)
    else
        p, n = size(R)
    end

    R_inv = inv(R)
    obs_err = MvNormal(zeros(p), R)

    @assert size(x_hist) == (M - 1, D)

    x_true = E[:, end]
    x_free = x_true + randn(D)/10

    errs = []

    errs_free = []

    B = zeros(n, n)

    E2 = zeros(n, m)
    for i=1:m
        E_i = E2[:, i]
        E_i = reshape(E_i, 2*(M - 1) + 1, D)
        E_i[M, :] = E[:, i]
        E_i[1:M-1, :] = x_hist
        E_i[M+1:end, :] = rk4(model, E[:, i], 0.0, window*(M - 1), Δt, 4)
        E_i = reshape(E_i, D*(2*(M - 1) + 1))
        E2[:, i] = E_i
    end
    E = E2

    for cycle=1:cycles
        y = zeros(p)
        y[1:D] = x_true + rand(obs_err)[1:D]
        x_m = mean(E, dims=2)
        y[D+1:end] = project(tree, x_m[:, 1], data, osc, k, 1)[2, :]

        X = (E .- x_m)/sqrt(m - 1)
        #B = B + X*X'
        Y = (H*E .- H*x_m)/sqrt(m - 1)
        Ω = real((I + Y'*R_inv*Y)^(-1))
        w = Ω*Y'*R_inv*(y - H*x_m)

        E = real(x_m .+ X*(w .+ sqrt(m - 1)*Ω^(1/2)))
        x_hist = vcat(x_hist[2:end, :], reshape(mean(E, dims=2), 2*(M - 1) + 1, D)[M, :]')
        err = sqrt(mean((reshape(mean(E, dims=2), 2*(M - 1) + 1, D)[M, :] .- x_true).^2))
        append!(errs, err)
        append!(errs_free, sqrt(mean((x_free .- x_true).^2)))

        for i=1:m
            E_i = E[:, i]
            E_i = reshape(E_i, 2*(M - 1) + 1, D)
            E_i[M, :] = rk4_inplace(model, E_i[M, :], 0.0, window, Δt)
            E_i[1:M-1, :] = x_hist
            E_i[M+1:end, :] = rk4(model, E_i[M, :], 0.0, window*(M - 1), Δt, 4)
            E_i = reshape(E_i, D*(2*(M - 1) + 1))
            E[:, i] = E_i
        end

        x_true = rk4_inplace(model, x_true, 0.0, window, Δt)
        x_free = rk4_inplace(model, x_free, 0.0, window, Δt)
    end
    return errs, errs_free
end

end
