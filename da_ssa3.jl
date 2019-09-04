module DA_SSA3

export ETKF_SSA

include("integrators.jl")
include("embedding.jl")
using .Integrators
using .Embedding

using Distributions
using LinearAlgebra
using NearestNeighbors

function ETKF_SSA(E::Array{Float64, 2}, model::Function, model_err::Function,
                  R::Symmetric{Float64, Array{Float64, 2}}, m::Int64, D, M,
                  r1, r2, tree1, tree2; psrm=true, H=I, Δt::Float64=0.1,
                  window::Float64=0.4, cycles::Int64=1000, outfreq=40,
                  inflation=1.0)
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
        X = x_m .+ inflation*(X .- x_m)
        #B = B + X*X'
        Y = (H*E .- H*x_m)/sqrt(m - 1)
        Ω = real((I + Y'*R_inv*Y)^(-1))
        w = Ω*Y'*R_inv*(y - H*x_m)

        E = real(x_m .+ X*(w .+ sqrt(m - 1)*Ω^(1/2)))
        err = sqrt(mean((mean(E, dims=2) .- x_true).^2))
        append!(errs, err)
        append!(errs_free, sqrt(mean((x_free .- x_true).^2)))

        for i=1:m
            E[:, i] = rk4_inplace(model_err, E[:, i], 0.0, window, Δt)
            if psrm
                E[:, i] = E[:, i] - r2[knn(tree2, E[:, i], 1)[1][1], :] + r1[knn(tree1, E[:, i], 1)[1][1], :]
            end
        end

        x_true = rk4_inplace(model, x_true, 0.0, window, Δt)
        x_free = rk4_inplace(model_err, x_free, 0.0, window, Δt)
    end
    return errs, errs_free, full_x_hist
end

end
