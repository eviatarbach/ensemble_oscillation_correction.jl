module DA_SSA

export ETKF_SSA

include("integrators.jl")
include("embedding.jl")
using .Integrators
using .Embedding

using Distributions
using LinearAlgebra

function ETKF_SSA(E::Array{Float64, 2}, model::Function,
                  R::Symmetric{Float64, Array{Float64, 2}}, m::Int64, tree, osc;
                  H=I, Δt::Float64=0.1, window::Float64=1.0, cycles::Int64=100,
                  outfreq=1)
    if H != I
        p, n = size(H)
    else
        p, n = size(R)
    end

    R_inv = inv(R)
    obs_err = MvNormal(zeros(p), R)

    x_true = E[:, end]
    x_free = x_true + randn(n)/10

    errs = []

    errs_free = []

    B = zeros(n, n)

    for cycle=1:cycles
        y = H*x_true + rand(obs_err)
        y[p] = project(tree, mean(E, dims=2), osc, 20, 1)[2]
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

        E = hcat(map((col)->rk4_inplace(model, col, 0.0, window, Δt), E[:, i] for i=1:m)...)

        x_true = rk4_inplace(model, x_true, 0.0, window, Δt)
        x_free = rk4_inplace(model, x_free, 0.0, window, Δt)
    end
    return errs, errs_free
end

end
