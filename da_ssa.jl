module DA_SSA

export ETKF_SSA

include("embedding.jl")

using .Embedding

using Statistics
using LinearAlgebra
using Distributed

using NearestNeighbors
using Distributions

struct DA_Info
    errs
    errs_free
    spread
    B
    alphas
end

function ETKF_SSA(; E::Array{Float64, 2}, model, model_err, integrator,
                  R::Symmetric{Float64, Array{Float64, 2}}, m::Int64,
                  Δt::Float64, window::Float64, cycles::Int64, outfreq::Int64,
                  D::Int64, M::Int64, k, r1, r2, tree1, tree2, psrm=true, H=I,
                  inflation=1.0, osc_vars=1:D, cov=false, EV, EV2, C_conds, C_conds2, modes)
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

    t = 0.0
    alpha_b = inflation
    v_b = 0.01^2
    alpha = alpha_b
    alphas = []
    for cycle=1:cycles
        println(cycle)
        y = H*x_true + rand(obs_err)
        x_m = mean(E, dims=2)

        X = (E .- x_m)/sqrt(m - 1)
        d = y - H*x_m
        B = X*X'

        alpha_b = alpha
        alpha_o = (tr((d*d') .* R_inv) - p)/tr((H*B*H') .* R_inv)
        v_o = (2/p)*((alpha_b*tr((H*B*H') .* R_inv) + p)/tr((H*B*H') .* R_inv))^2
        alpha = (alpha_b*v_o + alpha_o*v_b)/(v_o + v_b)
        append!(alphas, alpha)

        X = alpha*X
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
            #E[:, i] = integrator(model_err, E[:, i], t, t + window, Δt)
            if psrm
                # To improve performance, should do tree searches for all
                # ensemble members at once
                #future = vcat(E[:, i]', integrator(model_err, E[:, i], t + window, t + window*(M+1), Δt, inplace=false))[1:outfreq:end, :]
                #pred = sum(Embedding.reconstruct_cp(Embedding.transform_cp(future, M, 'b', C_conds), EV, M, D, modes), dims=1)[1, :]
                #println(i)
                #println(E[:, i])
                #println(pred)
                #future = vcat(E[:, i]', integrator(model, E[:, i], t + window, t + window*(M+1), Δt, inplace=false))[1:outfreq:end, :]
                #pred2 = sum(Embedding.reconstruct_cp(Embedding.transform_cp(future, M, 'b', C_conds2), EV2, M, D, modes), dims=1)[1, :]
                #println(pred2)
                inc = -mean(r2[filter((e)->e<size(r2)[1], knn(tree2, E[osc_vars, i], k)[1] .+ 1), :], dims=1) + mean(r1[filter((e)->e<size(r1)[1], knn(tree1, E[osc_vars, i], k)[1] .+ 1), :], dims=1)
                #inc = -pred' + mean(r1[knn(tree1, E[osc_vars, i], k)[1], :], dims=1)
                #inc = (-pred' + pred2').^3
                #E[:, i] += (H')*(H_osc')*inc'
            end
            E[:, i] = integrator(model_err, E[:, i], t, t + window, Δt)
            #println(E[:, i])
            if psrm
                if (~any(isnan.(inc)))
                    E[:, i] += (H')*(H_osc')*inc'
                end
            end
        end

        x_true = integrator(model, x_true, t, t + window, Δt)
        x_free = integrator(model_err, x_free, t, t + window, Δt)

        t += window
    end
    return DA_Info(errs, errs_free, spread, B, alphas)
end

end
