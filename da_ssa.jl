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

function find_point(r, tree, p, k, f)
    ind, dist = knn(tree, p[:], k)
    mask = (ind .+ f) .<= size(tree.data)[1]
    dist = dist[mask]
    ind = ind[mask]
    return sum(dist .* r[ind .+ f, :], dims=1)/sum(dist)
end

function ETKF_SSA(; E::Array{Float64, 2}, model, model_err, integrator,
                  R::Symmetric{Float64, Array{Float64, 2}}, m::Int64,
                  Δt::Float64, window::Int64, cycles::Int64, outfreq::Int64,
                  D::Int64, M::Int64, k, r, r_err, tree_err, tree_r_err, tree, tree_r, psrm=true, H=I,
                  inflation=1.0, osc_vars=1:D, cov=false, modes, da=true, stds)
    da = false
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
    r_spreads = []

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
    hist = []
    r_forecast = nothing
    weights = nothing
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

        X = inflation*X
        if cov
            B = B*(cycle - 1) + X*X'
            B = B/cycle
        end

        #println(w, sum(w))
        if da
            Y = (H*E .- H*x_m)/sqrt(m - 1)
            Ω = real((I + Y'*R_inv*Y)^(-1))
            w = Ω*Y'*R_inv*(y - H*x_m)

            #if psrm & (r_forecast != nothing)
            #    r_ens = vcat([find_point(r_err, tree_err, E[:, i], k, 0) for i=1:m]...)
            #    r_errs = sqrt.(mean((r_ens .- r_forecast).^2, dims=2))
            #    weights = 1 ./ ((r_errs)')
            #    weights = (weights .+ mean(weights))/2
            #    w_std = std(w)
            #    w = w .* weights
            #    w = w/std(w)*w_std
                #w = w .- mean(w)
                #println(mean(w))
                #w = w .- mean(w)
            #end

            E = real(x_m .+ X*(w .+ sqrt(m - 1)*Ω^(1/2)))
        else
            #ens_spread = mean(std(E, dims=2))
            #append!(spread, ens_spread)
            #E = x_true .+ rand(MvNormal(zeros(D), diagm(0=>(0.01*stds)[:].^2)), m)#(E .- x_true) ./ (mean(std(E, dims=2))) # 10*randn(size(E)...)
        end
        if psrm & (r_forecast != nothing)# & (ens_spread > mean(spread))
            r_ens = vcat([find_point(r_err, tree_err, E[:, i], k, 0) for i=1:m]...)
            r_errs = sqrt.(mean((r_ens .- r_forecast).^2, dims=2))
            r_spread = std(r_errs)
            append!(r_spreads, r_spread)

            slope = 0.9288489414364894
            intercept = 0.6331840617954532

            err_estimates = r_errs
            weights = (1 ./ (err_estimates.^2))/sum(1 ./ (err_estimates.^2))
            #println(maximum(weights))
            #weights = (weights .+ mean(weights))/2
            #sum_w = sum(w)
            #w = w .* (0.01*(1 ./ r_err))

            # Normalize
            #w = w .- sum(w)
            #x_m = sum(E .* weights, dims=2)/sum(weights)
            append!(hist, (copy(E), copy(r_errs), copy(x_true)))
            x_m = sum(E .* weights', dims=2)/sum(weights)
            append!(errs, sqrt(mean((x_m .- x_true).^2)))
        else
            append!(errs, sqrt(mean((mean(E, dims=2) .- x_true).^2)))
        end
        append!(errs_free, sqrt(mean((x_free .- x_true).^2)))

        if !da
            ens_spread = mean(std(E, dims=2))
            append!(spread, ens_spread)
            E = x_true .+ rand(MvNormal(zeros(D), diagm(0=>(0.1*stds)[:].^2)), m)#(E .- x_true) ./ (mean(std(E, dims=2)))#rand(MvNormal(zeros(D), diagm(0=>(0.1*stds)[:].^2)), m)
        end

        if psrm
            x_m = mean(E, dims=2)
            p2 = find_point(r, tree, x_m, k, 0)
            r_forecast = find_point(r, tree_r, p2, k, window)
        end

        for i=1:m
            E[:, i] = integrator(model_err, E[:, i], t, t + window*outfreq*Δt, Δt)
        end

        x_true = integrator(model, x_true, t, t + window*outfreq*Δt, Δt)
        x_free = integrator(model_err, x_free, t, t + window*outfreq*Δt, Δt)

        t += window*outfreq*Δt
    end
    return DA_Info(errs, errs_free, spread, hist, alphas)
end

end
