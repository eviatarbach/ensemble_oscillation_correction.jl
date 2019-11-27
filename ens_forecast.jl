module ens_forecast

include("embedding.jl")

using .Embedding

using Statistics
using LinearAlgebra

using NearestNeighbors
using Distributions

struct Forecast_Info
    errs
    errs_uncorr
    spread
    r_errs
    ens_errs
    ens
    x_trues
    errs_m
end

function find_point(r, tree, p, k, f)
    ind, dist = knn(tree, p[:], k)
    mask = (ind .+ f) .<= size(tree.data)[1]
    dist = dist[mask]
    ind = ind[mask]
    return sum(dist .* r[ind .+ f, :], dims=1)/sum(dist)
end

function find_point2(model, p, C_conds, outfreq, M, Δt, modes)
    future = vcat(p', integrator(model, p, 0.0, outfreq*Δt*(M-1), Δt, inplace=false))[1:outfreq:end, :]
    pred = sum(Embedding.reconstruct_cp(Embedding.transform_cp(future, M, 'b', C_conds), EV, M, D, modes), dims=1)[1, :]
    return pred
end

function forecast(; E::Array{Float64, 2}, model, model_err, integrator,
                      m::Int64, Δt::Float64, window::Int64, cycles::Int64,
                      outfreq::Int64, D::Int64, k, k_r, r, tree, tree_r,
                      osc_vars=1:D, stds, means, err_pct, mp)
    x_true = E[:, end]
    x0 = copy(x_true)

    errs = []
    errs_uncorr = []
    spread = []
    r_errs_hist = []
    ens_errs = []
    ens = []
    x_trues = []
    errs_m = []

    t = 0.0
    r_forecast = nothing

    for cycle=1:cycles
        println(cycle)
        x_m = mean(E, dims=2)

        if (r_forecast != nothing)
            r_ens = vcat([find_point(r, tree, E[:, i], k, 0) for i=1:m]...)
            r_errs = sqrt.(mean((r_ens .- r_forecast).^2, dims=2))
            append!(r_errs_hist, r_errs)
            append!(ens_errs, sqrt.(mean((E .- x_true).^2, dims=1)'))

            err_estimates = r_errs
            weights = (1 ./ (err_estimates.^2))/sum(1 ./ (err_estimates.^2))

            x_m = mean(E[:, sortperm(r_errs[:])[1:mp]], dims=2)#sum(E .* weights', dims=2)/sum(weights)
            append!(errs, sqrt(mean((x_m .- x_true).^2)))
            append!(errs_uncorr, sqrt(mean((mean(E, dims=2) .- x_true).^2)))
            append!(ens, E)
            append!(x_trues, x_true)
            append!(errs_m, sqrt(mean((means .- x_true).^2)))
        else
            append!(errs_uncorr, sqrt(mean((mean(E, dims=2) .- x_true).^2)))
        end

        ens_spread = mean(std(E, dims=2))
        append!(spread, ens_spread)
        E = x_true .+ rand(MvNormal(zeros(D), diagm(0=>(err_pct*stds)[:].^2)), m)

        x_m = mean(E, dims=2)
        p2 = find_point(r, tree, x_m, k, 0)
        r_forecast = find_point(r, tree_r, p2, k_r, window)

        for i=1:m
            E[:, i] = integrator(model_err, E[:, i], t, t + window*outfreq*Δt, Δt)
        end

        x_true = integrator(model, x_true, t, t + window*outfreq*Δt, Δt)

        t += window*outfreq*Δt
    end

    ens_errs = reshape(ens_errs, m, :)'
    r_errs_hist = reshape(r_errs_hist, m, :)'
    ens = reshape(ens, D, m, :)
    x_trues = reshape(x_trues, D, :)
    return Forecast_Info(errs, errs_uncorr, spread, r_errs_hist, ens_errs, ens,
                         x_trues, errs_m)
end

end
