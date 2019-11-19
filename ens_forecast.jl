module ens_forecast

using Statistics
using LinearAlgebra

using NearestNeighbors
using Distributions

struct Forecast_Info
    errs
    spread
    r_errs
    ens_errs
end

function find_point(r, tree, p, k, f)
    ind, dist = knn(tree, p[:], k)
    mask = (ind .+ f) .<= size(tree.data)[1]
    dist = dist[mask]
    ind = ind[mask]
    return sum(dist .* r[ind .+ f, :], dims=1)/sum(dist)
end

function forecast(; E::Array{Float64, 2}, model, model_err, integrator,
                      m::Int64, Δt::Float64, window::Int64, cycles::Int64,
                      outfreq::Int64, D::Int64, k, k_r, r, tree, tree_r,
                      osc_vars=1:D, stds, err_pct=0.1, correction)
    x_true = E[:, end]

    errs = []
    spread = []
    r_errs_hist = []
    ens_errs = []

    t = 0.0
    r_forecast = nothing

    for cycle=1:cycles
        println(cycle)
        x_m = mean(E, dims=2)

        if correction & (r_forecast != nothing)
            r_ens = vcat([find_point(r, tree, E[:, i], k, 0) for i=1:m]...)
            r_errs = sqrt.(mean((r_ens .- r_forecast).^2, dims=2))
            append!(r_errs_hist, r_errs)
            append!(ens_errs, sqrt.(mean((E .- x_true).^2, dims=1)'))

            err_estimates = r_errs
            weights = (1 ./ (err_estimates.^2))/sum(1 ./ (err_estimates.^2))

            x_m = sum(E .* weights', dims=2)/sum(weights)
            append!(errs, sqrt(mean((x_m .- x_true).^2)))
        else
            append!(errs, sqrt(mean((mean(E, dims=2) .- x_true).^2)))
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
    return Forecast_Info(errs, spread, r_errs_hist, ens_errs)
end

end
