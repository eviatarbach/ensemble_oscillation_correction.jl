module ens_forecast

include("ssa.jl")
include("analog.jl")
include("da.jl")

using .SSA
using .Analog
using .DA

using Statistics
using LinearAlgebra
using Random

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
    r_forecasts
end

function is_valid(p, model, integrator, t, Δt, test_time, bounds)
    end_point = integrator(model, p, t, t + test_time, Δt)
    return all(bounds[1] .<= end_point .<= bounds[2])
end

function forecast(; E::Array{Float64, 2}, model, model_err, integrator,
                      m::Int64, Δt::Float64, window::Int64, cycles::Int64,
                      outfreq::Int64, D::Int64, k, k_r, r, tree, tree_r,
                      osc_vars=1:D, stds, means, err_pct, mp, check_bounds,
                      test_time, bounds, da, R, inflation)
    if da
        R_inv = inv(R)
    end
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
    r_forecasts = []

    t = 0.0
    r_forecast = nothing

    for cycle=1:cycles
        println(cycle)
        x_m = mean(E, dims=2)

        append!(errs_uncorr, sqrt(mean((mean(E, dims=2) .- x_true).^2)))
        if (r_forecast != nothing)
            r_ens = vcat([find_point(r, tree, E[:, i], k, 0) for i=1:m]...)
            r_errs = sqrt.(mean((r_ens .- r_forecast).^2, dims=2))
            append!(r_errs_hist, r_errs)
            append!(ens_errs, sqrt.(mean((E .- x_true).^2, dims=1)'))
            append!(r_forecasts, r_forecast)

            err_estimates = r_errs

            if !da
                x_m = mean(E[:, (sortperm(r_errs[:]))[1:mp]], dims=2)
                append!(errs, sqrt(mean((x_m .- x_true).^2)))
                append!(ens, E)
                append!(x_trues, x_true)
                append!(errs_m, sqrt(mean((means .- x_true).^2)))
            else
                E = etkf(E=E, R_inv=R_inv, inflation=inflation,
                         H=x->find_point(r, tree, x, k, 0), y=r_forecast)
                x_m = mean(E, dims=2)
                append!(errs, sqrt(mean((x_m .- x_true).^2)))
            end
        end

        ens_spread = mean(std(E, dims=2))
        append!(spread, ens_spread)

        if !check_bounds
            perts = rand(MvNormal(zeros(D), diagm(0=>(err_pct*stds)[:].^2)), m)
        else
            perts = Array{Float64}(undef, D, m)
            i = 0
            while i < m
                pert = rand(MvNormal(zeros(D), diagm(0=>(err_pct*stds)[:].^2)),
                            1)
                if is_valid(x_true + pert[:], model_err, integrator, t, Δt, test_time,
                            bounds)
                    i += 1
                    perts[:, i] = pert
                end
            end
        end
        E = x_true .+ perts

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
    r_forecasts = reshape(r_forecasts, length(osc_vars), :)
    return Forecast_Info(errs, errs_uncorr, spread, r_errs_hist, ens_errs, ens,
                         x_trues, errs_m, r_forecasts)
end

end
