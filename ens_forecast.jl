module ens_forecast

include("ssa.jl")
include("analog.jl")
include("da.jl")

using .SSA
using .Analog
using .DA

export forecast

using Statistics
using LinearAlgebra
using Random

using NearestNeighbors
using Distributions
using PyCall

struct Forecast_Info
    errs
    errs_uncorr
    crps
    crps_uncorr
    spread
    r_errs
    ens_errs
    ens
    x_trues
    errs_m
    r_forecasts
    stds
    errs_y_fcst
end

xskillscore = pyimport("xskillscore")
xarray = pyimport("xarray")

function is_valid(p, model, integrator, t, Δt, test_time, bounds)
    end_point = integrator(model, p, t, t + test_time, Δt)
    return all(bounds[1] .<= end_point .<= bounds[2])
end

function forecast(; E::Array{float_type, 2}, model, model_err, integrator,
                  m::Integer, Δt::float_type, window::Integer, cycles::Integer,
                  outfreq::Integer, D::Integer, k, k_r, r, tree, tree_r,
                  osc_vars=1:D, means, stds, err_pct::float_type, mp, check_bounds,
                  test_time, bounds, da, R, inflation, var_model, α) where {float_type<:AbstractFloat}
    if da
        R_inv = inv(R)
    end
    x_true = E[:, end]
    x0 = copy(x_true)

    errs = []
    errs_uncorr = []
    crps = []
    crps_uncorr = []
    spread = []
    r_errs_hist = []
    ens_errs = []
    ens = []
    x_trues = []
    errs_m = []
    r_forecasts = []
    errs_y_fcst = []

    t = 0.0
    r_forecast = nothing
    hybrid_fcsts = zeros(size(E))*NaN

    for cycle=1:cycles
        println(cycle)

        x_m = mean(E, dims=2)
        append!(errs_uncorr, sqrt(mean((x_m .- x_true).^2)))

        if (r_forecast !== nothing)
            r_ens = vcat([find_point(r, tree, E[:, i], k, 0) for i=1:m]...)
            r_errs = sqrt.(mean((r_ens .- r_forecast).^2, dims=2))
            append!(r_errs_hist, r_errs)
            append!(ens_errs, sqrt.(mean((E .- x_true).^2, dims=1)'))
            append!(r_forecasts, r_forecast)

            if !da
                E_mp = E[:, (sortperm(r_errs[:]))[1:mp]]
                E_mp_array = xarray.DataArray(data=E_mp, dims=["dim", "member"])
                E_array = xarray.DataArray(data=E, dims=["dim", "member"])
                x_m = mean(E_mp, dims=2)
                append!(errs, sqrt(mean((x_m .- x_true).^2)))
                if var_model !== nothing
                    append!(errs_y_fcst, sqrt(mean((mean(hybrid_fcsts, dims=2) .- x_true).^2)))
                end
                append!(crps, xskillscore.crps_ensemble(x_true, E_mp_array).values[1])
                append!(crps_uncorr, xskillscore.crps_ensemble(x_true, E_array).values[1])
                append!(ens, E)
            else
                E_array = xarray.DataArray(data=E, dims=["dim", "member"])
                E = etkf(E=E, R_inv=R_inv, inflation=inflation,
                         H=x->find_point(r, tree, x, k, 0), y=r_forecast)
                E_corr_array = xarray.DataArray(data=E, dims=["dim", "member"])
                x_m = mean(E, dims=2)
                append!(errs, sqrt(mean((x_m .- x_true).^2)))
                append!(crps, xskillscore.crps_ensemble(x_true, E_corr_array).values[1])
                append!(crps_uncorr, xskillscore.crps_ensemble(x_true, E_array).values[1])
                append!(ens, E)
            end
            append!(x_trues, x_true)
            append!(errs_m, sqrt(mean((means .- x_true).^2)))
        end

        ens_spread = mean(std(E, dims=2))
        append!(spread, ens_spread)

        if !check_bounds
            perts = rand(MvNormal(zeros(float_type, D), diagm(0=>(err_pct*stds)[:].^2)), m)
        else
            perts = Array{float_type}(undef, D, m)
            i = 0
            while i < m
                pert = rand(MvNormal(zeros(float_type, D), diagm(0=>(err_pct*stds)[:].^2)),
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
            integration = integrator(model_err, E[:, i], t,
                                     t + window*outfreq*Δt, Δt, inplace=false)
            E[:, i] = integration[end, :]
            if var_model !== nothing
                if window >= var_model.k_ar
                    y_fcst = var_model.forecast(integration[end-var_model.k_ar+1:end, :],
                                                window)[end, :]
                    hybrid_fcsts[:, i] = α*E[:, i] + (1 - α)*y_fcst
                end
            end
        end

        x_true = integrator(model, x_true, t, t + window*outfreq*Δt, Δt)

        t += window*outfreq*Δt
    end

    ens_errs = reshape(ens_errs, m, :)'
    r_errs_hist = reshape(r_errs_hist, m, :)'
    ens = reshape(ens, D, m, :)
    x_trues = reshape(x_trues, D, :)
    r_forecasts = reshape(r_forecasts, length(osc_vars), :)
    return Forecast_Info(errs, errs_uncorr, crps, crps_uncorr, spread,
                         r_errs_hist, ens_errs, ens, x_trues, errs_m,
                         r_forecasts, stds, errs_y_fcst)
end

end
