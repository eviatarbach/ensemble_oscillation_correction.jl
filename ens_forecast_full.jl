module ens_forecast_full

include("analog.jl")
include("da.jl")

using .Analog
using .DA

export forecast

using Statistics
using LinearAlgebra
using Random

using NearestNeighbors
using Distributions

struct Forecast_Info
    errs
    errs_uncorr
    spread
    ens_errs
    ens
    x_trues
    errs_m
end

function is_valid(p, model, integrator, t, Δt, test_time, bounds)
    end_point = integrator(model, p, t, t + test_time, Δt)
    return all(bounds[1] .<= end_point .<= bounds[2])
end

function forecast(; E::Array{float_type, 2}, model, model_err, integrator,
                  m::Integer, Δt::float_type, window::Integer, cycles::Integer,
                  outfreq::Integer, D::Integer, k, tree, means, stds,
                  err_pct::float_type, check_bounds, test_time, bounds, R,
                  inflation, y) where {float_type<:AbstractFloat}
    R_inv = inv(R)

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
    y_forecast = nothing

    for cycle=1:cycles
        println(cycle)

        x_m = mean(E, dims=2)
        append!(errs_uncorr, sqrt(mean((x_m .- x_true).^2)))

        if (y_forecast != nothing)
            append!(ens_errs, sqrt.(mean((E .- x_true).^2, dims=1)'))

            E = etkf(E=E, R_inv=R_inv, inflation=inflation,
                     H=x->x', y=y_forecast)
            x_m = mean(E, dims=2)
            append!(errs, sqrt(mean((x_m .- x_true).^2)))

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
        y_forecast = find_point(y, tree, x_m, k, window)

        for i=1:m
            E[:, i] = integrator(model_err, E[:, i], t, t + window*outfreq*Δt, Δt)
        end

        x_true = integrator(model, x_true, t, t + window*outfreq*Δt, Δt)

        #y_forecast = x_true'

        t += window*outfreq*Δt
    end

    ens_errs = reshape(ens_errs, m, :)'
    ens = reshape(ens, D, m, :)
    x_trues = reshape(x_trues, D, :)
    return Forecast_Info(errs, errs_uncorr, spread, ens_errs, ens,
                         x_trues, errs_m)
end

end
