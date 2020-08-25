module full

include("analog.jl")
using .Analog

using LinearAlgebra
using Statistics
using Random

using Distributions
using NearestNeighbors

include("ens_forecast_full.jl")

using .ens_forecast_full

function setup(; model, Δt, outfreq, obs_err_pct, record_length, transient,
               y0, D, integrator, window, k)
   y = integrator(model, y0, 0., record_length*outfreq*Δt, Δt; inplace=false)[1:outfreq:end, :][(transient + 1):end, :]
   if (obs_err_pct > 0)
      R = Symmetric(diagm(0 => obs_err_pct*std(y, dims=1)[1, :]))
      obs_err = MvNormal(zeros(D), R)
      y = y + rand(obs_err, size(y)[1])'
   end

   R = Analog.error_cov_full(y, window, k)

   tree = KDTree(copy(y'))

   return tree, y, R
end

function run(; model, model_err, integrator, m, D, k, outfreq, Δt, cycles,
             window, record_length, obs_err_pct, ens_err_pct, transient, y0,
             check_bounds, test_time=false, inflation)

    tree, y, R = setup(model=model, record_length=record_length,
                       integrator=integrator, Δt=Δt, y0=y0, transient=transient,
                       outfreq=outfreq, obs_err_pct=obs_err_pct, D=D,
                       window=window, k=k)

    E = integrator(model_err, y[end, :], 0.0, m*Δt*outfreq, Δt, inplace=false)[1:outfreq:end, :]'

    stds = std(y, dims=1)[:]
    means = mean(y, dims=1)
    bounds = (minimum(y, dims=1)[:], maximum(y, dims=1)[:])

    ens_info = forecast(E=copy(E), model=model, model_err=model_err,
                        integrator=integrator, m=m, Δt=Δt, window=window,
                        cycles=cycles, outfreq=outfreq, D=D, k=k, tree=tree,
                        means=means, stds=stds, err_pct=ens_err_pct,
                        check_bounds=check_bounds, test_time=test_time,
                        bounds=bounds, R=R, inflation=inflation, y=y)

    return ens_info
end
end
