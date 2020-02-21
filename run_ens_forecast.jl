module run_ens_forecast

using LinearAlgebra
using Statistics
using Random

using Distributions
using NearestNeighbors

include("./embedding.jl")
using .Embedding

include("ens_forecast.jl")
using .ens_forecast

struct SSA_Info
    EW
    EV
    r
    y
end

nanmean(x) = mean(filter(!isnan,x))
nanmean(x,y) = mapslices(nanmean,x,dims=y)

function optimal_ens(info)
    D, m, N = size(info.ens)
    errs = [[sqrt(mean((nanmean(info.ens[:, (sortperm(info.r_errs[j, :]))[1:i], j], 2) - info.x_trues[:, j]).^2)) for j=1:N] for i=1:m]
    #errs_rand = [[sqrt(mean((nanmean(info.ens[:, shuffle(sortperm(info.r_errs[j, :]))[1:i], j], 2) - info.x_trues[:, j]).^2)) for j=1:N] for i=1:m]
    return mean(hcat(errs...), dims=1)'#, mean(hcat(errs_rand...), dims=1)'
end

function ens_forecast_compare(; model, model_err, integrator, m, M, D, k, k_r, modes,
                             osc_vars, outfreq, Δt, cycles, window, record_length, obs_err_pct,
                             ens_err_pct, transient, brownian_noise, y0, mp)
    u0 = y0

    tree, tree_r, EW_nature, EV_nature, y_nature, r, C1 = Embedding.create_tree(model=model, record_length=record_length,
                                            integrator=integrator, Δt=Δt, u0=u0,
                                            transient=transient, outfreq=outfreq,
                                            obs_err_pct=obs_err_pct, osc_vars=osc_vars,
                                            D=D, M=M, modes=modes, varimax=false, brownian_noise=brownian_noise)

    ssa_info_nature = SSA_Info(EW_nature, EV_nature, r, y_nature)

    E = integrator(model_err, y_nature[end, :], 0.0, m*Δt*outfreq, Δt, inplace=false)[1:outfreq:end, :]'

    stds = std(y_nature, dims=1)
    means = mean(y_nature, dims=1)

    ens_info = ens_forecast.forecast(E=copy(E), model=model, model_err=model_err,
                               integrator=integrator, m=m, Δt=Δt, window=window,
                               cycles=cycles, outfreq=outfreq, D=D, k=k, k_r=k_r,
                               r=r, tree=tree, tree_r=tree_r, osc_vars=osc_vars,
                               stds=stds, means=means, err_pct=ens_err_pct,
                               mp=mp)

    return ens_info, ssa_info_nature
end
end
