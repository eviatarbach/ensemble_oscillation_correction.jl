module enoc

include("analog.jl")
using .Analog

using LinearAlgebra
using Statistics
using Random
using Serialization

using Distributions
using NearestNeighbors
using PyCall

include("ssa.jl")
include("ens_forecast.jl")
include("ssa_varimax.jl")

using .SSA
using .ens_forecast
using .ssa_varimax

nanmean(x) = mean(filter(!isnan,x))
nanmean(x,y) = mapslices(nanmean,x,dims=y)

tsa = pyimport("statsmodels.tsa.api")

function optimal_ens(info)
    D, m, N = size(info.ens)
    errs = [[sqrt(mean((nanmean(info.ens[:, (sortperm(info.r_errs[j, :]))[1:i], j], 2) - info.x_trues[:, j]).^2)) for j=1:N] for i=1:m]
    errs_rand = [[sqrt(mean((nanmean(info.ens[:, shuffle(sortperm(info.r_errs[j, :]))[1:i], j], 2) - info.x_trues[:, j]).^2)) for j=1:N] for i=1:m]
    return mean(hcat(errs...), dims=1)', mean(hcat(errs_rand...), dims=1)'
end

function setup(; model, Δt, outfreq, obs_err_pct, M, record_length, transient,
               y0, D, osc_vars, modes, integrator, pcs=nothing, varimax, da,
               window, k, k_r)
    y = integrator(model, y0, 0., record_length*outfreq*Δt, Δt; inplace=false)[1:outfreq:end, :][(transient + 1):end, :]
    if (obs_err_pct > 0)
        R = Symmetric(diagm(0 => obs_err_pct*std(y, dims=1)[1, :]))
        obs_err = MvNormal(zeros(D), R)
        y = y + rand(obs_err, size(y)[1])'
    end

    ssa_info = ssa_decompose(y[:, osc_vars], M)

    if varimax
        ssa_info.eig_vals, ssa_info.eig_vecs = varimax_rotate!(ssa_info.eig_vals, ssa_info.eig_vecs, M, ssa_info.D, maximum(modes))
    end

    r = ssa_reconstruct(ssa_info, modes, sum_modes=true)

    if da
        R = error_cov(y, r, M, window, k, k_r, osc_vars)
    else
        R = false
    end

    if pcs === nothing
        tree = KDTree(copy(y'))
        tree_r = KDTree(copy(r'))
    else
        _, _, v = svd(r)
        v = v[:, 1:pcs]
        tree = KDTree(copy(y'))
    end

    return tree, tree_r, ssa_info, y, r, R
end

function run(; model, model_err, integrator, m, M, D, k, k_r, modes, osc_vars,
             outfreq, Δt, cycles, window, record_length, obs_err_pct,
             ens_err_pct, transient, y0, mp, varimax,
             check_bounds, test_time=false, da, inflation, y_fcst, α=nothing,
             preload=nothing)

    if (preload === nothing) | !isfile(preload)
        tree, tree_r, ssa_info, y, r, R = setup(model=model, record_length=record_length,
                                                integrator=integrator, Δt=Δt, y0=y0,
                                                transient=transient, outfreq=outfreq,
                                                obs_err_pct=obs_err_pct, osc_vars=osc_vars,
                                                D=D, M=M, modes=modes, varimax=varimax,
                                                da=da, window=window, k=k, k_r=k_r)

        if y_fcst
            var_model = tsa.VAR(y).fit(maxlags=15, ic="aic")
        else
            var_model = nothing
        end

        serialize(preload, (tree, tree_r, ssa_info, y, r, var_model))
    else
        tree, tree_r, ssa_info, y, r, var_model = deserialize(preload)
    end

    E = integrator(model_err, y[end, :], 0.0, m*Δt*outfreq, Δt, inplace=false)[1:outfreq:end, :]'

    stds = std(y, dims=1)[:]
    means = mean(y, dims=1)
    bounds = (minimum(y, dims=1)[:], maximum(y, dims=1)[:])

    ens_info = forecast(E=copy(E), model=model, model_err=model_err,
                        integrator=integrator, m=m, Δt=Δt, window=window,
                        cycles=cycles, outfreq=outfreq, D=D, k=k, k_r=k_r, r=r,
                        tree=tree, tree_r=tree_r, osc_vars=osc_vars, means=means,
                        stds=stds, err_pct=ens_err_pct, mp=mp,
                        check_bounds=check_bounds, test_time=test_time,
                        bounds=bounds, da=da, R=R, inflation=inflation,
                        var_model=var_model, α=α)

    return ens_info, ssa_info
end
end
