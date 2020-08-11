module run_ens_forecast

using LinearAlgebra
using Statistics
using Random

using Distributions
using NearestNeighbors

include("./ssa.jl")
using .SSA

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
    errs_rand = [[sqrt(mean((nanmean(info.ens[:, shuffle(sortperm(info.r_errs[j, :]))[1:i], j], 2) - info.x_trues[:, j]).^2)) for j=1:N] for i=1:m]
    return mean(hcat(errs...), dims=1)', mean(hcat(errs_rand...), dims=1)'
end

function create_tree(; model, Δt, outfreq, obs_err_pct, M, record_length, transient, u0, D,
                     osc_vars, modes, integrator, pcs=nothing, varimax,
                     da, window, k, k_r)
   y = integrator(model, u0, 0., record_length*outfreq*Δt, Δt; inplace=false)[1:outfreq:end, :][(transient + 1):end, :]
   if (obs_err_pct > 0)
      R = Symmetric(diagm(0 => obs_err_pct*std(y, dims=1)[1, :]))
      obs_err = MvNormal(zeros(D), R)
      y = y + rand(obs_err, size(y)[1])'
   end

   ssa_info = ssa_decompose(y[:, osc_vars], M)

   if varimax
      ssa_info.eig_vals, ssa_info.eig_vecs = ssa_varimax.varimax_rotate!(ssa_info.eig_vals, ssa_info.eig_vecs, M, ssa_info.D, maximum(modes))
   end

   r = ssa_reconstruct(ssa_info, modes, sum_modes=true)

   if da
      function find_point(r, tree, p, k, f)
          ind, dist = knn(tree, p, k)
          mask = (ind .+ f) .<= size(tree.data)[1]
          dist = dist[mask]
          ind = ind[mask]
          return sum(dist .* r[validation .+ ind .- 1 .+ f, :], dims=1)/sum(dist)
      end
      validation = round(Int, 0.1*size(y)[1])
      tree = KDTree(copy((y[validation:end, :])'))
      tree_r = KDTree(copy((r[validation:end, :])'))
      errs = Array{Float64}(undef, length(osc_vars), length(M:validation-window))

      for (i, i_p) in enumerate(M:validation-window)
          p = y[i_p, :]
          p2 = find_point(r, tree, p, k, 0)

          forecast = find_point(r, tree_r, p2[:], k_r, window)
          err = r[i_p + window, :] - forecast'

          errs[:, i] = err
      end

      R = cov(errs')
   else
      R = false
   end

   if pcs == nothing
      tree = KDTree(copy(y'))
      tree_r = KDTree(copy(r'))
   else
      _, _, v = svd(r)
      v = v[:, 1:pcs]
      tree = KDTree(copy(y'))
   end

   return tree, tree_r, EW, EV, y, r, C, R
end

function ens_forecast_compare(; model, model_err, integrator, m, M, D, k, k_r, modes,
                             osc_vars, outfreq, Δt, cycles, window, record_length, obs_err_pct,
                             ens_err_pct, transient, brownian_noise, y0, mp, varimax, check_bounds, test_time=false,
                             da, inflation=1.0)
    u0 = y0

    tree, tree_r, EW_nature, EV_nature, y_nature, r, C1, R = Embedding.create_tree(model=model, record_length=record_length,
                                            integrator=integrator, Δt=Δt, u0=u0,
                                            transient=transient, outfreq=outfreq,
                                            obs_err_pct=obs_err_pct, osc_vars=osc_vars,
                                            D=D, M=M, modes=modes, varimax=varimax, brownian_noise=brownian_noise,
                                            da=da, window=window, k=k, k_r=k_r)

    ssa_info_nature = SSA_Info(EW_nature, EV_nature, r, y_nature)

    E = integrator(model_err, y_nature[end, :], 0.0, m*Δt*outfreq, Δt, inplace=false)[1:outfreq:end, :]'

    stds = std(y_nature, dims=1)
    means = mean(y_nature, dims=1)
    bounds = (minimum(y_nature, dims=1)[:], maximum(y_nature, dims=1)[:])

    ens_info = ens_forecast.forecast(E=copy(E), model=model, model_err=model_err,
                               integrator=integrator, m=m, Δt=Δt, window=window,
                               cycles=cycles, outfreq=outfreq, D=D, k=k, k_r=k_r,
                               r=r, tree=tree, tree_r=tree_r, osc_vars=osc_vars,
                               stds=stds, means=means, err_pct=ens_err_pct,
                               mp=mp, check_bounds=check_bounds,
                               test_time=test_time, bounds=bounds, da=da,
                               R=R, inflation=inflation)

    return ens_info, ssa_info_nature
end
end
