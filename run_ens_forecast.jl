module run_ens_forecast

using LinearAlgebra
using Statistics

using Distributions
using NearestNeighbors

include("./embedding.jl")
using .Embedding

include("ens_forecast.jl")
using .ens_forecast

struct SSA_Info
    EW
    EV
end

function ens_forecast_compare(; model, model_err, integrator, m, M, D, k, k_r, modes,
                             osc_vars, outfreq, Δt, cycles, window, record_length, obs_err_pct,
                             ens_err_pct, transient)
    u0 = randn(D)

    tree, tree_r, EW_nature, EV_nature, y_nature, r, C1 = Embedding.create_tree(model=model, record_length=record_length,
                                            integrator=integrator, Δt=Δt, u0=u0,
                                            transient=transient, outfreq=outfreq,
                                            obs_err_pct=obs_err_pct, osc_vars=osc_vars,
                                            D=D, M=M, modes=modes)

    ssa_info_nature = SSA_Info(EW_nature, EV_nature)

    R = Symmetric(diagm(0 => obs_err_pct*std(y_nature, dims=1)[1, :]))

    E = integrator(model_err, y_nature[end, :], 0.0, m*Δt*outfreq, Δt, inplace=false)[1:outfreq:end, :]'

    stds = std(y_nature, dims=1)

    da_info1 = ens_forecast.forecast(E=copy(E), model=model, model_err=model_err,
                               integrator=integrator, m=m, Δt=Δt, window=window,
                               cycles=cycles, outfreq=outfreq, D=D, k=k, k_r=k_r,
                               r=r, tree=tree, tree_r=tree_r, osc_vars=osc_vars,
                               stds=stds, err_pct=ens_err_pct, correction=true)

    da_info2 = ens_forecast.forecast(E=copy(E), model=model, model_err=model_err,
                               integrator=integrator, m=m, Δt=Δt, window=window,
                               cycles=cycles, outfreq=outfreq, D=D, k=k, k_r=k_r,
                               r=r, tree=tree, tree_r=tree_r, osc_vars=osc_vars,
                               stds=stds, err_pct=ens_err_pct, correction=false)

    return da_info1, da_info2, ssa_info_nature
end
end
