module run_da_ssa

using LinearAlgebra

using Distributions
using NearestNeighbors

include("./embedding.jl")
using .Embedding

include("da_ssa.jl")
using .DA_SSA

struct SSA_Info
    EW
    EV
end

function etkf_da_ssa_compare(; model, model_err, integrator, m, M, D, k, modes,
                             osc_vars, outfreq, Δt, cycles, window, inflation1,
                             inflation2, record_length, obs_err_pct,
                             ens_err_pct, transient, cov=false)
    u0 = randn(D)

    tree_nature, EW_nature, EV_nature, y_nature, r_nature = Embedding.create_tree(model=model, record_length=record_length,
                                            integrator=integrator, Δt=Δt, u0=u0,
                                            transient=transient, outfreq=outfreq,
                                            obs_err_pct=obs_err_pct, osc_vars=osc_vars,
                                            D=D, M=M, modes=modes)

    tree_model, EW_model, EV_model, y_model, r_model = Embedding.create_tree(model=model_err, record_length=record_length,
                                            integrator=integrator, Δt=Δt, u0=u0,
                                            transient=transient, outfreq=outfreq,
                                            obs_err_pct=0, osc_vars=1:D, D=D, M=M, modes=modes)

    ssa_info_nature = SSA_Info(EW_nature, EV_nature)
    ssa_info_model = SSA_Info(EW_model, EV_model)

    R = Symmetric(diagm(0 => obs_err_pct*std(y_nature, dims=1)[1, :]))
    ens_err = MvNormal(zeros(D), diagm(0=>ens_err_pct*std(y_nature, dims=1)[1, :]))

    H = diagm(0=>ones(D))

    x0 = integrator(model_err, y_nature[end, :], 0.0, M*Δt*outfreq, Δt)
    E = hcat([x0 for i=1:m]...)

    E += rand(ens_err, m)

    da_info1 = DA_SSA.ETKF_SSA(E=copy(E), model=model, model_err=model_err,
                              integrator=integrator, R=R, m=m, Δt=Δt,
                              window=window, cycles=cycles, outfreq=outfreq,
                              D=D, k=k, M=M, r1=r_nature, r2=r_model, tree1=tree_nature,
                              tree2=tree_model, H=H, psrm=true, inflation=inflation1,
                              osc_vars=osc_vars, cov=cov)

    da_info2 = DA_SSA.ETKF_SSA(E=copy(E), model=model, model_err=model_err,
                               integrator=integrator, R=R, m=m, Δt=Δt,
                               window=window, cycles=cycles, outfreq=outfreq,
                               D=D, k=k, M=M, r1=r_nature, r2=r_model, tree1=tree_nature,
                               tree2=tree_model, H=H, psrm=false,
                               inflation=inflation2, osc_vars=osc_vars, cov=cov)

    return da_info1, da_info2, ssa_info_nature, ssa_info_model
end
end
