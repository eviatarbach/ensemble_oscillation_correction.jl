module run_da_ssa

using LinearAlgebra
using Statistics

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

    tree, tree_r, EW_nature, EV_nature, y_nature, r, C1 = Embedding.create_tree(model=model, record_length=record_length,
                                            integrator=integrator, Δt=Δt, u0=u0,
                                            transient=transient, outfreq=outfreq,
                                            obs_err_pct=obs_err_pct, osc_vars=osc_vars,
                                            D=D, M=M, modes=modes)

    tree_err, tree_r_err, EW_err, EV_err, y_err, r_err, C2 = Embedding.create_tree(model=model_err, record_length=record_length,
                                             integrator=integrator, Δt=Δt, u0=u0,
                                             transient=transient, outfreq=outfreq,
                                             obs_err_pct=0, osc_vars=1:D, D=D, M=M, modes=modes)

    # C_conds = Embedding.precomp(C2, M, D, 'b')
    # C_conds2 = Embedding.precomp(C1, M, D, 'b')

    ssa_info_nature = SSA_Info(EW_nature, EV_nature)
    #ssa_info_model = SSA_Info(EW_model, EV_model)

    R = Symmetric(diagm(0 => obs_err_pct*std(y_nature, dims=1)[1, :]))
    ens_err = MvNormal(zeros(D), diagm(0=>ens_err_pct*std(y_nature, dims=1)[1, :]))

    H = diagm(0=>ones(D))
    #for i=7:D
    #    H[i, i] = 0
    #end

    E = integrator(model_err, y_nature[end, :], 0.0, m*Δt*outfreq, Δt, inplace=false)[1:outfreq:end, :]'
    #E = hcat([x0 for i=1:m]...)

    #E += rand(ens_err, m)

    stds = std(y_nature, dims=1)

    da_info1 = DA_SSA.ETKF_SSA(E=copy(E), model=model, model_err=model_err,
                              integrator=integrator, R=R, m=m, Δt=Δt,
                              window=window, cycles=cycles, outfreq=outfreq,
                              D=D, k=k, M=M, r=r, r_err=r_err, tree_err=tree_err,
                              tree_r_err=tree_r_err, tree=tree,
                              tree_r=tree_r, H=H, psrm=true, inflation=inflation1,
                              osc_vars=osc_vars, cov=cov, modes=modes, stds=stds)

    da_info2 = DA_SSA.ETKF_SSA(E=copy(E), model=model, model_err=model_err,
                               integrator=integrator, R=R, m=m, Δt=Δt,
                               window=window, cycles=cycles, outfreq=outfreq,
                               D=D, k=k, M=M, r=r, r_err=r_err, tree_err=tree_err,
                               tree_r_err=tree_r_err, tree=tree,
                               tree_r=tree_r, H=H, psrm=false,
                               inflation=inflation2, osc_vars=osc_vars, cov=cov,
                               modes=modes, stds=stds)

    return da_info1, da_info2, ssa_info_nature
end
end
