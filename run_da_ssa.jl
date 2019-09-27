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
    y1 = integrator(model, u0, 0., record_length, Δt; inplace=false)[(transient + 1):outfreq:end, :]

    R = Symmetric(diagm(0 => obs_err_pct*std(y1, dims=1)[1, :]))
    obs_err = MvNormal(zeros(D), R/2)
    ens_err = MvNormal(zeros(D), diagm(0=>ens_err_pct*std(y1, dims=1)[1, :]))
    x0 = integrator(model_err, y1[end, :], 0.0, M*Δt*outfreq, Δt)

    y1 = y1[:, osc_vars] + (rand(obs_err, size(y1)[1])')[:, osc_vars]

    y2 = integrator(model_err, u0, 0., record_length, Δt; inplace=false)[(transient + 1):outfreq:end, osc_vars]

    EW1, EV1, X1 = Embedding.mssa(y1, M)
    ssa_info1 = SSA_Info(EW1, EV1)
    #EW1, EV1 = Embedding.var_rotate!(EW1, EV1, M, D, 20)
    EW2, EV2, X2 = Embedding.mssa(y2, M)
    ssa_info2 = SSA_Info(EW2, EV2)
    #EW2, EV2 = Embedding.var_rotate!(EW2, EV2, M, D, 20)

    r1 = sum(Embedding.reconstruct(X1, EV1, M, length(osc_vars), modes),
             dims=1)[1, :, :]
    r2 = sum(Embedding.reconstruct(X2, EV2, M, length(osc_vars), modes),
             dims=1)[1, :, :]

    tree1 = KDTree(copy(y1'))
    tree2 = KDTree(copy(y2'))

    H = diagm(0=>ones(D))

    E = hcat([x0 for i=1:m]...)

    E += rand(ens_err, m)

    da_info1 = DA_SSA.ETKF_SSA(E=copy(E), model=model, model_err=model_err,
                              integrator=integrator, R=R, m=m, Δt=Δt,
                              window=window, cycles=cycles, outfreq=outfreq,
                              D=D, k=k, M=M, r1=r1, r2=r2, tree1=tree1,
                              tree2=tree2, H=H, psrm=true, inflation=inflation1,
                              osc_vars=osc_vars, cov=cov)

    da_info2 = DA_SSA.ETKF_SSA(E=copy(E), model=model, model_err=model_err,
                               integrator=integrator, R=R, m=m, Δt=Δt,
                               window=window, cycles=cycles, outfreq=outfreq,
                               D=D, k=k, M=M, r1=r1, r2=r2, tree1=tree1,
                               tree2=tree2, H=H, psrm=false,
                               inflation=inflation2, osc_vars=osc_vars, cov=cov)

    return da_info1, da_info2, ssa_info1, ssa_info2
end
end
