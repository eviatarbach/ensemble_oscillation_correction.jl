module run_da_ssa

using LinearAlgebra
using NearestNeighbors
using Distributions
using Distributed

rmprocs(procs())
addprocs()

include("da_ssa.jl")
include("embedding.jl")

using .DA_SSA
using .Embedding

function etkf_da_ssa(model, model_err, M, D, modes, integrator, outfreq, Δt,
                    m, cycles, window, inflation, record_length, obs_err_pct,
                    ens_err_pct)
    u0 = randn(D)
    y1 = integrator(model, u0, 0., record_length, Δt; inplace=false)[5001:outfreq:end, :]
    R = Symmetric(diagm(0 => obs_err_pct*std(y1, dims=1)[1, :]))
    obs_err = MvNormal(zeros(D), R/2)
    y1 = y1 + rand(obs_err, size(y1)[1])'
    y2 = integrator(model_err, u0, 0., record_length, Δt; inplace=false)[5001:outfreq:end, :]

    EW1, EV1, X1 = mssa(y1, M)
    #EW1, EV1 = Embedding.var_rotate!(EW1, EV1, M, D, 20)
    EW2, EV2, X2 = mssa(y2, M)
    #EW2, EV2 = Embedding.var_rotate!(EW2, EV2, M, D, 20)

    r1 = sum(Embedding.reconstruct(X1, EV2, M, D, modes), dims=1)[1, :, :]
    r2 = sum(Embedding.reconstruct(X2, EV2, M, D, modes), dims=1)[1, :, :]

    tree1 = KDTree(copy(y1'))
    tree2 = KDTree(copy(y2'))

    p = D
    n = D
    H = diagm(0=>ones(D))

    x0 = integrator(model_err, y1[end, :], 0.0, M*Δt*outfreq, Δt)

    dist = MvNormal(zeros(D), diagm(0=>ens_err_pct*std(y1, dims=1)[1, :]))

    E = hcat([x0 for i=1:m]...)
    for i=1:m
        E[:, i] = E[:, i] + rand(dist, 1)
    end

    errs, errs_free, B = DA_SSA.ETKF_SSA(copy(E), model, model_err, R, m,
                                   D, M, r1, r2, tree1, tree2; window=window,
                                   H=H, outfreq=outfreq, cycles=cycles,
                                   inflation=inflation, integrator=integrator)

    errs_no, _, B2 = DA_SSA.ETKF_SSA(copy(E), model, model_err, R, m,
                                  D, M, r1, r2, tree1, tree2; window=window,
                                  H=H, outfreq=outfreq, cycles=cycles, psrm=false,
                                  inflation=inflation, integrator=integrator)

    return errs, errs_no, errs_free
end
end
