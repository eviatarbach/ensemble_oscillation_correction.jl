using LinearAlgebra
using NearestNeighbors
using Distributions
using Distributed

addprocs()

include("da_ssa3.jl")
include("embedding.jl")
@everywhere include("models.jl")
@everywhere include("integrators.jl")
using .DA_SSA3
using .Embedding
@everywhere using .Models
@everywhere using .Integrators

M = 30
D = 128
modes = 1:4
model = "true"
model_err = "false"
integrator = Integrators.ks_integrate
outfreq = 4
Δt = 0.25
m = 20
cycles = 1000

R = Symmetric(diagm(0 => 0.1*ones(D)))
obs_err = MvNormal(zeros(D), R/2)

u0 = randn(D)
y1 = integrator(model, u0, 0., 15000.0, Δt; inplace=false)[5001:outfreq:end, :]
y1 = y1 + rand(obs_err, size(y1)[1])'
y2 = integrator(model_err, u0, 0., 15000.0, Δt; inplace=false)[5001:outfreq:end, :]

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

x0 = integrator(model_err, y1[end, :], 0.0, 30*Δt*outfreq, Δt)

dist = MvNormal(zeros(D), diagm(0=>0.01*ones(D)))

E = hcat([x0 for i=1:m]...)
for i=1:m
    E[:, i] = E[:, i] + rand(dist, 1)
end

errs, errs_free, full_x_hist = DA_SSA3.ETKF_SSA(copy(E), model, model_err, R, m,
                               D, M, r1, r2, tree1, tree2; window=outfreq*Δt,
                               H=H, outfreq=outfreq, cycles=cycles,
                               inflation=1.01, integrator=integrator)

errs_no, _, full_x_hist = DA_SSA3.ETKF_SSA(copy(E), model, model_err, R, m,
                              D, M, r1, r2, tree1, tree2; window=outfreq*Δt,
                              H=H, outfreq=outfreq, cycles=cycles, psrm=false,
                              inflation=1.01, integrator=integrator)
