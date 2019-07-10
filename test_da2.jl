using Distributed
using LinearAlgebra
using NearestNeighbors
using Statistics

include("da_ssa2.jl")
include("embedding.jl")
include("models.jl")
include("integrators.jl")
using .DA_SSA2
using .Embedding
using .Models
using .Integrators

M = 30
m = 20
D = 8
u0 = ones(D)
y = rk4(Models.harmonic, u0, 0., 15000.0, 0.1)[5000:5:end, :]

x0 = y[end, :]
E = Integrators.rk4(Models.harmonic2, x0, 0.0, 0.4*((2*(M - 1) + 1)*D + m), 0.1, 5)'

EW, EV, X = mssa(y, 30)

r = Embedding.reconstruct(X, EV, M, D, 1:8);
osc = sum(r[1:8, :, :], dims=1)[1, :, :]
pcs = 4
_, _, v = svd(osc)
v = v[:, 1:pcs]
tree = KDTree(copy((osc[10000:end, :]*v)'))
k = 10
obs_errs = Embedding.estimate_errs(osc, tree, y[1:10000, :], v)[k + 1, :]

tree = KDTree(copy((osc*v)'))

H = vcat(diagm(0=>ones(D)), diagm(0=>ones(D)))

R = Symmetric(diagm(0 => vcat(1e6*ones(D),
                              1e-8*var(osc, dims=1)[1, :])))

x0 = E[:, end]
oracle = vcat(reverse(rk4(Models.harmonic, x0, 0.0, -14.5, -0.1, 5), dims=1),
              x0', rk4(Models.harmonic, x0, 0.0, 1014.5, 0.1, 5))
osc2 = vcat([sum(Embedding.transform(oracle, i, EV, M, D, 1:8), dims=1) for i=30:2029]...)

errs, errs_free, full_x_hist, B = DA_SSA2.ETKF(E[:, end-m+1:end], Models.harmonic, Models.harmonic2, R, m, tree, osc,
                             v, k, D, M, osc2; window=0.5, outfreq=5, H=H, cycles=2000, ave_window=100)
