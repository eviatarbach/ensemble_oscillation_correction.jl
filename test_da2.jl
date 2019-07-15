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
D = 3
u0 = ones(27)
y = rk4(Models.harmonic, u0, 0., 10000.0, 0.1)[1:5:end, :]

x0 = y[end, :]
E = Integrators.rk4(Models.harmonic2, x0, 0.0, 0.4*((2*(M - 1) + 1)*D + m), 0.1, 5)'

EW, EV, X = mssa(y[:, 25:end], 30)

r = Embedding.reconstruct(X, EV, M, D, 1:8);
osc = sum(r[1:8, :, :], dims=1)[1, :, :]
#pcs = 4
#_, _, v = svd(osc)
#v = v[:, 1:pcs]
#tree = KDTree(copy((osc[10000:end, :]*v)'))
#k = 10
#obs_errs = Embedding.estimate_errs(osc, tree, y[1:10000, :], v)[k + 1, :]

#tree = KDTree(copy((osc*v)'))

H = zeros(6, 27)
H[1:3, 25:27] = diagm(0=>ones(3))
H[4:6, 25:27] = diagm(0=>ones(3))

R = Symmetric(diagm(0 => vcat(10.0*ones(3),
                              0.1*ones(3))))

x0 = E[:, end]
oracle = vcat(reverse(rk4(Models.harmonic, x0, 0.0, -14.5, -0.1, 5), dims=1),
              x0', rk4(Models.harmonic, x0, 0.0, 5014.5, 0.1, 5))
osc2 = vcat([sum(Embedding.transform(oracle[:, 25:27], i, EV, M, D, 1:8), dims=1) for i=30:10029]...)

errs, errs_free, full_x_hist, B = DA_SSA2.ETKF(E[:, end-m+1:end], Models.harmonic, Models.harmonic2, R, m, tree, osc,
                             v, k, 27, M, osc2; window=0.5, outfreq=5, H=H, cycles=10000, ave_window=100)
