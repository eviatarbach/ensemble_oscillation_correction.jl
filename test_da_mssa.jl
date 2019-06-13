using Distributed
using LinearAlgebra
using NearestNeighbors

include("da_ssa.jl")
include("embedding.jl")
include("models.jl")
include("integrators.jl")
using .DA_SSA
using .Embedding
using .Models
using .Integrators

# x0 = zeros(3)
# E = hcat([rk4_inplace(rossler, x0, 0.0, last, 0.01) for last=range(10.0, stop=100.0, length=20)]...)
#
# H = zeros(2, 3)
# H[1, 1] = 1
# H[2, 2] = 1
# R = Symmetric(diagm(0 => 0.1*ones(2)))
#
# errs, errs_free = ETKF(E, rossler, R, 20, cycles=1000; H=H)

# errs, errs_free = DA.run_da(E, rossler, Symmetric(diagm(0 => 0.1*ones(3))), 20,
#                            cycles=1000, H=I)

M = 30
D = 3
u0 = rand(D)
y = rk4(rossler, u0, 0., 1500.0, 0.01)
low = y[50001:40:end, :]
EW, EV, X = mssa(low, 30)
T = obs_operator(EV, M, D, 1) + obs_operator(EV, M, D, 2)

p = D + D  # obs. of model variables + obs. of D channels
n = (2*(M - 1) + 1)*D  # M - 1 past and M - 1 future states, + present
H = zeros(p, n)
H = reshape(H, p, 2*(M - 1) + 1, D)
H[1, M, 1] = 1
H[2, M, 2] = 1
H[3, M, 3] = 1
#H[1, 2 + M] = 1
#H[2, 2 + M] = 1
#H[3, 2 + M] = 1
H = reshape(H, p, n)
H[4:6, :] = T'

x = reshape(low[1:1+58, :], 177, 1)
y = H*x

# 20 seems to be best k
k = 20
r = Embedding.reconstruct(X, EV, M, D, 1:2);
osc = sum(r[1:2, :, :], dims=1)[1, :, :]
tree = KDTree(copy(low'))
R = Symmetric(diagm(0 => [0.1, 0.1, 0.1, 0.42, 0.42, 0.42]))
#project(tree, low[100, :], low, osc, k, 100)

x0 = zeros(3)
E = hcat([Integrators.rk4_inplace(rossler, x0, 0.0, last, 0.01) for last=0.4:0.4:0.4*(M-1)]...)

errs, errs_free = DA_SSA.ETKF_SSA(E[:, end-19:end], rossler, R, 20, tree, osc,
                                  low, 20, D, M, E'; H=H, cycles=1000)
