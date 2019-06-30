using LinearAlgebra
using NearestNeighbors
using Distributions

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
D = 4
u0 = randn(D)
y = rk4(Models.pendulum, u0, 0., 15000.0, 0.1)[5000:5:end, :]
EW, EV, X = mssa(y, 30)
T = obs_operator(EV, M, D, 1) + obs_operator(EV, M, D, 2)

p = D + D  # obs. of model variables + obs. of D channels
n = (2*(M - 1) + 1)*D  # M - 1 past and M - 1 future states, + present
H = zeros(p, n)
H = reshape(H, p, 2*(M - 1) + 1, D)
for i=1:D
    H[i, M, i] = 1
end

H = reshape(H, p, n)
H[D+1:end, :] = T'

#x = reshape(low[1:1+58, :], 177, 1)
#y = H*x

# 20 seems to be best k
k = 5
r = Embedding.reconstruct(X, EV, M, D, 1:2);
osc = sum(r[1:2, :, :], dims=1)[1, :, :]

pcs = 2
_, _, v = svd(osc)
v = v[:, 1:pcs]
tree = KDTree(copy((osc*v)'))

R = Symmetric(diagm(0 => vcat(1.0*ones(4),
                              1e6*[0.0173403, 0.0375418, 1.40342, 1.27895])))
# #project(tree, low[100, :], low, osc, k, 100)
#
m = 20
x0 = y[end, :]
E = Integrators.rk4(Models.pendulum2, x0, 0.0, 0.5*((2*(M - 1) + 1)*D + m), 0.1, 5)'

x_hist = zeros(m, 2*(M - 1) + 1, D)

# #x_hist = reshape(x_hist, m, 2*(M - 1) + 1, D)
dist = MvNormal(zeros(D), diagm(0=>0.005*ones(D)))
for i=1:m
    x_hist[i, :, :] = E[:, 1:1+(2*(M - 1) + 1) - 1]' + rand(dist, 2*(M - 1) + 1)'
    #x_hist[i, :, :] = E[:, i:i+(2*(M - 1) + 1) - 1]'
end
x_hist = reshape(x_hist, m, (2*(M - 1) + 1)*D)'

errs, errs_free, full_x_hist, B = DA_SSA.ETKF_SSA(copy(x_hist), Models.pendulum, Models.pendulum2, R, m, tree, osc,
                               v, k, D, M; window=0.5, outfreq=5, H=H, cycles=1000)
