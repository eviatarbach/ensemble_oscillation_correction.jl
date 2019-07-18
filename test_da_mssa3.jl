using LinearAlgebra
using NearestNeighbors
using Distributions

include("da_ssa3.jl")
include("embedding.jl")
include("models.jl")
include("integrators.jl")
using .DA_SSA3
using .Embedding
using .Models
using .Integrators

M = 30
D = 6

u0 = ones(D)
y = rk4(Models.harmonic2, u0, 0., 15000.0, 0.1)
low = y[5001:4:end, :]
EW, EV, X = mssa(low, 30)
T = obs_operator(EV, M, D, 1) + obs_operator(EV, M, D, 2) + obs_operator(EV, M, D, 3) + obs_operator(EV, M, D, 4) + obs_operator(EV, M, D, 5) + obs_operator(EV, M, D, 6)

p = D
n = D
H = diagm(0=>ones(D))

k = 30
r = Embedding.reconstruct(X, EV, M, D, 1:6);
osc = sum(r[1:6, :, :], dims=1)[1, :, :]

R = Symmetric(diagm(0 => 0.1*ones(D)))
# #project(tree, low[100, :], low, osc, k, 100)
#
m = 20
x0 = low[end, :]
E = Integrators.rk4(Models.harmonic2, x0, 0.0, 0.4*((M - 1)*D + m), 0.1, 4)'

x_hist = zeros(m, M - 1, D)

dist = MvNormal(zeros(D), diagm(0=>0.01*ones(D)))
for i=1:m
    x_hist[i, :, :] = E[:, 1:(M - 1)]' + rand(dist, M - 1)'
    #x_hist[i, :, :] = E[:, i:i+(2*(M - 1) + 1) - 1]'
end
x_hist = reshape(x_hist, m, (M - 1)*D)'

u0 = ones(D)
y = rk4(Models.harmonic, u0, 0., 15000.0, 0.1)
low = y[5001:4:end, :]
EW, EV, X = mssa(low, 30)

x0 = E[:, M]
oracle = vcat(reverse(rk4(Models.harmonic, x0, 0.0, -11.6, -0.1, 4), dims=1),
              x0', rk4(Models.harmonic, x0, 0.0, 411.6, 0.1, 4))
osc2 = vcat([sum(Embedding.transform(oracle, i, EV, M, D, 1:6), dims=1) for i=30:1029]...)

E = hcat([E[:, M] for i=1:20]...)
for i=1:m
    E[:, i] = E[:, i] + rand(dist, 1)
end

errs, errs_free, full_x_hist = DA_SSA3.ETKF_SSA(E, Models.harmonic, Models.harmonic2, R, m, 0, osc,
                               0, 20, D, M, osc2, T, copy(x_hist); window=0.4, H=H, outfreq=4,
                               cycles=1000)
