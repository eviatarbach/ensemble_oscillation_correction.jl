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
D = 8
n_modes = 4

u0 = ones(D)
y = rk4(Models.harmonic, u0, 0., 15000.0, 0.1)
low = y[5001:4:end, :]
EW1, EV1, X1 = mssa(low, 30)

p = D
n = D
H = diagm(0=>ones(D))

R = Symmetric(diagm(0 => 0.1*ones(D)))

m = 20
x0 = low[end, :]
E = Integrators.rk4(Models.harmonic2, x0, 0.0, 0.4*((M - 1)*D + m), 0.1, 4)'

dist = MvNormal(zeros(D), diagm(0=>0.01*ones(D)))

u0 = ones(D)
y = rk4(Models.harmonic2, u0, 0., 15000.0, 0.1)
low = y[5001:4:end, :]
EW2, EV2, X2 = mssa(low, 30)

x0 = E[:, M]

oracle1 = oracle1 = vcat(reverse(rk4(Models.harmonic, x0, 0.0, -11.6, -0.1, 4), dims=1), x0', rk4(Models.harmonic, x0, 0.0, 411.6, 0.1, 4))
EW3, EV3, X3 = mssa(oracle1, 30)
osc1 = sum(Embedding.reconstruct(X3, EV1, M, D, 1:n_modes), dims=1)[1, 30:end, :]

#oracle2 = vcat(x0', rk4(Models.harmonic2, x0, 0.0, 400.0, 0.1, 4))
#EW3, EV3, X3 = mssa(oracle2, 30)
#osc2 = sum(Embedding.reconstruct(X3, EV1, M, D, 1:n_modes), dims=1)[1, :, :]

E = hcat([x0 for i=1:20]...)
for i=1:m
    E[:, i] = E[:, i] + rand(dist, 1)
end

T = sum(Embedding.obs_operator(EV1, M, D, i) for i=1:n_modes)

errs, errs_free, full_x_hist = DA_SSA3.ETKF_SSA(copy(E), Models.harmonic, Models.harmonic2, R, m, 0, 0,
                               0, 20, D, M, osc1, T; window=0.4, H=H, outfreq=4,
                               cycles=1000)
