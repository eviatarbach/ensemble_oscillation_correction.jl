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
D = 18
n_modes = 2
model = Models.rossler
model_err = Models.rossler2
outfreq = 4
Δt = 0.1
m = 20
cycles = 1000

u0 = ones(D)
y1 = rk4(model, u0, 0., 15000.0, Δt)[5001:outfreq:end, :]
y2 = rk4(model_err, u0, 0., 15000.0, Δt)[5001:outfreq:end, :]

EW1, EV1, X1 = mssa(y1, M)
EW2, EV2, X2 = mssa(y2, M)

r1 = sum(Embedding.reconstruct(X1, EV1, M, D, 1:n_modes), dims=1)[1, :, :]
r2 = sum(Embedding.reconstruct(X2, EV1, M, D, 1:n_modes), dims=1)[1, :, :]

tree1 = KDTree(copy(y1'))
tree2 = KDTree(copy(y2'))

p = D
n = D
H = diagm(0=>ones(D))

R = Symmetric(diagm(0 => 3.0*ones(D)))

x0 = Integrators.rk4_inplace(model_err, y1[end, :], 0.0, 30*Δt*outfreq, Δt)

dist = MvNormal(zeros(D), diagm(0=>0.01*ones(D)))

E = hcat([x0 for i=1:m]...)
for i=1:m
    E[:, i] = E[:, i] + rand(dist, 1)
end

errs, errs_free, full_x_hist = DA_SSA3.ETKF_SSA(copy(E), model, model_err, R, m,
                               20, D, M, r1, r2, tree1, tree2; window=outfreq*Δt, H=H, outfreq=outfreq,
                               cycles=cycles, inflation=1.01)

errs_no, _, full_x_hist = DA_SSA3.ETKF_SSA(copy(E), model, model_err, R, m,
                              20, D, M, r1, r2, tree1, tree2; window=outfreq*Δt, H=H, outfreq=outfreq,
                              cycles=cycles, psrm=false, inflation=1.01)
