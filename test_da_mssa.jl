using Distributed
using LinearAlgebra

include("da.jl")
include("embedding.jl")
include("models.jl")
include("integrators.jl")
using .DA
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
EW, EV, X = mssa(low[:, 1:1], 30)
t = obs_operator(EV, M, 1, 1) + obs_operator(EV, M, 1, 2)

p = D + 1
n = (D - 1) + 2*(M - 1) + 1
H = zeros(p, n)
H[2, 1] = 1
H[3, 2] = 1
H[1, 2 + M] = 1
H[p, D:end] = t
