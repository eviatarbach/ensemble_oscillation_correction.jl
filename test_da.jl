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

u0 = zeros(9)
y = rk4(Models.rossler, u0, 0., 15000.0, 0.1)
low = y[5001:4:end, :]

x0 = low[end, :]
E = Integrators.rk4(Models.rossler2, x0, 0.0, 0.4*((2*(M - 1) + 1)*D + m), 0.1, 4)'
#E = hcat([Integrators.rk4_inplace(rossler, x0, 0.0, last, 0.01) for last=range(10.0, stop=100.0, length=20)]...)

H = zeros(9, 9)
H = diagm(0=>ones(9))
R = Symmetric(diagm(0 => [0.4, 0.4, 0.02, 0.4, 0.4, 0.02, 0.4, 0.4, 0.02]))

errs2, errs_free2, x_hist2 = DA.ETKF(E[:, end-19:end], Models.rossler, Models.rossler2, R, 20, cycles=1000; H=H, window=0.4)

# errs, errs_free = DA.run_da(E, rossler, Symmetric(diagm(0 => 0.1*ones(3))), 20,
#                            cycles=1000, H=I)
