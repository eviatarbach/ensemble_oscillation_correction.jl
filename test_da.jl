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

x0 = zeros(3)
E = hcat([Integrators.rk4_inplace(rossler, x0, 0.0, last, 0.01) for last=0.4:0.4:0.4*(29)]...)
#E = hcat([Integrators.rk4_inplace(rossler, x0, 0.0, last, 0.01) for last=range(10.0, stop=100.0, length=20)]...)

H = zeros(2, 3)
H[1, 1] = 1
H[2, 2] = 1
R = Symmetric(diagm(0 => 1.0*ones(2)))

errs2, errs_free2, x_hist2 = DA.ETKF(E[:, end-19:end], rossler, R, 20, cycles=1000; H=H, window=0.4)

# errs, errs_free = DA.run_da(E, rossler, Symmetric(diagm(0 => 0.1*ones(3))), 20,
#                            cycles=1000, H=I)
