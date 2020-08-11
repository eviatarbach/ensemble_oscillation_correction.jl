include("ssa.jl")
using .SSA

n = 101
D = 2
J = 10
M = 20

x = zeros(n, D, J)

y = hcat([sin.(a:0.3:(a+30)) for a=1:10]...)
x[:, 1, :] = y
x[:, 2, :] = -2*y

# M-SSA with multiple subseries
ssa_info = ssa_decompose(x, M)
r = SSA.ssa_reconstruct(ssa_info, 1:2, sum_modes=true)
@assert isapprox(x, r)
@assert size(x) == size(r)

# Regular M-SSA
x_mssa = x[:, :, 1]
ssa_info = ssa_decompose(x_mssa, M)
r = SSA.ssa_reconstruct(ssa_info, 1:2, sum_modes=true)
@assert isapprox(x_mssa, r)
@assert size(x_mssa) == size(r)

# Single-channel SSA
x_ssa = x[:, 1, 1]
ssa_info = ssa_decompose(x_ssa, M)
r = SSA.ssa_reconstruct(ssa_info, 1:2, sum_modes=true)
@assert isapprox(x_ssa, r)
@assert size(x_ssa) == size(r)

# Single-channel SSA with multiple subseries
x_ssa2 = x[:, 1:1, :]
ssa_info = ssa_decompose(x_ssa2, M)
r = SSA.ssa_reconstruct(ssa_info, 1:2, sum_modes=true)
@assert isapprox(x_ssa2, r)
@assert size(x_ssa2) == size(r)
