using Plots
using Statistics

include("ssa.jl")
include("ssa_cp.jl")
include("models.jl")
include("integrators.jl")

using .SSA
using .SSA_CP
using .Models
using .Integrators

model = Models.colpitts_true
Δt = 0.1
M = 30
D = 6
outfreq = 4
modes = 1:6

y = rk4(model, randn(D), 0., 1550.0 - Δt, Δt, inplace=false)[500:outfreq:end, :]

ssa_info = SSA.ssa_decompose(copy(y)[1:end, :], M)
C_conds = SSA_CP.precomp(ssa_info.C, M, D, 'b')
Xp = SSA_CP.transform(copy(y)[M:end, :], M, 'b', C_conds)
display(heatmap(reverse(Xp[1:100, :], dims=2)', clim=(-15, 15)))
ssa_info_n = SSA.ssa_decompose(copy(y)[M:end, :], M)

x_true = SSA.ssa_reconstruct(ssa_info, modes, sum_modes=true)
x_normal = SSA.ssa_reconstruct(ssa_info_n, modes, sum_modes=true)
ssa_info.X = Xp
x_cp = SSA.ssa_reconstruct(ssa_info, modes, sum_modes=true)

display(plot([x_normal[1:M, 1], x_cp[M:M+M, 1], x_true[M:M+M, 1]],
             labels=["Normal" "Conditional prediction" "Truth"]))

println("Error normal: ", sqrt.(mean((x_normal[1:M, :] - x_true[M:M+M-1, :]).^2)))
println("Error CP: ", sqrt.(mean((x_cp[M:M+M-1, :] - x_true[M:M+M-1, :]).^2)))
