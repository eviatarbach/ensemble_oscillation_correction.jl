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

N = size(y)[1]
start_i = N - M + 1

ssa_info = SSA.ssa_decompose(copy(y)[1:end, :], M)
C_conds = SSA_CP.precomp(ssa_info.C, M, D, 'f')
Xp = SSA_CP.transform(copy(y)[1:end-M+1, :], M, 'f', C_conds)
display(heatmap(reverse(Xp[end-100:end, :], dims=2)', clim=(-15, 15)))
ssa_info_n = SSA.ssa_decompose(copy(y)[1:end-M+1, :], M)

x_true = SSA.ssa_reconstruct(ssa_info, modes, sum_modes=true)
x_normal = SSA.ssa_reconstruct(ssa_info_n, modes, sum_modes=true)
ssa_info.X = Xp
x_cp = SSA.ssa_reconstruct(ssa_info, modes, sum_modes=true)

display(plot([x_normal[start_i-M+1:end, 1], x_cp[start_i-M+1:end, 1],
              x_true[start_i-M+1:end, 1]],
              labels=["Normal" "Conditional prediction" "Truth"]))

println("Error normal: ", sqrt.(mean((x_normal[start_i-M+1:end, :] - x_true[start_i-M+1:start_i, :]).^2)))
println("Error CP: ", sqrt.(mean((x_cp[start_i-M+1:start_i, :] - x_true[start_i-M+1:start_i, :]).^2)))
