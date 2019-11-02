using Plots
using Statistics

include("embedding.jl")
include("models.jl")
include("integrators.jl")

using .Embedding
using .Models
using .Integrators

model = Models.colpitts_true
Δt = 0.1
M = 30
D = 9
outfreq = 4
modes = 1:6

y = Integrators.rk4(model, randn(9), 0., 1550.0 - Δt, Δt, inplace=false)[500:outfreq:end, :]

EW, EV, X, C = Embedding.mssa(copy(y)[1:end, :], M)
C_conds = Embedding.precomp(C, M, D, 'f')
Xp = Embedding.transform_cp(copy(y)[1:end-29, :], M, 'f', C_conds)
heatmap(reverse(Xp[end-100:end, :], dims=2)', clim=(-15, 15))
EWn, EVn, Xn = Embedding.mssa(copy(y)[1:end-29, :], M)

x_true = sum(Embedding.reconstruct(copy(X), EV, M, D, modes), dims=1)[1, :, :]
x_normal = sum(Embedding.reconstruct(copy(Xn), EV, M, D, modes), dims=1)[1, :, :]
x_cp = sum(Embedding.reconstruct(copy(Xp), EV, M, D, modes), dims=1)[1, :, :]

plot([x_normal[3721-29:end, 1], x_cp[3721-29:end, 1], x_true[3721-29:end, 1]],
     labels=["Normal", "CP", "Truth"])

println("Error normal: ", sqrt.(mean((x_normal[3721-29:end, :] - x_true[3721-29:3721, :]).^2)))
println("Error CP: ", sqrt.(mean((x_cp[3721-29:3721, :] - x_true[3721-29:3721, :]).^2)))
