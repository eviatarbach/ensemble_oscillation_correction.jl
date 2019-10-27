include("models.jl")
include("integrators.jl")
include("embedding.jl")
using .Models
using .Integrators
using .Embedding

using LinearAlgebra
using Statistics

using NearestNeighbors

D = 42
model = "true"
model_err = "false"
integrator = Integrators.ks_integrate
record_length = 10000.0
outfreq = 4
Δt = 0.25
transient = 500
M = 30
modes = 2:5
pcs = 1:6

y = integrator(model, randn(D), 0., record_length - Δt, Δt, inplace=false)[transient:outfreq:end, :]
EW, EV, X, C = Embedding.mssa(copy(y)[1:end, :], M)
r = sum(Embedding.reconstruct(X, EV, M, D, modes), dims=1)[1, :, :]

C_conds = Embedding.precomp(C, M, D, 'b')

function find_point(r, tree, p, k, f)
    ind, dist = knn(tree, p, k)
    mask = (ind .+ f) .<= size(tree.data)[1]
    dist = dist[mask]
    ind = ind[mask]
    return sum(dist .* r[1000 .+ ind .- 1 .+ f, :], dims=1)/sum(dist)
end

function find_point2(p, C_conds)
    future = vcat(p', integrator(model, p, 0.0, outfreq*Δt*(M-1), Δt, inplace=false))[1:outfreq:end, :]
    pred = sum(Embedding.reconstruct_cp(Embedding.transform_cp(future, M, 'b', C_conds), EV, M, D, modes), dims=1)[1, :]
    return pred
end

# function find_point2(r, tree, p, k, f)
#     ind, dist = knn(tree, (p'*v_sel ./ n2)', k)
#     ind = ind[1]
#     dist = dist[1]
#     mask = (ind .+ f) .<= size(tree.data)[1]
#     dist = dist[mask]
#     ind = ind[mask]
#     return sum(dist .* r[1000 .+ ind .- 1 .+ f, :], dims=1)/sum(dist)
# end

tree = KDTree(copy((y[1000:end, :])'))
tree_r = KDTree(copy((r[1000:end, :])'))

errs = [[norm(find_point(r, tree, y[i, :], k, 0)' - r[i, :]) for k=1:50] for i=1:100]
errs2 = [norm(find_point2(y[i, :], C_conds) - r[i, :]) for i=1:100]

f_errs = []
for i_p=30:200
    p = y[i_p, :]
    p2 = find_point2(p, C_conds)
    forecast = [find_point(r, tree_r, p2[:], 41, i) for i=0:50]
    err = sqrt.(mean((vcat(forecast...) - r[i_p:i_p + 50, :]).^2, dims=2))
    append!(f_errs, err)
end
f_errs = reshape(f_errs, 51, :)
# r_norm = (r .- mean(r, dims=1)) ./ std(r, dims=1)
# _, s, v = svd(r)
# v_sel = v[:, pcs]
# n1 = std(r*v_sel, dims=1)
# n2 = std(y*v_sel, dims=1)
# tree2 = KDTree(copy((r[1000:end, :]*v_sel ./ n1)'))
# errs2 = [[norm(find_point2(r, tree2, y[i, :], k, 0)' - r[i, :]) for k=1:50] for i=1:999]
