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
modes = 1:4
#pcs = 1:6

y = integrator(model, randn(D), 0., record_length - Δt, Δt, inplace=false)[transient:outfreq:end, :]
#y_err = integrator(model_err, randn(D), 0., record_length - Δt, Δt, inplace=false)[transient:outfreq:end, :]

EW, EV, X, C = Embedding.mssa(copy(y)[1:end, :], M)
#EW, EV, X_err, C_err = Embedding.mssa(copy(y_err)[1:end, :], M)

r = sum(Embedding.reconstruct(X, EV, M, D, modes), dims=1)[1, :, :]
#r_err = sum(Embedding.reconstruct(X_err, EV, M, D, modes), dims=1)[1, :, :]

#C_conds = Embedding.precomp(C, M, D, 'b')
#C_conds_err = Embedding.precomp(C_err, M, D, 'b')

function find_point(r, tree, p, k, f)
    ind, dist = knn(tree, p, k)
    mask = (ind .+ f) .<= size(tree.data)[1]
    dist = dist[mask]
    ind = ind[mask]
    return sum(dist .* r[1000 .+ ind .- 1 .+ f, :], dims=1)/sum(dist)
end

function find_point2(model, p, C_conds)
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
tree_err = KDTree(copy((y_err[1000:end, :])'))

tree_r = KDTree(copy((r[1000:end, :])'))
tree_r_err = KDTree(copy((r_err[1000:end, :])'))

#errs = [[norm(find_point(r, tree, y[i, :], k, 0)' - r[i, :]) for k=1:50] for i=1:100]
#errs2 = [norm(find_point2(y[i, :], C_conds) - r[i, :]) for i=1:100]

A = sum(Embedding.obs_operator(EV, M, D, i) for i in modes)
B = pinv(A')

lead = 60
f_errs = []
f_errs_corr = []
f_errs_null = []
# bias = mean(y - y_err, dims=1)
for i_p=30:130
    p = y[i_p, :]
    p2 = find_point(r, tree_r, p, 41, 0)
    #p2 = find_point2(p, C_conds_err)
    forecast = find_point(r, tree_r, p2[:], 41, lead)

    #p2_err = find_point2(model_err, p, C_conds_err)
    #forecast_err = find_point(r_err, tree_r_err, p2_err[:], 41, lead)

    pred = integrator(model, p, 0., outfreq*Δt*lead, Δt, inplace=true)
    pred_err = integrator(model_err, p, 0., outfreq*Δt*lead, Δt, inplace=true)

    err = sqrt.(mean((pred_err - pred).^2))
    append!(f_errs, err)

    err_corr = sqrt.(mean(((0.5*(pred_err) + 0.5*reshape(B*(forecast'), :, D)[M, :]) - pred).^2))
    append!(f_errs_corr, err_corr)

    err_null = sqrt.(mean((0.5*(pred_err) + 0.5*mean(y, dims=1)' - pred).^2))
    append!(f_errs_null, err_null)
end
# f_errs = reshape(f_errs, 51, :)
# r_norm = (r .- mean(r, dims=1)) ./ std(r, dims=1)
# _, s, v = svd(r)
# v_sel = v[:, pcs]
# n1 = std(r*v_sel, dims=1)
# n2 = std(y*v_sel, dims=1)
# tree2 = KDTree(copy((r[1000:end, :]*v_sel ./ n1)'))
# errs2 = [[norm(find_point2(r, tree2, y[i, :], k, 0)' - r[i, :]) for k=1:50] for i=1:999]

# f_errs = []
# f_errsm = []
# for i_p=30:100
#     p = y[i_p, :]
#     p2 = find_point(r, tree, p, 41, 0)
#     #p2 = find_point2(model, p, C_conds)
#     forecast = [find_point(r, tree_r, p2[:], 41, i) for i=0:50]
#     err = sqrt.(mean((vcat(forecast...) - r[i_p:i_p + 50, :]).^2, dims=2))
#     errm = sqrt.(mean((mean(r, dims=1) .- r[i_p:i_p + 50, :]).^2, dims=2))
#     append!(f_errs, err)
#     append!(f_errsm, errm)
# end
# f_errs = reshape(f_errs, 51, :)
# f_errsm = reshape(f_errsm, 51, :)
