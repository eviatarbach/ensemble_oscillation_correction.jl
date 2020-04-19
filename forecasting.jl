include("models.jl")
include("integrators.jl")
include("embedding.jl")
using .Models
using .Integrators
using .Embedding

using LinearAlgebra
using Statistics

using NearestNeighbors

D = 6
model = Models.lorenz96_true
model_err = Models.lorenz96_err
integrator = Integrators.rk4
record_length = 10000.0
outfreq = 5
Δt = 0.1
transient = 2000
M = 100
modes = 1:2
k = 1
k_r = 10
#pcs = 1:6

y0 = [0.723667, 0.101699, 0.0719784, 0.923862, 0.321385, 0.579979]#[0.7, 0, 0]

y = integrator(model, y0, 0., record_length - Δt, Δt, inplace=false)[transient:outfreq:end, :]
#y_err = integrator(model_err, randn(D), 0., record_length - Δt, Δt, inplace=false)[transient:outfreq:end, :]

EW, EV, X, C = Embedding.mssa(copy(y)[1:end, :], M)
#EW, EV = Embedding.var_rotate!(EW, EV, M, D, 6)
#EW, EV, X_err, C_err = Embedding.mssa(copy(y_err)[1:end, :], M)

r = sum(Embedding.reconstruct(X, EV, M, D, modes), dims=1)[1, :, :]
#r_err = sum(Embedding.reconstruct(X_err, EV, M, D, modes), dims=1)[1, :, :]

C_conds_b = Embedding.precomp(C, M, D, 'b')
C_conds_f = Embedding.precomp(C, M, D, 'f')
#C_conds_err = Embedding.precomp(C, M, D, 'b')

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

function find_point3(past, C_conds)
    Xp = Embedding.transform_cp(past, M, 'f', C_conds)
    pred = sum(Embedding.reconstruct(Xp, EV, M, D, modes), dims=1)[1, :, :]
    return pred
end

function find_point2(future, C_conds)
    Xp = Embedding.transform_cp(past, M, 'b', C_conds)
    pred = sum(Embedding.reconstruct(Xp, EV, M, D, modes), dims=1)[1, :, :]
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

f_errs = []
f_errsp = []
f_errsmo = []
f_errscp = []
max_lead = 200
for i_p=M:10:1000-max_lead
    p = y[i_p, :]
    past = y[i_p - (M-1):i_p, :]
    #p2 = find_point3(past, C_conds_f)[M, :]'
    p2 = find_point(r, tree, p, k, 0)
    #p2 = find_point2(model, p, C_conds_b)

    forecast = [find_point(r, tree_r, p2[:], k_r, i) for i=1:max_lead]
    err = sqrt.(mean((vcat(p2, vcat(forecast...)) - r[i_p:i_p + max_lead, :]).^2, dims=2))

    errp = sqrt.(mean((p2 .- r[i_p:i_p + max_lead, :]).^2, dims=2))

    forecast_model = vcat(p', integrator(model_err, p, 0., outfreq*Δt*max_lead, Δt, inplace=false))[1:outfreq:end, :]
    Xp = Embedding.transform_cp(forecast_model, M, 'b', C_conds_b)
    x_cp = sum(Embedding.reconstruct(copy(Xp), EV, M, D, modes), dims=1)[1, :, :]
    errmo = sqrt.(mean((x_cp[M:end, :] .- r[i_p:i_p + max_lead, :]).^2, dims=2))

    forecast_cp = find_point3(past, C_conds_f)[M:end, :]
    f2 = [forecast_cp[i, :] for i in 1:M]
    err_cp = sqrt.(mean((vcat(f2'...) .- r[i_p:i_p + (M - 1), :]).^2, dims=2))

    append!(f_errs, err)
    append!(f_errsp, errp)
    append!(f_errsmo, errmo)
    append!(f_errscp, err_cp)
end
f_errs = reshape(f_errs, max_lead + 1, :)
f_errsp = reshape(f_errsp, max_lead + 1, :)
f_errsmo = reshape(f_errsmo, max_lead + 1, :)
f_errscp = reshape(f_errscp, M, :)

plot([mean(f_errs, dims=2), mean(f_errsp, dims=2), mean(f_errsmo, dims=2),
      mean(f_errscp, dims=2)],
      labels=["NN", "Persistence", "Model with error", "CP"])
