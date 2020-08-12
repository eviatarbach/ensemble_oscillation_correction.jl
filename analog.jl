module Analog

export find_point, error_cov

function find_point(r, tree, p, k, f)
    ind, dist = knn(tree, p, k)
    mask = (ind .+ f) .<= size(tree.data)[1]
    dist = 1 ./ dist[mask]
    ind = ind[mask]
    return sum(dist .* r[validation .+ ind .- 1 .+ f, :], dims=1)/sum(dist)
end

function find_point(r, tree, p, k, f)
    ind, dist = knn(tree, p[:], k)
    mask = (ind .+ f) .<= size(tree.data)[1]
    dist = 1 ./ dist[mask]
    ind = ind[mask]
    return sum(dist .* r[ind .+ f, :], dims=1)/sum(dist)
end

function error_cov(r, y, tree, tree_r, M)
    validation = round(Int, 0.1*size(y)[1])
    tree = KDTree(copy((y[validation:end, :])'))
    tree_r = KDTree(copy((r[validation:end, :])'))
    errs = Array{Float64}(undef, length(osc_vars), length(M:validation-window))

    for (i, i_p) in enumerate(M:validation-window)
        p = y[i_p, :]
        p2 = find_point(r, tree, p, k, 0)

        forecast = find_point(r, tree_r, p2[:], k_r, window)
        err = r[i_p + window, :] - forecast'

        errs[:, i] = err
    end

    R = cov(errs')
    return R
end

end
