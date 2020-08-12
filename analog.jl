module Analog

export find_point, error_cov

using NearestNeighbors

function find_point(r, tree, p, k, f)
    ind, dist = knn(tree, p[:], k)
    mask = (ind .+ f) .<= size(tree.data)[1]
    dist = 1 ./ dist[mask]
    ind = ind[mask]
    return sum(dist .* r[ind .+ f, :], dims=1)/sum(dist)
end

function error_cov(y, r, M, window, k, k_r; validation_pct=0.1)
    validation = round(Int, validation_pct*size(y)[1])
    tree = KDTree(copy((y[validation:end, :])'))
    tree_r = KDTree(copy((r[validation:end, :])'))
    errs = Array{Float64}(undef, length(osc_vars), length(M:validation-window))

    for (i, i_p) in enumerate(M:validation-window)
        p = y[i_p, :]
        p_r = find_point(r, tree, p, k, validation + 1)

        forecast = find_point(r, tree_r, p_r[:], k_r, validation + window + 1)
        err = r[i_p + window, :] - forecast'

        errs[:, i] = err
    end

    R = cov(errs')
    return R
end

end
