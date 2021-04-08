module Analog

export find_point, error_cov

using Statistics

using NearestNeighbors

function find_point(r, tree, p, k, f)
    ind, dist = knn(tree, p[:], k)
    mask = (ind .+ f) .<= size(r)[1]
    dist = 1 ./ dist[mask]
    ind = ind[mask]
    return sum(dist .* r[ind .+ f, :], dims=1)/sum(dist)
end

function error_cov(y, r, M, window, k, k_r, osc_vars; validation_pct=0.1)
    validation = round(Int, validation_pct*size(y)[1])
    tree = KDTree(copy((y[validation+1:end, :])'))
    tree_r = KDTree(copy((r[validation+1:end, :])'))
    errs = Array{Float64}(undef, length(osc_vars), length(M:validation-window))

    for (i, i_p) in enumerate(M:validation-window)
        p = y[i_p, :]
        p_r = find_point(r, tree, p, k, validation)

        forecast = find_point(r, tree_r, p_r[:], k_r, validation + window)
        err = r[i_p + window, :] - forecast'

        errs[:, i] = err
    end

    R = cov(errs')
    return R
end

function error_corr(y, r, M, window, k, k_r, osc_vars; validation_pct=0.1)
    validation = round(Int, validation_pct*size(y)[1])
    tree = KDTree(copy((y[validation+1:end, osc_vars])'))
    tree_r = KDTree(copy((r[validation+1:end, :])'))
    corrs = Array{Float64}(undef, length(M:validation-window))
    errs_m = Array{Float64}(undef, length(M:validation-window))
    m = mean(r, dims=1)

    for (i, i_p) in enumerate(M:validation-window)
        p = y[i_p, osc_vars]
        p_r = find_point(r, tree, p, k, validation)

        forecast = find_point(r, tree_r, p_r[:], k_r, validation + window)

        corrs[i] = sqrt(mean((r[i_p + window, :] - (forecast')[:]).^2))
        errs_m[i] = sqrt(mean((r[i_p + window, :] - m').^2))
    end

    return mean(corrs), mean(errs_m)
end

end
