function estimate_errs(osc, tree, data, pcs, max_k=30)
   N, D = size(data)

   errs = []
   for k=2:max_k
      errsk = []
      for i=1:100
         ii = rand(1:N-1)
         append!(errsk, (osc[ii+1, :] - project(tree, (data[ii, :]'*pcs)', osc, k, 1)[2, :]).^2)
      end
      append!(errs, mean(reshape(errsk, D, 100), dims=2))
   end
   return reshape(errs, D, :)'
end

function create_tree(; model, Δt, outfreq, obs_err_pct, M, record_length, transient, u0, D,
                     osc_vars, modes, integrator, pcs=nothing, varimax, brownian_noise,
                     da, window, k, k_r)
   y = integrator(model, u0, 0., record_length*outfreq*Δt, Δt; inplace=false)[1:outfreq:end, :][(transient + 1):end, :]
   if (obs_err_pct > 0)
      R = Symmetric(diagm(0 => obs_err_pct*std(y, dims=1)[1, :]))
      obs_err = MvNormal(zeros(D), R)
      y = y + rand(obs_err, size(y)[1])'
   end

#   if (brownian_noise != false)
#      for i=1:D
#         W = WienerProcess(0.0, 0.0)
#         prob = NoiseProblem(W, (0.0, size(y)[1]*brownian_noise))
#         sol = solve(prob; dt=brownian_noise)
#         y[:, i] = y[:, i] + std(y, dims=1)[1, i]*sol.u[1:size(y)[1]]
#      end
#   end

   EW, EV, X, C = Embedding.mssa(y[:, osc_vars], M)

   if varimax
      EW, EV = Embedding.var_rotate!(EW, EV, M, D, 20)
   end

   r_all = Embedding.reconstruct(X, EV, M, length(osc_vars), modes)
   r = sum(r_all, dims=1)[1, :, :]

   if da
      function find_point(r, tree, p, k, f)
          ind, dist = knn(tree, p, k)
          mask = (ind .+ f) .<= size(tree.data)[1]
          dist = dist[mask]
          ind = ind[mask]
          return sum(dist .* r[validation .+ ind .- 1 .+ f, :], dims=1)/sum(dist)
      end
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
   else
      R = false
   end

   if pcs == nothing
      tree = KDTree(copy(y'))
      tree_r = KDTree(copy(r'))
   else
      _, _, v = svd(r)
      v = v[:, 1:pcs]
      tree = KDTree(copy(y'))
   end

   return tree, tree_r, EW, EV, y, r, C, R
end

function transform(x, n, EV::Array{Float64, 2}, M, D, ks)
   R = zeros(length(ks), D)

   for (ik, k) in enumerate(ks)
      ek = reshape(EV[:, k], M, D)
      for m=1:M
         inner_sum = sum([sum([x[n - m + mp, dp]*ek[mp, dp] for mp=1:M]) for dp=1:D])
         for d=1:D
            R[ik, d] += 1/M*inner_sum*ek[m, d]
         end
      end
   end
   return R
end

function transform1(x, EV::Array{Float64, 2}, M, D, ks)
   R = zeros(length(ks), D)

   for (ik, k) in enumerate(ks)
      ek = reshape(EV[:, k], M, D)
      for d=1:D
         R[ik, d] = sum([sum([x[mp, dp]*ek[mp, dp] for mp=1:M]) for dp=1:D])*ek[1, d]
      end
   end
   return R
end

function project(tree, point, osc, k, N)
   idx, dists = knn(tree, point, k)
   len = size(osc)[1]
   mask = (idx .<= (len - N)) .& (dists .> 0)
   idx = idx[mask]
   dists = dists[mask]

   return sum((1 ./ dists).*[osc[id:id+N, :] for id in idx])/sum(1 ./ dists)
end
