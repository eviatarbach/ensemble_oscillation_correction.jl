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
