module Embedding

export mssa, reconstruct, transform, project, obs_operator, transform1

using LinearAlgebra
using ToeplitzMatrices
using NearestNeighbors
using Statistics

function mssa(x::Array{Float64, 2}, M::Int64)
   N, D = size(x)

   idx = Hankel([float(i) for i in 1:N-M+1], [float(i) for i in N-M+1:N])
   idx = round.(Int, idx)
   xtde = zeros(N-M+1, M, D)

   for d=1:D
      xtde[:, :, d] = x[:, d][idx]
   end
   xtde = reshape(xtde, N-M+1, D*M, 1)[:, :, 1]

   C = xtde'*xtde/(N-M+1)
   EW, EV = eigen(C)

   EW = reverse(EW)
   EV = reverse(EV, dims=2)

   A = xtde*EV

   return EW, EV, xtde
end

function varimax(A::Array{Float64, 3}, reltol=sqrt(eps(Float64)),
                 maxit=1000, normalize=true, G=[])
   M, D, S = size(A)
   B = A

   B = reshape(B, D*M, S, 1)[:, :, 1]
   D = D*M

   T = diagm(0=>ones(S))

   for iter=1:maxit
      maxTheta = 0
      for i = 1:(S-1)
         for j = (i+1):S
            u = B[:, i].^2 - B[:,j].^2
            v = 2*B[:,i].*B[:,j]
            if M>1
               u = sum(reshape(u, M, :), dims=1)'
               v = sum(reshape(v, M, :), dims=1)'
            end

            usum = sum(u, dims=1);
            vsum = sum(v, dims=1);
            numer = 2*u'*v - 2*usum*vsum/D;
            denom = u'*u - v'*v - (usum^2 - vsum^2)/D;
            theta = atan(numer[1, 1], denom[1, 1])/4;
            maxTheta = max(maxTheta, abs(theta));
            Tij = [cos(theta) -sin(theta); sin(theta) cos(theta)];
            B[:, [i, j]] = B[:, [i, j]] * Tij;
            T[:, [i, j]] = T[:, [i, j]] * Tij;
         end
      end
      if (maxTheta < reltol)
         break
      end
   end

   return B, T
end

function var_rotate!(EW, EV, M, D, S)
   EVscaled=EV[:,1:S]*diagm(0=>sqrt.(EW[1:S]))
   EVreshape=reshape(EVscaled,M,D,S)

   _, T = varimax(EVreshape)

   EV[:,1:S]=EV[:,1:S]*T
   EW[1:S]=diag(T'*diagm(0=>EW[1:S])*T)

   EV = EV[:, sortperm(EW, rev=true)]
   EW = sort(EW, rev=true)

   return EW, EV
end

function reconstruct(X::Array{Float64, 2}, EV::Array{Float64, 2}, M::Int64,
                     D::Int64, ks)
   N = size(X)[1] + M - 1
   A = X*EV
   R = zeros(length(ks), N, D)

   for (ik, k) in enumerate(ks)
      ek = reshape(EV[:, k], M, D)
      for n=1:N
         if 1 <= n <= M - 1
            M_n = n
            L_n = 1
            U_n = n
         elseif M <= n <= N - M + 1
            M_n = M
            L_n = 1
            U_n = M
         elseif N - M + 2 <= n <= N
            M_n = N - n + 1
            L_n = n - N + M
            U_n = M
         end
         for d=1:D
            R[ik, n, d] = 1/M_n*sum([A[n - m + 1, k]*ek[m, d] for m=L_n:U_n])
         end
      end
   end
   return R
end

function transform(x, n, EV::Array{Float64, 2}, M, D, ks)
   R = zeros(length(ks), D)

   for (ik, k) in enumerate(ks)
      ek = reshape(EV[:, k], M, D)
      for d=1:D
         R[ik, d] = 1/M*sum([sum([sum([x[n - m + mp, dp]*ek[mp, dp] for mp=1:M]) for dp=1:D])*ek[m, d] for m=1:M])
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

function obs_operator(EV, M, D, k)
   n = (M - 1)*2 + 1
   A = diagm(0 => ones(n*D))
   return vcat([transform(reshape(A[:, i], n, D), M, EV, M, D, k) for i=1:n*D]...)
end

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

end
