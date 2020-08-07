module Embedding

export ssa, reconstruct, transform, project, obs_operator, transform1, obs_operator1

using LinearAlgebra
using Statistics
using Distributions
using Distributed

using NearestNeighbors

"""
Singular spectrum analysis with the Broomhead–King approach
"""
function ssa(x::Array{T, dim}, M::Integer) where {T<:AbstractFloat} where dim
   if (dim == 1)
      # Single-channel SSA
      N = length(x)
      D = 1
      k = 1
   elseif (dim == 2)
      # Multi-channel SSA
      N, D = size(x)
      k = 1
   elseif (dim == 3)
      # Multi-channel SSA with multiple non-contiguous samples of a series
      N, D, k = size(x)
   else
      throw(ArgumentError("x must be of 1, 2, or 3 dimensions"))
   end

   N′ = N-M+1
   xtde = zeros(N′*k, D*M)

   for k_i = 1:k
      for d = 1:D
         for i = 1:N′
             xtde[(k_i-1)*N′+i, 1+M*(d-1):M*d] = x[i:i+M-1, d, k_i]
         end
      end
   end

   if N′*k >= D*M
      C = xtde'*xtde/(N′*k)
      EW, EV = eigen(C)
   else
      # Use PCA transpose trick; see section A2 of Ghil et al. (2002)
      C = xtde*xtde'/(N′*k)
      EW, EV = eigen(C)

      EV = xtde'*EV

      # Normalize eigenvectors
      EV = EV./mapslices(norm, EV, dims=1)
   end

   EW = reverse(EW)
   EV = reverse(EV, dims=2)

   return EW, EV, xtde
end

function reconstruct(xtde::Array{T, 2}, EV::Array{T, 2}, M::Integer,
                     D::Integer, ks) where {T<:AbstractFloat}
   N = size(X)[1] + M - 1
   A = xtde*EV
   R = SharedArray{T, 3}((length(ks), N, D))

   for (ik, k) in enumerate(ks)
      ek = reshape(EV[:, k], M, D)
      @sync @distributed for n = 1:N
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

function precomp(C, M, D, mode)
   C_conds = Array{Array{Float64, 2}, 1}()

   for offset=1:M-1

      if mode == 'f'
         C_11 = reshape(reshape(C, M, D, M, D)[1:end-offset, :, 1:end-offset, :], (M - offset)*D, (M - offset)*D)
         C_21 = reshape(reshape(C, M, D, M, D)[end-offset+1:end, :, 1:end-offset, :], :, (M - offset)*D)
      elseif mode == 'b'
         C_11 = reshape(reshape(C, M, D, M, D)[offset+1:end, :, offset+1:end, :], (M - offset)*D, (M - offset)*D)
         C_21 = reshape(reshape(C, M, D, M, D)[1:offset, :, offset+1:end, :], :, (M - offset)*D)
      end

      C_11 += 1e-8*I
      C_11_inv = inv(C_11)

      push!(C_conds, C_21*C_11_inv)
   end
   return C_conds
end

"""
Varimax rotation as per
   Groth, A. and M. Ghil, 2011: Multivariate singular spectrum analysis and the
      road to phase synchronization, Physical Review E, 84, 036206.
   Groth, A. and M. Ghil, 2015: Monte Carlo Singular Spectrum Analysis (SSA)
      revisited: Detecting oscillator clusters in multivariate datasets, Journal
      of Climate, 28, 7873-7893.

Code based on Matlab implementation by Andreas Groth at
https://www.mathworks.com/matlabcentral/fileexchange/65939-multichannel-singular-spectrum-analysis-varimax-tutorial

Copyright (c) 2018, Andreas Groth
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
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

function varimax_rotate!(EW, EV, M, D, S)
   EVscaled=EV[:,1:S]*diagm(0=>sqrt.(EW[1:S]))
   EVreshape=reshape(EVscaled,M,D,S)

   _, T = varimax(EVreshape)

   EV[:,1:S]=EV[:,1:S]*T
   EW[1:S]=diag(T'*diagm(0=>EW[1:S])*T)

   EV = EV[:, sortperm(EW, rev=true)]
   EW = sort(EW, rev=true)

   return EW, EV
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

function transform_cp(x::Array{Float64, 2}, M::Int64, mode, C_conds)
   # Method from "Singular Spectrum Analysis With Conditional Predictions for
   # Real-Time State Estimation and Forecasting", Ogrosky et al. (2019)

   N, D = size(x)

   idx = Hankel([float(i) for i in 1:N-M+1], [float(i) for i in N-M+1:N])
   idx = round.(Int, idx)
   xtde = zeros(N-M+1, M, D)

   for d=1:D
      xtde[:, :, d] = x[:, d][idx]
   end
   xtde = reshape(xtde, N-M+1, D*M, 1)[:, :, 1]

   Xp = zeros(N, M*D)

   if mode == 'f'
      Xp[1:N-M+1, :] = xtde

      xtde_end = reshape(xtde[end, :], M, D)
      indices = (N-M+2):N
   elseif mode == 'b'
      Xp[M:end, :] = xtde

      xtde_end = reshape(xtde[1, :], M, D)
      indices = M-1:-1:1
   end

   Xp = reshape(Xp, N, M, D)

   for k=indices
      if mode == 'f'
         offset = k - (N - M + 1)

         # Fill in upper diagonal with known values
         Xp[k, 1:(end-offset), :] = xtde_end[(offset + 1):end, :]

         C_cond = C_conds[offset]

         # Fill in unknown values
         Xp[k, (end-offset+1):end, :] = C_cond*reshape(Xp[k, 1:end-offset, :], D*(M - offset))
      elseif mode == 'b'
         offset = M - k

         Xp[k, offset+1:end, :] = xtde_end[1:end-offset, :]

         C_cond = C_conds[offset]

         Xp[k, 1:offset, :] = C_cond*reshape(Xp[k, (offset+1):end, :], D*(M - offset))
      end
   end

   Xp = reshape(Xp, N, M*D)

   return Xp
end

function project(tree, point, osc, k, N)
   idx, dists = knn(tree, point, k)
   len = size(osc)[1]
   mask = (idx .<= (len - N)) .& (dists .> 0)
   idx = idx[mask]
   dists = dists[mask]

   return sum((1 ./ dists).*[osc[id:id+N, :] for id in idx])/sum(1 ./ dists)
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
end
