module SSA

export ssa_eigen, ssa_reconstruct, transform, project, transform1

using LinearAlgebra
using Statistics
using Distributions
using Distributed

using NearestNeighbors

struct SSA_Info
   EW::Array{<:AbstractFloat, 1}
   EV::Array{<:AbstractFloat, 2}
   X::Array{<:AbstractFloat, 2}
   C::Array{<:AbstractFloat, 2}
   N::Integer
   D::Integer
   K::Integer
   SSA_Info(;EW, EV, X, C, N, D, K) = new(EW, EV, X, C, N, D, K)
end

"""
Singular spectrum analysis eigendecomposition with the Broomhead–King approach
"""
function ssa_eigen(x::Array{float_type, dim},
                   M::Integer) where {float_type<:AbstractFloat} where dim
   if (dim == 1)
      # Single-channel SSA
      N = length(x)
      D = 1
      K = 1
   elseif (dim == 2)
      # Multi-channel SSA
      N, D = size(x)
      K = 1
   elseif (dim == 3)
      # Multi-channel SSA with multiple non-contiguous samples of a series
      N, D, K = size(x)
   else
      throw(ArgumentError("x must be of 1, 2, or 3 dimensions"))
   end

   if M > N
      throw(ArgumentError("M cannot be greater than N"))
   end

   N′ = N-M+1
   X = zeros(float_type, N′*K, D*M)

   for k = 1:K
      for d = 1:D
         for i = 1:N′
             X[(k-1)*N′+i, 1+M*(d-1):M*d] = x[i:i+M-1, d, k_i]
         end
      end
   end

   if N′*k >= D*M
      C = X'*X/(N′*K)
      EW, EV = eigen(C)
   else
      # Use PCA transpose trick; see section A2 of Ghil et al. (2002)
      C = X*X'/(N′*K)
      EW, EV = eigen(C)

      EV = X'*EV

      # Normalize eigenvectors
      EV = EV./mapslices(norm, EV, dims=1)
   end

   EW = reverse(EW)
   EV = reverse(EV, dims=2)

   return SSA_Info(EW=EW, EV=EV, X=X, C=C, N=N, D=D, K=K)
end

function ssa_reconstruct(X::Array{float_type, 2}, EV::Array{float_type, 2},
                         M::Integer, D::Integer,
                         ks) where {float_type<:AbstractFloat}
   N = size(X)[1] + M - 1
   A = X*EV
   R = SharedArray{float_type, 3}((length(ks), N, D))

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
end
