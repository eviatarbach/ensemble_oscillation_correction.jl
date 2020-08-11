module SSA

export ssa_decompose, ssa_reconstruct

using LinearAlgebra
using Distributed
using SharedArrays

struct SSA_Info
   eig_vals::Array{<:AbstractFloat, 1}
   eig_vecs::Array{<:AbstractFloat, 2}
   X::Array{<:AbstractFloat, 2}
   C::Array{<:AbstractFloat, 2}
   N::Integer
   D::Integer
   J::Integer
   M::Integer
   SSA_Info(;eig_vals, eig_vecs, X,
            C=nothing, N, D, J, M) = new(eig_vals, eig_vecs, X, C, N, D, J, M)
end

"""
Singular spectrum analysis eigendecomposition with the Broomhead–King approach
"""
function ssa_decompose(x::Array{float_type, dim},
                       M::Integer) where {float_type<:AbstractFloat} where dim
   if (dim == 1)
      # Single-channel SSA
      N = length(x)
      D = 1
      J = 1
   elseif (dim == 2)
      # Multi-channel SSA
      N, D = size(x)
      J = 1
   elseif (dim == 3)
      # Multi-channel SSA with multiple non-contiguous samples of a series
      N, D, J = size(x)
   else
      throw(ArgumentError("x must be of 1, 2, or 3 dimensions"))
   end

   if M > N
      throw(ArgumentError("M cannot be greater than N"))
   end

   N′ = N-M+1
   X = zeros(float_type, N′*J, D*M)

   for j = 1:J
      for d = 1:D
         for i = 1:N′
             X[(j-1)*N′+i, 1+M*(d-1):M*d] = x[i:i+M-1, d, j]
         end
      end
   end

   if N′*J >= D*M
      C = X'*X/(N′*J)
      eig_vals, eig_vecs = eigen(C)
   else
      # Use PCA transpose trick; see section A2 of Ghil et al. (2002)
      C = X*X'/(N′*J)
      eig_vals, eig_vecs = eigen(C)

      EV = X'*eig_vecs

      # Normalize eigenvectors
      eig_vecs = eig_vecs./mapslices(norm, eig_vecs, dims=1)
   end

   eig_vals = reverse(eig_vals)
   eig_vecs = reverse(eig_vecs, dims=2)

   return SSA_Info(eig_vals=eig_vals, eig_vecs=eig_vecs, X=X, C=C, N=N, D=D,
                   J=J, M=M)
end

"""
Reconstruct the specified modes
"""
function ssa_reconstruct(ssa_info, modes; sum_modes=false)
   eig_vecs = ssa_info.eig_vecs
   X = ssa_info.X
   M = ssa_info.M
   N = ssa_info.N
   D = ssa_info.D
   J = ssa_info.J
   R = SharedArray{eltype(X), 4}((length(modes), N, D, J))

   for (i_k, k) in enumerate(modes)
      ek = reshape(eig_vecs[:, k], M, D)
      for j = 1:J
         A = X[1+(j-1)*(N-M+1):j*(N-M+1), :]*eig_vecs
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
               R[i_k, n, d, j] = 1/M_n*sum([A[n - m + 1, k]*ek[m, d] for m=L_n:U_n])
            end
         end
      end
   end

   if sum_modes
      R = sum(R, dims=1)[1, :, :, :]
   end

   return R
end
end
